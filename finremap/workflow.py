"""全遊動翼単体の「格子生成 → SU2 Euler 解析 → 表面 Cp の補間と比較」を、設定 JSON 1 つから行う。

  python -m finremap.workflow prepare <設定.json>      実行フォルダと各ケースの設定を作る（計算はしない）
  python -m finremap.workflow mesh <実行フォルダ>       格子を生成する（gmsh が必要）
  python -m finremap.workflow submit <実行フォルダ>     PBS に投入する（--backend local でその計算機で順に実行）
  python -m finremap.workflow status <実行フォルダ>     進み具合
  python -m finremap.workflow analyze <実行フォルダ>    surface_flow.vtu を読み、舵角を戻して補間・比較する

prepare の後は、実行フォルダの中の写し（<実行フォルダ>/scripts）を使う:
  PYTHONPATH=<実行フォルダ>/scripts python -m finremap.workflow submit <実行フォルダ>

SU2 は、execution.container.image があれば podman のイメージの中で、null ならホストで動かす。
1 ケース = 1 ジョブ。全ケースの後に analyze のジョブが走る。始まったケースは自動では再実行しない。
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULES = ["__init__.py", "workflow.py", "mesh.py", "vtu.py", "remap.py", "analysis.py", "parts.py"]

DEFAULTS = dict(
    output_root="runs",
    geometry=dict(root_chord=1.0, tip_chord=0.5, span=0.8, tip_le_x=0.35, thickness_ratio=0.06,
                  edge_radius_ratio=0.005, edge_zone=0.03, hinge_x=0.45),
    domain=dict(x=[-2.0, 5.0], y_max=3.0, z=[-3.0, 3.0], h_far=0.5),
    flow=dict(mach=2.0, pressure=101325.0, temperature=288.15),
    meshes=dict(threads=1, reference=dict(h_fin=0.012, h_edge=0.001, algo2d=6), deflections=[]),
    solver=dict(iter=3000, limiter_iter=400, residual_target=-8.0, cfl=[5.0, 50.0], volume_output=False),
    execution=dict(backend="pbs", ranks=8, su2_cfd="SU2_CFD", mpirun=["mpirun", "-np", "{ranks}"], python="python3",
                   setup=[], case_time_limit_s=None, local_exclusive=True,
                   container=dict(engine="podman", image=None, options=[]),
                   pbs=dict(qsub="qsub", queue=None, resources=["select=1:ncpus=8:mpiprocs=8"], job_prefix="fin",
                            extra_directives=[], analysis_queue=None, analysis_resources=["select=1:ncpus=1"])),
)


class Cancelled(Exception):
    pass


def _cancel(signum, frame):
    raise Cancelled(signum)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def merge(base, extra, where=""):
    out = dict(base)
    for k, v in extra.items():
        require(k in base or where == "", f"設定に未知の項目がある: {where}{k}")
        out[k] = merge(base[k], v, f"{where}{k}.") if isinstance(base.get(k), dict) and isinstance(v, dict) else v
    return out


def deg_tag(deg):
    return f"{deg:02.0f}" if float(deg).is_integer() else f"{deg:04.1f}".replace(".", "p")


# ----------------------------------------------------------------------------- prepare

def cfg_text(plan, aoa):
    flow, sol, geo = plan["flow"], plan["solver"], plan["geometry"]
    area = 0.5 * (geo["root_chord"] + geo["tip_chord"]) * geo["span"]
    files = "( RESTART, SURFACE_PARAVIEW" + (", PARAVIEW )" if sol["volume_output"] else " )")
    return f"""% 全遊動翼単体 Euler。翼は対称面 (y=0) に取り付け。舵軸は y 軸に平行で x={geo['hinge_x']}。
SOLVER= EULER
MATH_PROBLEM= DIRECT
RESTART_SOL= NO
MACH_NUMBER= {flow['mach']}
AOA= {aoa}
SIDESLIP_ANGLE= 0.0
FREESTREAM_PRESSURE= {flow['pressure']}
FREESTREAM_TEMPERATURE= {flow['temperature']}
GAMMA_VALUE= 1.4
GAS_CONSTANT= 287.058
REF_DIMENSIONALIZATION= DIMENSIONAL
REF_ORIGIN_MOMENT_X= {geo['hinge_x']}
REF_ORIGIN_MOMENT_Y= 0.0
REF_ORIGIN_MOMENT_Z= 0.0
REF_LENGTH= 1.0
REF_AREA= {area}
MARKER_EULER= ( fin )
MARKER_SYM= ( sym )
MARKER_FAR= ( far )
MARKER_PLOTTING= ( fin )
MARKER_MONITORING= ( fin )
NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
CONV_NUM_METHOD_FLOW= ROE
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= 0.05
LIMITER_ITER= {sol['limiter_iter']}
TIME_DISCRE_FLOW= EULER_IMPLICIT
CFL_NUMBER= {sol['cfl'][0]}
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.5, 1.2, {sol['cfl'][0]}, {sol['cfl'][1]} )
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1E-4
LINEAR_SOLVER_ITER= 10
MGLEVEL= 0
ITER= {sol['iter']}
CONV_FIELD= RMS_DENSITY
CONV_RESIDUAL_MINVAL= {sol['residual_target']}
CONV_STARTITER= 10
MESH_FILENAME= mesh.su2
MESH_FORMAT= SU2
TABULAR_FORMAT= CSV
CONV_FILENAME= history
RESTART_FILENAME= restart_flow
VOLUME_FILENAME= flow
SURFACE_FILENAME= surface_flow
OUTPUT_WRT_FREQ= {sol['iter']}
OUTPUT_FILES= {files}
VOLUME_OUTPUT= ( COORDINATES, SOLUTION, PRIMITIVE )
HISTORY_OUTPUT= ( ITER, RMS_RES, AERO_COEFF, CFL_NUMBER, LINEAR_SOLVER, NONPHYSICAL_POINTS )
SCREEN_OUTPUT= ( INNER_ITER, RMS_DENSITY, LIFT, DRAG, MOMENT_Y, AVG_CFL )
SCREEN_WRT_FREQ_INNER= 50
"""


def container_record(c):
    """ジョブが使うイメージを ID で固定する。"""
    if not c["image"]:
        return None
    try:
        out = subprocess.run([c["engine"], "image", "inspect", "--format", "{{json .}}", c["image"]],
                             capture_output=True, text=True, timeout=120)
    except OSError as ex:
        raise ValueError(f"コンテナエンジン {c['engine']!r} が見つからない: {ex}")
    require(out.returncode == 0, f"イメージ {c['image']} が {c['engine']} に無い: {out.stderr.strip()}")
    info = json.loads(out.stdout.strip().splitlines()[0])
    info = info[0] if isinstance(info, list) else info
    ver = subprocess.run([c["engine"], "--version"], capture_output=True, text=True, timeout=60).stdout.strip()
    return dict(engine=c["engine"], engine_version=ver, image=c["image"], image_id=info["Id"],
                repo_tags=info.get("RepoTags"), options=c["options"])


def prepare(config_path, skip_image_check=False):
    config_path = Path(config_path).resolve()
    cfg = merge(DEFAULTS, load(config_path))
    name = cfg.get("name", "")
    require(name and all(z.isalnum() or z in "-_." for z in name), "name は英数字と - _ . だけにする")
    require(cfg["meshes"]["deflections"], "meshes.deflections に舵角を 1 つ以上書く")
    run = (config_path.parent / cfg["output_root"] / name).resolve()
    require(not run.exists(), f"{run} は既にある。上書きしない（name を変えるか、フォルダを移す）")

    meshes = {"ref": dict(delta_deg=0.0, **cfg["meshes"]["reference"])}
    cases, order, pairs = {}, [], []
    for d in cfg["meshes"]["deflections"]:
        tag = deg_tag(d["deg"])
        require(d["deg"] != 0 and f"d{tag}" not in meshes, f"舵角 {d['deg']} は使えない（0 または重複）")
        meshes[f"d{tag}"] = dict(delta_deg=float(d["deg"]), **{**cfg["meshes"]["reference"],
                                                               **{k: v for k, v in d.items() if k != "deg"}})
        src, ref = f"d{tag}-a00", f"ref-a{tag}"
        cases[ref] = dict(mesh="ref", aoa_deg=float(d["deg"]), delta_deg=0.0)
        cases[src] = dict(mesh=f"d{tag}", aoa_deg=0.0, delta_deg=float(d["deg"]))
        order += [ref, src]
        pairs.append(dict(source=src, reference=ref, delta_deg=float(d["deg"])))

    plan = dict(name=name, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), config=config_path.name,
                geometry=cfg["geometry"], domain=cfg["domain"], flow=cfg["flow"], solver=cfg["solver"],
                mesh_threads=cfg["meshes"]["threads"], meshes=meshes, cases=cases, order=order, pairs=pairs,
                execution=cfg["execution"])
    c = cfg["execution"]["container"]
    if not c["image"]:
        plan["execution"]["container"] = None
    elif skip_image_check:  # イメージ名のまま実行する（ID での固定なし）
        plan["execution"]["container"] = dict(c, image_id=c["image"], unchecked=True)
    else:
        plan["execution"]["container"] = container_record(c)

    (run / "scripts" / "finremap").mkdir(parents=True)
    for m in MODULES:
        shutil.copy2(HERE / m, run / "scripts" / "finremap" / m)
    plan["scripts"] = {m: sha(run / "scripts" / "finremap" / m) for m in MODULES}
    shutil.copy2(config_path, run / "config.json")
    for cid, case in cases.items():
        (run / "cases" / cid).mkdir(parents=True)
        (run / "cases" / cid / "case.cfg").write_text(cfg_text(plan, case["aoa_deg"]))
    for sub in ("meshes", "pbs", "analysis"):
        (run / sub).mkdir()
    dump(run / "plan.json", plan)
    log(f"実行フォルダ: {run}")
    log(f"格子 {len(meshes)} 個、ケース {len(cases)} 個。次は mesh、その後 submit")
    return run


# ----------------------------------------------------------------------------- mesh

def make_meshes(run):
    from . import mesh as mesh_mod

    run = Path(run).resolve()
    plan = load(run / "plan.json")
    for mid, m in plan["meshes"].items():
        out = run / "meshes" / mid
        if (out / "mesh.json").exists():
            log(f"{mid}: 既にある（作り直さない）")
            continue
        t0 = time.time()
        info = mesh_mod.generate(out, m["delta_deg"], plan["geometry"], plan["domain"], m["h_fin"], m["h_edge"],
                                 m["algo2d"], plan["mesh_threads"])
        info["sha256"] = sha(out / "mesh.su2")
        dump(out / "mesh.json", info)
        log(f"{mid}: 舵角 {m['delta_deg']:g} deg, 翼面 {info['fin_nodes']} 節点, 全体 {info['nodes']} 節点 "
            f"({time.time() - t0:.0f} s)")


# ----------------------------------------------------------------------------- submit

def solver_command(ex, cwd, run, tag):
    """mpirun + SU2_CFD。イメージがあればその中で動かす。(コマンド, コンテナ名) を返す。"""
    inner = [z.replace("{ranks}", str(ex["ranks"])) for z in ex["mpirun"]] + [ex["su2_cfd"], "case.cfg"]
    c = ex.get("container")
    if not c:
        return inner, None
    name = "".join(z if z.isalnum() or z in "_.-" else "_" for z in f"su2-{tag}-{os.getpid()}")
    env = [z for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS") for z in ("-e", f"{k}=1")]
    return [c["engine"], "run", "--rm", "--init", "--name", name, "-v", f"{run}:{run}", "-w", str(cwd), *env,
            *c["options"], c["image_id"], *inner], name


def job_script(plan, run, cid):
    ex = plan["execution"]
    pbs = ex["pbs"]
    analysis = cid is None
    name = pbs["job_prefix"] + ("ANA" if analysis else f"{plan['order'].index(cid) + 1:03d}")
    lines = ["#!/bin/bash", f"#PBS -N {name}", "#PBS -j oe", f"#PBS -o {run / 'pbs' / ((cid or 'analyze') + '.out')}"]
    queue = pbs["analysis_queue"] if analysis and pbs["analysis_queue"] else pbs["queue"]
    if queue:
        lines.append(f"#PBS -q {queue}")
    lines += [f"#PBS -l {r}" for r in (pbs["analysis_resources"] if analysis else pbs["resources"]) or []]
    lines += [f"#PBS {d}" for d in pbs["extra_directives"]]
    lines += [*ex["setup"], "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
              f"export MPLCONFIGDIR={run}/pbs/mpl", f"export PYTHONPATH={run}/scripts", f"cd {run}"]
    cmd = "analyze" if analysis else "run-case"
    lines.append(f"exec {ex['python']} -m finremap.workflow {cmd} {run}" + ("" if analysis else f" {cid}"))
    return "\n".join(lines) + "\n"


def solver_processes():
    try:
        ps = subprocess.check_output(["ps", "-axo", "pid,comm"], text=True).splitlines()[1:]
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(z.split()[0]) for z in ps if z.split()[-1].rsplit("/", 1)[-1] in ("SU2_CFD", "mpirun", "prterun")]


def submit(run, backend=None, only=None, dry_run=False):
    run = Path(run).resolve()
    plan = load(run / "plan.json")
    backend = backend or plan["execution"]["backend"]
    cases = only or plan["order"]
    require(all(c in plan["cases"] for c in cases), "--only に未知のケースがある")
    for c in cases:
        require((run / "meshes" / plan["cases"][c]["mesh"] / "mesh.json").exists(),
                f"{c}: 格子がない。先に mesh を実行する")
        require(not (run / "cases" / c / "state.json").exists(),
                f"{c} は既に始まっている。自動では再投入しない（archive-case で記録を移してからやり直す）")
    ledger_path = run / "pbs" / "jobs.json"
    ledger = load(ledger_path) if ledger_path.exists() else dict(cases={}, analyze=[])
    (run / "pbs" / "mpl").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(run / "scripts"))

    if backend == "local":
        require(not dry_run, "--dry-run は pbs 用")
        if plan["execution"]["local_exclusive"]:
            require(not solver_processes(), "この計算機で別の SU2 / mpirun が動いている。終わってから実行する")
        for c in cases:
            ledger["cases"][c] = dict(backend="local", submitted_unix=time.time())
            dump(ledger_path, ledger)
            with (run / "pbs" / f"{c}.out").open("w") as out:
                rc = subprocess.run([sys.executable, "-m", "finremap.workflow", "run-case", str(run), c],
                                    stdout=out, stderr=subprocess.STDOUT, cwd=run, env=env).returncode
            ledger["cases"][c]["returncode"] = rc
            dump(ledger_path, ledger)
            sp = run / "cases" / c / "state.json"
            log(f"{c}: 終了コード {rc}, {load(sp).get('outcome') if sp.exists() else '未開始（pbs/' + c + '.out を見る）'}")
        with (run / "pbs" / "analyze.out").open("w") as out:
            rc = subprocess.run([sys.executable, "-m", "finremap.workflow", "analyze", str(run)],
                                stdout=out, stderr=subprocess.STDOUT, cwd=run, env=env).returncode
        log(f"analyze: 終了コード {rc}（出力は analysis/、ログは pbs/analyze.out）")
        return

    qsub = plan["execution"]["pbs"]["qsub"]
    for c in cases:
        script = run / "pbs" / f"{c}.sh"
        script.write_text(job_script(plan, run, c))
        if dry_run:
            print(script.read_text())
            cmd, _ = solver_command(plan["execution"], run / "cases" / c, run, f"{plan['name']}-{c}")
            print("# このジョブが実行するソルバーのコマンド:\n# " + " ".join(cmd) + "\n")
            continue
        out = subprocess.run([qsub, str(script)], capture_output=True, text=True, cwd=run)
        require(out.returncode == 0, f"{c}: qsub に失敗: {out.stderr.strip()}")
        jid = out.stdout.strip().splitlines()[-1]
        ledger["cases"][c] = dict(backend="pbs", job_id=jid, script=script.name, submitted_unix=time.time())
        dump(ledger_path, ledger)
        log(f"{c}: {jid}")
    script = run / "pbs" / "analyze.sh"
    script.write_text(job_script(plan, run, None))
    if dry_run:
        print(script.read_text())
        return
    ids = [v["job_id"] for v in ledger["cases"].values() if v.get("job_id")]
    out = subprocess.run([qsub, "-W", "depend=afterany:" + ":".join(ids), str(script)],
                         capture_output=True, text=True, cwd=run)
    require(out.returncode == 0, f"analyze のジョブの qsub に失敗: {out.stderr.strip()}")
    ledger["analyze"].append(dict(job_id=out.stdout.strip().splitlines()[-1], depends_on=ids, submitted_unix=time.time()))
    dump(ledger_path, ledger)
    log(f"analyze のジョブ {ledger['analyze'][-1]['job_id']}（{len(ids)} ケースの後）")


# ----------------------------------------------------------------------------- run-case

def last_history(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    keys = [k.strip().strip('"') for k in rows[0]]
    row = dict(zip(keys, (v.strip() for v in rows[-1])))
    pick = lambda k: float(row[k]) if k in row else None
    return dict(iter=int(float(row["Inner_Iter"])), rms_rho=pick("rms[Rho]"), nonphysical=pick("Nonphysical_Points"),
                CL=pick("CL"), CD=pick("CD"), CFx=pick("CFx"), CFy=pick("CFy"), CFz=pick("CFz"),
                CMx=pick("CMx"), CMy=pick("CMy"), CMz=pick("CMz"))


def link_mesh(source, target):
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        os.symlink(os.path.relpath(source, target.parent), target)
        return "symlink"


def run_case(run, cid):
    run = Path(run).resolve()
    plan = load(run / "plan.json")
    require(cid in plan["cases"], f"未知のケース: {cid}")
    ex, case, cwd = plan["execution"], plan["cases"][cid], run / "cases" / cid
    state_path = cwd / "state.json"
    require(not state_path.exists(), f"{cid} は既に始まっている。自動では再実行しない")
    mesh = run / "meshes" / case["mesh"] / "mesh.su2"
    require(mesh.exists(), f"格子がない: {mesh}")
    c = ex.get("container")
    if c:
        ok = subprocess.run([c["engine"], "image", "inspect", c["image_id"]], capture_output=True).returncode == 0
        require(ok, f"イメージ {c['image_id']} がこのホストに無い（未開始のまま終わる）")

    signal.signal(signal.SIGTERM, _cancel)
    signal.signal(signal.SIGINT, _cancel)
    cmd, cname = solver_command(ex, cwd, run, f"{plan['name']}-{cid}")
    record = dict(case=cid, mesh=case["mesh"], aoa_deg=case["aoa_deg"], delta_deg=case["delta_deg"], command=cmd,
                  container=cname, image_id=c["image_id"] if c else None, ranks=ex["ranks"],
                  mesh_sha256=sha(mesh), cfg_sha256=sha(cwd / "case.cfg"), pbs_job_id=os.environ.get("PBS_JOBID"),
                  started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if not (cwd / "mesh.su2").exists():
        record["mesh_link"] = link_mesh(mesh, cwd / "mesh.su2")
    dump(state_path, dict(status="running", started=record["started"], pbs_job_id=record["pbs_job_id"]))
    t0, rc, status = time.time(), None, "finished"
    with open(cwd / "su2.log", "w") as logf:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=ex["case_time_limit_s"])
        except (Cancelled, subprocess.TimeoutExpired) as stop:
            status = "cancelled" if isinstance(stop, Cancelled) else "timeout"
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    if cname:  # クライアントが止まってもコンテナが残ることがあるので、必ず消す
        subprocess.run([c["engine"], "rm", "-f", cname], capture_output=True)
    final = last_history(cwd / "history.csv") if (cwd / "history.csv").exists() else {}
    if status != "finished":
        outcome = status
    elif rc != 0 or not (cwd / "surface_flow.vtu").exists() or final.get("nonphysical"):
        outcome = "failed"
    else:
        outcome = "converged" if final.get("rms_rho", 0.0) <= plan["solver"]["residual_target"] + 1e-9 else "unconverged"
    record.update(exit_code=rc, wall_s=round(time.time() - t0, 1), final=final, outcome=outcome)
    dump(cwd / "run.json", record)
    dump(state_path, dict(status="done", outcome=outcome, exit_code=rc, finished=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    log(f"{cid}: {outcome}（終了コード {rc}, {record['wall_s']} s, 反復 {final.get('iter')}, "
        f"log10 rms[Rho] {final.get('rms_rho')}）")
    return 0 if outcome in ("converged", "unconverged") else 1


def archive_case(run, cid, confirm_not_running=False):
    """やり直すために、ケースの記録を cases/<case>/archive-<n>/ へ移す。case.cfg は残す。"""
    run = Path(run).resolve()
    cwd = run / "cases" / cid
    require((cwd / "state.json").exists(), f"{cid} には記録がない")
    require(load(cwd / "state.json").get("status") != "running" or confirm_not_running,
            f"{cid} は running のまま。ジョブが無いことを確かめてから --confirm-not-running を付ける")
    n = 1 + len(list(cwd.glob("archive-*")))
    dest = cwd / f"archive-{n}"
    dest.mkdir()
    for p in cwd.iterdir():
        if p.name.startswith("archive-") or p.name == "case.cfg":
            continue
        p.rename(dest / p.name)
    shutil.copy2(cwd / "case.cfg", dest / "case.cfg")
    ledger_path = run / "pbs" / "jobs.json"
    if ledger_path.exists():
        ledger = load(ledger_path)
        if cid in ledger["cases"]:
            ledger.setdefault("history", []).append(dict(case=cid, **ledger["cases"].pop(cid)))
            dump(ledger_path, ledger)
    log(f"{cid}: 記録を {dest.name} へ移した")


def status(run):
    run = Path(run).resolve()
    plan = load(run / "plan.json")
    for mid in plan["meshes"]:
        p = run / "meshes" / mid / "mesh.json"
        print(f"格子 {mid:<8}" + (f"翼面 {load(p)['fin_nodes']} 節点" if p.exists() else "未生成"))
    for cid in plan["order"]:
        sp, rp = run / "cases" / cid / "state.json", run / "cases" / cid / "run.json"
        if not sp.exists():
            print(f"{cid:<12}未開始")
            continue
        st = load(sp)
        fin = load(rp).get("final", {}) if rp.exists() else {}
        print(f"{cid:<12}{st.get('outcome', st['status']):<12}反復 {fin.get('iter')}  log10 rms[Rho] {fin.get('rms_rho')}"
              f"  CL {fin.get('CL')}  CD {fin.get('CD')}")
    print("analysis/summary.json: " + ("あり" if (run / "analysis" / "summary.json").exists() else "なし"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("config")
    p.add_argument("--skip-image-check", action="store_true",
                   help="イメージを調べずに計画を作る（別の計算機で prepare するとき。ID での固定はされない）")
    sub.add_parser("mesh").add_argument("run")
    p = sub.add_parser("submit")
    p.add_argument("run")
    p.add_argument("--backend", choices=["pbs", "local"])
    p.add_argument("--only", nargs="+")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("run-case")
    p.add_argument("run")
    p.add_argument("case")
    p = sub.add_parser("archive-case")
    p.add_argument("run")
    p.add_argument("case")
    p.add_argument("--confirm-not-running", action="store_true")
    sub.add_parser("status").add_argument("run")
    sub.add_parser("analyze").add_argument("run")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "prepare":
            prepare(a.config, a.skip_image_check)
        elif a.cmd == "mesh":
            make_meshes(a.run)
        elif a.cmd == "submit":
            submit(a.run, a.backend, a.only, a.dry_run)
        elif a.cmd == "run-case":
            return run_case(a.run, a.case)
        elif a.cmd == "archive-case":
            archive_case(a.run, a.case, a.confirm_not_running)
        elif a.cmd == "status":
            status(a.run)
        elif a.cmd == "analyze":
            from .analysis import analyze
            analyze(a.run)
    except ValueError as ex:
        print(f"エラー: {ex}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
