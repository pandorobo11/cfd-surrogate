# 全遊動翼の格子生成・SU2 解析・表面 Cp と Cf の補間（`finremap`）

2026年11月14日作成。舵角ごとに別々に作った表面格子の Cp を、舵角を戻して共通の基準格子へ補間する手順を、格子生成から結果処理までスクリプトで実行する方法をまとめる。手法と最初の検証は[記録（2026-11-14）](../reports/20261114-fin-remap-su2/README.md)にある。

`finremap/workflow.py` は、設定 JSON を 1 つ与えると、格子生成、SU2 の Euler 解析、補間と比較までを行う。設定に `flow.reynolds` を書くと、解析が層流の Navier–Stokes になり、摩擦係数 Cf のベクトルも補間する（[摩擦係数 Cf の補間](#摩擦係数-cf-の補間層流計算)）。SU2 は podman のイメージの中で動かし、OpenPBS にケースごとのジョブを投入する。PBS のない計算機では、同じ手順をその場で順に実行できる。

## 全体の流れ

```sh
# 0. 設定を用意する（例をコピーし、podman のイメージ名や PBS のキューを書き換える）
cp finremap/examples/fin-round-m2.json my-fin.json
# 1. 実行フォルダと各ケースの設定を作る（計算はしない）
python3 -m finremap.workflow prepare my-fin.json
# 2. 格子を生成する（gmsh が必要）
python3 -m finremap.workflow mesh runs/fin-round-m2
# 3. 解析を投入する（ケースごとに 1 ジョブ、最後に結果処理のジョブ）
export PYTHONPATH=$PWD/runs/fin-round-m2/scripts
python3 -m finremap.workflow submit runs/fin-round-m2 --dry-run   # 投入せずにジョブスクリプトを表示
python3 -m finremap.workflow submit runs/fin-round-m2
# 進み具合
python3 -m finremap.workflow status runs/fin-round-m2
# 4. 結果処理（投入したジョブが自動で行う。手動でも何度でも実行できる）
python3 -m finremap.workflow analyze runs/fin-round-m2
```

コマンドはリポジトリのルートで実行する。

- **実行ごとにフォルダを分ける**：`prepare` は `<output_root>/<name>/` を新しく作る。同じ名前のフォルダがあれば止まり、上書きしない。
- **スクリプトの写し**：`prepare` は `finremap/` を実行フォルダの `scripts/` に写し、ハッシュを `plan.json` に記録する。ジョブはこの写しを実行するので、計算中にリポジトリを更新しても実行中の計算は変わらない。`submit` 以降は `PYTHONPATH` をこの写しに向けて実行する。
- **始まったケースは自動では再実行しない**：`state.json` があるケースは `submit` も `run-case` も拒否する。やり直すときは `archive-case` で記録を移してから、そのケースだけ投入する。

```sh
python3 -m finremap.workflow archive-case runs/fin-round-m2 d05-a00
python3 -m finremap.workflow submit runs/fin-round-m2 --only d05-a00
```

`running` のまま止まったケースは、`qstat` でジョブが無いことを確かめてから `archive-case` に `--confirm-not-running` を付ける。

## 必要な環境

| 場所 | 必要なもの |
|---|---|
| `prepare`・`submit`・`status` | Python 3.9 以上（標準ライブラリだけ） |
| `mesh` | `numpy` と `gmsh`（`pip install gmsh`） |
| ジョブの中の解析 | SU2 8.5.0 と MPI（podman のイメージの中、またはホスト） |
| `analyze` | `numpy`、`scipy`、`matplotlib` |

`mesh` は計算ノードでなくてもよい。gmsh のない計算機で解析するときは、別の計算機で `prepare`（`--skip-image-check` を付ける）と `mesh` を行い、実行フォルダごと写す。`plan.json` には絶対パスを書いていないので、写した先でそのまま `submit` できる。この場合、イメージは ID ではなく名前のまま実行される。

## 実行されるケース

`meshes.deflections` の舵角 1 つにつき、2 ケースを作る。

| ケース | 格子 | 迎角 | 役割 |
|---|---|---|---|
| `d<舵角>-a00` | 舵角を付けた格子 | 0 | 補間の元になる解 |
| `ref-a<舵角>` | 基準格子（舵角 0） | 舵角と同じ | 比較相手 |

翼単体では「舵角 δ・迎角 0」と「舵角 0・迎角 δ」が同じ流れになるので、前者の Cp を基準格子へ補間し、後者と比べる。舵角 5°・10° なら、格子は 3 個、ケースは 4 個になる。舵角が整数でないとき、名前は `d07p5-a00` のようになる。

## 設定ファイル

[`finremap/examples/fin-round-m2.json`](../../finremap/examples/fin-round-m2.json) が例である。省略した項目は既定値になる。知らない項目名があると `prepare` が止まる。

| 項目 | 内容 |
|---|---|
| `name` | 実行フォルダの名前。英数字と `- _ .` |
| `output_root` | 実行フォルダを作る場所。相対パスは設定ファイルの場所から。既定は `runs`（Git の管理外） |
| `geometry` | 付け根翼弦 `root_chord`、翼端翼弦 `tip_chord`、翼幅 `span`、翼端の前縁位置 `tip_le_x`、最大翼厚比 `thickness_ratio`、前縁・後縁の半径÷局所翼弦長 `edge_radius_ratio`、縁として別に扱う範囲 `edge_zone`（翼弦比）、舵軸の位置 `hinge_x` |
| `domain` | 計算領域の `x`・`z` の範囲、`y_max`、遠方の格子幅 `h_far` |
| `flow` | `mach`、`pressure`、`temperature`（一様流）。`reynolds`（付け根翼弦基準）を書くと層流の Navier–Stokes になる。既定の `null` は Euler |
| `meshes.reference` | 基準格子の翼面の格子幅 `h_fin`、縁まわりの格子幅 `h_edge`、Gmsh の 2 次元アルゴリズム `algo2d` |
| `meshes.deflections` | 舵角 `deg` の一覧。`h_fin`・`h_edge`・`algo2d` を書くと、その舵角だけ基準格子と違う値になる |
| `meshes.boundary_layer` | 翼面のプリズム層（境界層格子）。既定の `null` は四面体だけの格子。項目は[プリズム層](#プリズム層境界層格子)を参照。`flow.reynolds` があるときだけ使える |
| `meshes.threads` | 格子生成のスレッド数。既定の 1 では、同じ設定から同じ格子ができる。増やすと速いが、節点数が実行ごとに少し変わる |
| `solver` | 最大反復数 `iter`、制限関数を固定する反復 `limiter_iter`、収束とみなす密度残差 `residual_target`（常用対数）、CFL の下限と上限 `cfl`、流れ場全体の VTU を出すか `volume_output` |
| `execution` | `backend`（`pbs` または `local`）、`ranks`（MPI 数）、`su2_cfd`、`mpirun`（`{ranks}` が置き換わる）、`python`、`setup`（ジョブの中で先に実行する行。例：`module load ...`）、`case_time_limit_s`、`container`、`pbs` |
| `execution.container` | `engine`（既定 `podman`）、`image`（`null` ならホストの `su2_cfd` を直接実行）、`options`（`podman run` に足す引数） |
| `execution.pbs` | `queue`、`resources`（`#PBS -l` の行）、`job_prefix`、`extra_directives`、結果処理のジョブ用の `analysis_queue`・`analysis_resources` |

同じ設定で舵角だけを変えると、Gmsh はほぼ同じ翼面格子を作る。補間の検証として使うときは、例のように舵角ごとに格子幅かアルゴリズムを変える。

## podman のイメージの SU2 を使う

`execution.container.image` を設定すると、各ケースを次のコマンドで実行する（`submit --dry-run` で実際のコマンドを表示する）。

```sh
podman run --rm --init --name su2-<name>-<case>-<pid> -v <実行フォルダ>:<実行フォルダ> -w <ケースのフォルダ> \
  -e OMP_NUM_THREADS=1 ... <options> <イメージID> mpirun -np 8 SU2_CFD case.cfg
```

- **MPI はイメージの中で動かす**。`mpirun` と `su2_cfd` は、イメージの中のコマンド名またはパスで書く。1 ジョブ 1 ノードの前提である。
- **実行フォルダは同じパスでマウントする**。格子は各ケースのフォルダへハードリンク（できなければ相対のシンボリックリンク）で置くので、実行フォルダの外のファイルは使わない。
- **イメージは ID で固定する**。`prepare` がイメージ名を `podman image inspect` で調べ、ID を `plan.json` に記録する。ジョブはこの ID を実行するので、途中でイメージを作り直しても実行中の計算は変わらない。イメージが無ければ `prepare` は止まる。`run-case` も、始める前に ID がそのホストにあるか確かめ、無ければ `state.json` を作らずに終わる。
- **停止**：`qdel` などで SIGTERM が届くと、SU2 を止め、`podman rm -f <コンテナ名>` でコンテナを消してから `cancelled` と記録する。

ワークステーションで最初に確かめることは、[su2-public の掃引の説明](https://github.com/pandorobo11/su2-public/blob/main/docs/intake-euler/intake-sweep-pbs.md#podman-イメージの-su2-を使う)と同じである（イメージの中で SU2 8.5.0 が動くこと、root で動かすときの `--allow-run-as-root`、PBS のジョブの中から podman が動くこと、`submit --only <case>` で 1 ケースだけ試すこと）。

## PBS を使わない実行

```sh
python3 -m finremap.workflow submit runs/fin-round-m2 --backend local
```

その計算機でケースを順に実行し、最後に結果処理を行う。同じ計算機で別の SU2 や mpirun が動いていれば始めない（`execution.local_exclusive`）。

## 実行フォルダの中身

| 場所 | 内容 |
|---|---|
| `plan.json`、`config.json` | 計画（ケース、格子の設定、実行方法、スクリプトのハッシュ、イメージ ID）と、元の設定の写し |
| `scripts/finremap/` | 実行に使うスクリプトの写し |
| `meshes/<格子>/` | `mesh.su2` と、節点数・品質・ハッシュを書いた `mesh.json` |
| `cases/<ケース>/` | `case.cfg`、`su2.log`、`history.csv`、`restart_flow.dat`、`surface_flow.vtu`、`run.json`（コマンド、イメージ ID、ハッシュ、終了コード、最終の残差と空力係数）、`state.json` |
| `pbs/` | ジョブスクリプト、ジョブの出力、投入の記録 `jobs.json` |
| `analysis/` | 結果処理の出力（下の節） |

ケースの結果（`run.json` の `outcome`）は次の 5 つである。

| outcome | 意味 | 結果処理 |
|---|---|---|
| `converged` | 密度残差が `residual_target` 以下 | 使う |
| `unconverged` | 正常に終わったが、残差が届かない | 使う（表に状態を表示する） |
| `failed` | SU2 の異常終了、非物理点、または `surface_flow.vtu` がない | 使わない |
| `cancelled`、`timeout` | 止められた、または `case_time_limit_s` を超えた | 使わない |

失敗したケースの自動のやり直しはしない。原因を確かめてから `archive-case` で記録を移し、再投入する。

## 結果処理（`analyze`）

各ケースの `surface_flow.vtu` から、節点の座標、三角形の接続、`Pressure_Coefficient` を読む。格子ファイルや CSV は読まない。

1. 舵角 δ の格子の節点を、舵軸まわりに −δ 回して翼固定座標に戻す。
2. 基準格子の各節点から、元格子の三角形面上の最近接点を求める。法線が逆向きの三角形（裏面）は候補から外す。
3. 最近接点の重心座標で、三角形の 3 頂点の Cp を平均する。
4. 候補の三角形（重心が近い 24 個）がすべて裏面だった節点は、Cp を NaN にする。力の積分ではその節点を含む三角形を外し、Cp の差の RMS と最大値ではその節点を除く。数は「対応する面が見つからない節点」（`unmatched_nodes`）として出す。

出力は `analysis/` に置く。

| ファイル | 内容 |
|---|---|
| `summary.json`、`results.csv` | 舵角の組ごと・手法ごとの、翼固定軸の法線力・軸力・舵軸まわりのモーメント、補間前からの変化、比較相手との Cp の差（全体、縁のまわり、それ以外） |
| `remap-<ケース>.png` | 後縁・前縁のまわりの Cp、上面の Cp（元・補間後・比較相手）、差の分布、翼幅中央の比較 |
| `remapped-<ケース>.vtu` | 基準格子の上に、補間した Cp（`Cp_mapped`）、比較相手の Cp（`Cp_reference`）、差、最近傍節点による値、投影距離を載せたもの。ParaView で開ける |

標準出力には、最初に SU2 が出力した `CFz`・`CMy` と、VTU の Cp を積分した値を並べて表示する。ここが一致しないときは、読み込みか三角形の向きに誤りがある。

比較する手法は 4 つである。A は回転を戻さずに最近傍節点、B は回転を戻して最近傍節点、C は回転を戻して投影＋重心座標、D は C に法線チェックを加えたもの。実際に使うのは D で、A〜C は比較のために計算する。

## 摩擦係数 Cf の補間（層流計算）

2026年11月17日追加。Cf の補間と積分の手順を確かめるための、小さい粘性計算である。Cf の値そのものの精度は求めない。

```sh
cp finremap/examples/fin-round-m2-laminar.json my-fin-laminar.json   # flow.reynolds が入った例
```

### 解析の設定

`flow.reynolds` を書くと、各ケースの `case.cfg` が次のように変わる。格子は、`meshes.boundary_layer` を書かなければ Euler と同じ四面体だけの格子で、書くとプリズム層を付ける（[プリズム層](#プリズム層境界層格子)）。

| 項目 | 内容 |
|---|---|
| ソルバー | 層流の Navier–Stokes（乱流モデルなし） |
| 翼面 | 断熱壁（熱流束 0） |
| 粘性係数 | 一定。一様流の圧力・温度・Mach 数と付け根翼弦から、指定したレイノルズ数になる値を計算して書く |
| 熱伝導 | プラントル数 0.72 で一定 |
| 追加の出力 | `forces_breakdown.dat`（力の圧力分と摩擦分の内訳）。`run-case` が `run.json` の `forces_breakdown` に写す |

一様流の圧力と温度は Euler の計算と同じ値のままにしている。密度残差の大きさが変わらないので、`residual_target` をそのまま使える。

レイノルズ数は低くする。境界層を厚くして、境界層格子のない格子でも Cf が滑らかに出るようにするためである。例の 5,000 は、平板の層流境界層の厚さの見積もり（後縁で翼弦の 1 割弱）が翼面の格子幅の数倍になるように選んだ。ただし、境界層格子のない格子の Cf は、プリズム層を付けた格子の値の半分以下で、格子による差も約 3% と大きかった（[記録](../reports/20261119-fin-cf-remap-prism/README.md)）。Cf の値を使うときはプリズム層を付ける。

### 結果処理

SU2 の表面出力の `Skin_Friction_Coefficient`（3 成分）を読む。これは、物体が受ける向きの壁面せん断応力を一様流の動圧で割ったベクトルである。位置の対応は Cp の手法 D と同じものを使う。

1. Cf の成分を、座標と同じ回転（舵軸まわりに −δ）で翼固定座標に戻す。
2. 3 成分をそれぞれ重心座標で補間する。
3. 基準格子の節点の法線方向の成分を除く（接平面へ射影）。

比較する手法は 3 つである。実際に使うのは G で、E と F は比較のために計算する。

| 手法 | 内容 |
|---|---|
| E | 成分を回さずに補間する（座標だけ回転を戻す） |
| F | 成分を回して補間する |
| G | F の後に接平面へ射影する |

摩擦力は、三角形ごとに 3 頂点の Cf を平均し、面積を掛けて足す（`remap.integrate_friction`）。法線は掛けない。

標準出力と `summary.json`・`results.csv` に加わる内容は次のとおり。

| 出力 | 内容 |
|---|---|
| 力の内訳の照合 | 元格子の `CFx`・`CFz`・`CMy` について、SU2 の圧力分・摩擦分と、VTU の Cp・Cf を積分した値 |
| 摩擦力（翼固定軸） | 軸力 `CA_f`、縁のまわりとそれ以外に分けた軸力 `CA_f_edge`・`CA_f_rest`、法線力 `CN_f`、舵軸まわりのモーメント `CMy_f` と、補間前からの変化 |
| Cf の差 | 補間した Cf と、基準格子で迎角を付けて解いた Cf の差ベクトルの大きさ（全体、縁のまわり、それ以外の RMS と最大） |
| `cf_normal_rms` | 補間した Cf の、基準格子の法線方向の成分の RMS。G では 0 になる |

`remap-cf-<ケース>.png` に Cf の図を、`remapped-<ケース>.vtu` に `Cf_mapped`（G）、`Cf_reference`、`Cf_difference`、`Cf_unprojected`（F）を出す。

SU2 の摩擦分と VTU の積分は、完全には一致しない。SU2 は力の積分に粘性応力の法線成分も含めるが、出力される Cf は接線成分だけである（SU2 のソースで確認）。粗い格子での動作確認では、摩擦分の `CFx` は小数第 5 位まで一致し、`CFz` に 0.0003 の差が出た。層流の検証では、restart と格子から壁面の速度勾配を計算し直し、z 成分の差が粘性応力の法線成分であること、x 成分の小さな差が主に SU2 の節点面積の取り方によることを確かめた（[記録](../reports/20261117-fin-cf-remap-laminar/README.md#摩擦分の差の内訳11-月-17-日に追加)）。法線成分は物理的には 0 になるべき離散化誤差なので、力は VTU の Cp と Cf の積分を正とする。

### 動作確認（2026-11-17、Mac mini）

粗い格子（翼面の格子幅 0.04 と 0.045、縁まわり 0.004 と 0.0045、全体で約 5〜6 万節点）で、舵角 5° の 2 ケースを `prepare` から `analyze` まで通した。2 ケースとも約 265 反復で収束した。コードが動くことの確認であり、補間の精度の結論ではない。例の設定の格子での計算と結果は、[記録（2026-11-17）](../reports/20261117-fin-cf-remap-laminar/README.md)にある。

## セル中心の値の補間（`remap.map_cell_values`）

2026年11月18日追加。補間元も補間先も三角形ごとの値（セル中心の値）のときは、元格子のつながりから「セル中心を頂点とする三角形格子」を作り、補間先の三角形の重心をそこへ投影して、重心座標で 3 つのセル値を平均する。手法と検証は[記録（2026-11-18）](../reports/20261118-cell-centered-remap/README.md)にある。

```python
from finremap import remap

q = dst_points[dst_tri].mean(axis=1)                       # 補間先の三角形の重心
qn, _ = remap.face_normals_areas(dst_points, dst_tri)      # 補間先の三角形の法線（外向き）
val, dist, band = remap.map_cell_values(src_points, src_tri, src_cell, q, qn)
f, m = remap.integrate_forces_cells(dst_points, dst_tri, val, ref_area, origin)
```

- **セル中心の三角形格子**（`remap.centroid_mesh`）：元格子の各節点について、その節点を共有する三角形を辺でたどって並べ、重心を結んだ多角形を耳切りで三角形に分ける。三角形の数は元の約 2 倍になる。
- **環を切る辺**：境界の辺、隣の三角形との法線の角度が `feature_angle_deg`（既定 45°）を超える辺、`part`（三角形ごとの部位番号）が変わる辺、3 枚以上の三角形が共有する辺。翼単体の格子では、翼端の縁と翼根の境界だけが切られた。
- **表と裏**：同じ節点を共有する三角形の重心だけを結ぶので、上面と下面を結ぶ三角形はできない。投影では、補間先の法線との角度が 60° を超える三角形も候補から外す（`min_cos`）。
- **縁の帯**：境界と特徴線に沿った幅が三角形の 1/3 程度の帯は覆われない。そこに落ちた点（戻り値の `band`）は、最も近い三角形の平面で線形に外挿する（`extrapolate=False` なら縁の辺の上の値）。学習データに使うときは印を付けておく。
- **力の積分**：セル値は `remap.integrate_forces_cells` で三角形ごとに積分する。節点値から `remap.cell_values`（3 頂点の平均）で作ったセル値なら、`integrate_forces` と同じ値になる。
- 保存型ではない。翼単体の検証では、補間前後の力の変化は 0.1% 未満だった。

## 一般の VTU で部位を選んで回す雛形（`finremap/parts.py`）

胴体と複数の翼を含む表面格子に広げるための雛形である。部位ごとに節点を選び、それぞれの舵軸まわりに舵角ぶん逆回転して、基準形状（全舵角 0）の座標を作る。

```sh
python3 -m finremap.parts surface_flow.vtu parts.json out.vtu
```

`parts.json` は部位の一覧である。

```json
[
 {"name": "fin1", "select": {"component_near": [0.5, 0.4, 0.0]},
  "origin": [0.45, 0.0, 0.0], "axis": [0, 1, 0], "angle_deg": 10.0}
]
```

| 項目 | 内容 |
|---|---|
| `select` | 部位の選び方。`{"array": "名前", "values": [...]}`（節点データまたはセルデータの値）、`{"box": [[x0,y0,z0],[x1,y1,z1]]}`（直方体の中）、`{"component_near": [x,y,z]}`（その点に最も近い節点を含む、つながった面）、`{"all": true}` |
| `origin`、`axis` | 舵軸が通る点と、舵軸の向き。任意の向きを指定できる |
| `angle_deg` | その格子での舵角。軸に対して右ねじの向きを正とする |

出力の VTU には、基準形状の座標 `ReferencePosition` と部位番号 `PartIndex`（どの部位にも入らない節点は −1）が節点データとして加わる。Python からは `parts.to_reference(mesh, parts)` で同じ処理を呼べる。得られた座標を `remap.map_barycentric` に渡せば、共通格子へ補間できる。

雛形として未対応のことは次のとおり。

- **SU2 の表面 VTU は境界の名前を持たない**。翼と胴体が離れていれば `component_near` で分けられる。付け根で節点を共有しているときは、`box` で選ぶか、部位番号の配列を別に用意する。
- **舵角で濡れ面が変わる部分**（付け根の隙間、翼に隠れる胴体の面）は、回しても形状が一致しない。補間後の投影距離が大きい節点として現れるので、しきい値で除外する処理を足す必要がある。
- **複数の部位に入る節点**はエラーにしている。
- 胴体付きの形状での確認はまだ行っていない。確かめたのは、合成した 2 枚の面と、翼単体の SU2 出力である。

VTU の読み込み（`finremap/vtu.py`）は、Piece が 1 つのファイルで、ascii・base64・raw appended、非圧縮・zlib 圧縮、ヘッダ型 UInt32・UInt64 に対応する。LZ4・LZMA 圧縮と、複数の Piece には対応していない。

## プリズム層（境界層格子）

2026年11月19日追加。`meshes.boundary_layer` に dict を書くと、翼面の三角形からプリズムの層を押し出し、その外側を四面体で埋めた格子を作る（`finremap/prism.py`）。翼面の三角形格子は、プリズム層のない格子と同じものになる（同じ `h_fin`・`h_edge`・`algo2d` なら同じ節点）。

```sh
cp finremap/examples/fin-round-m2-laminar-bl.json my-fin-bl.json
```

| 項目 | 既定値 | 内容 |
|---|---|---|
| `layers` | 15 | 層の数 |
| `ratio` | 1.2 | 層の高さの公比。最初の層の高さは T (ratio − 1) / (ratio^layers − 1) |
| `thickness_coeff` | 7.0 | 層全体の厚さ T = thickness_coeff √(x ν/U)。x は局所前縁からの距離、ν/U は付け根翼弦 ÷ `flow.reynolds`。平板の層流境界層の厚さ 5√(x ν/U) に、Mach 2 の断熱壁で厚くなる分の余裕を見た値 |
| `t_min` | 0.003 | T の下限（前縁のまわり） |
| `t_te` | 0.005 | 後縁での T。後縁から翼弦の `te_zone` の範囲で、T をこの値まで滑らかに減らす。丸い後縁を厚い層で包むと、層の上面の三角形が引き伸ばされるため |
| `te_zone` | 0.2 | 上の範囲（翼弦比） |
| `root_blend` | 0.15 | 対称面 (y=0) 上の節点では、押し出す向きの y 成分を 0 にする。翼の付け根からこの距離の範囲で、本来の法線へ滑らかにつなぐ |
| `smooth_iter` | 30 | 押し出す向きと T の平滑化の回数。翼端の角で向きが急に変わらないようにする |

作り方は次のとおり。

1. 翼面の三角形を流体側へ向け、節点の法線（角度重み）を平滑化して押し出す向きにする。
2. 節点ごとの T と公比から、各層の節点を置く。各層の三角形が裏返っていないこと、プリズムを 3 つの四面体に分けた体積が正であることを確かめ、`mesh.json` の `prism_stats` に書く。
3. 層の上面の付け根の輪を穴にした対称面と、遠方境界の 5 面を、Gmsh の組み込みの幾何で 2 次元格子にする。
4. 層の上面・対称面・遠方境界で閉じた三角形の面の内側を、Gmsh（HXT）で四面体に分ける。格子幅は、四面体だけの格子と同じ「翼面・縁からの距離のしきい値」（Python のコールバック）で決める。
5. SU2 形式で書く。対称面の境界 `sym` には、三角形と、プリズム層の側面の四角形が入る。要素と境界の向きは、SU2 が並べ替えを報告しない向き（プリズムは VTK の順、境界の法線は流体の側）にしてある。

`mesh.json` には、プリズムと四面体の数、四面体の最小 SICN、`boundary_layer`（使った値）、`prism_stats`（T と最初の層の高さの範囲、品質の指標）が入る。

`mesh` の手順には `numpy`・`scipy`・`gmsh` が要る。SU2 のメモリ（全プロセスの RSS の合計）は、2026-11-19 の実行で 1 節点あたり約 6〜7 KB だった。24 GB の計算機では 3.1M 節点の格子がスワップして大幅に遅くなり、2.6M 節点までは計算できた（[記録](../reports/20261119-fin-cf-remap-prism/README.md#計算時間とメモリ性能の観測)）。

格子を変えたときの Cf の差は、プリズム層を付けると、四面体だけの格子の約 3% から 0.1% 未満になった。補間前後の変化は同じ程度（0.011% 以下）だった（[記録](../reports/20261119-fin-cf-remap-prism/README.md)）。

圧力の法線力は、舵角を付けた格子の解を補間すると、基準格子に迎角を付けて別に解いた解より 0.8〜0.9% 小さくなった。基準格子を計算領域の箱ごと回して迎角 0° で解くと、翼から見て同じ格子なのに迎角を付けた解と 1.03% 違い、舵角を付けた格子の解とは 0.10% で一致した。この差は補間ではなく、SU2 の解の座標系への依存で、制限関数を外して解き直すと消える（速度の成分ごとに掛かる Venkatakrishnan の制限関数による。同じ記録）。プリズム層の格子で、補間した値を別に解いた解と比べるときは、基準格子を舵角だけ回した格子で迎角 0° として解いたものを比べる相手にする（`plan.json` の `pairs` はまだ迎角を付けた解を使う）。

## 試験

```sh
python3 -m pytest tests/test_finremap.py tests/test_prism.py
```

VTU の読み書き（`vtk` があれば、VTK が書いた各形式のファイルを読めることも確かめる）、回転、補間（スカラーとベクトル）、セル中心の三角形格子とセル値の補間、接平面への射影、圧力と摩擦の力の積分、部位の選択、`prepare`（Euler と層流の設定）とジョブスクリプトの生成、力の内訳の読み込みを確かめる。SU2 と gmsh は使わない。`tests/test_prism.py` は、付け根の輪の並べ方と層の厚さを確かめ、`gmsh` があれば粗いプリズム層の格子を作って、要素の数・使われない節点がないこと・対称面の要素が y=0 にあることを確かめる。

## 動作確認（2026-11-14、Mac mini）

例の設定を、イメージなし（ホストの SU2 8.5.0）・`--backend local`・12 MPI にして、`prepare` から `analyze` まで通した。結果の場はローカルの `results/` にあり、共有しない。

共通条件は、例の設定のとおり（Mach 2 の Euler、最大翼厚 6%、縁の半径は翼弦の 0.5%、格子生成は 1 スレッド）。

| 格子 | 舵角 [deg] | 翼面の格子幅 | 縁まわりの格子幅 | 2 次元アルゴリズム | 翼面の節点数 | 全節点数 |
|---|---|---|---|---|---|---|
| ref | 0 | 0.012 | 0.0010 | 6 | 138,453 | 517,113 |
| d05 | 5 | 0.013 | 0.0011 | 6 | 116,493 | 430,946 |
| d10 | 10 | 0.011 | 0.0009 | 5 | 187,025 | 654,979 |

| ケース | 格子 | 迎角 [deg] | 結果 | 反復数 | 密度残差（log10） | CL | CD | CMy（舵軸） | 計算時間 [s] |
|---|---|---|---|---|---|---|---|---|---|
| ref-a05 | ref | 5 | converged | 418 | −8.35 | 0.17820 | 0.05037 | −0.00872 | 141 |
| d05-a00 | d05 | 0 | converged | 416 | −8.22 | 0.17828 | 0.05027 | −0.00876 | 113 |
| ref-a10 | ref | 10 | converged | 418 | −8.17 | 0.35625 | 0.09810 | −0.01808 | 135 |
| d10-a00 | d10 | 0 | converged | 507 | −8.05 | 0.35629 | 0.09822 | −0.01803 | 232 |

補間前後の力の変化と、比較相手との Cp の差（手法 D）：

| 元のケース → 比較相手 | 法線力の変化 | 軸力の変化 | 舵軸モーメントの変化 | Cp の差の RMS（全体） | 縁のまわり | それ以外 |
|---|---|---|---|---|---|---|
| d05-a00 → ref-a05 | −0.013% | −0.005% | −0.061% | 0.0081 | 0.0090 | 0.0049 |
| d10-a00 → ref-a10 | −0.022% | −0.019% | −0.063% | 0.0083 | 0.0094 | 0.0044 |

- CSV と格子ファイルから読んでいた最初の検証（[記録](../reports/20261114-fin-remap-su2/README.md)）と、同じ大きさの結果になった。格子は作り直したので、節点数と値の末尾は少し違う。
- 格子生成を 1 スレッドでもう一度行うと、同じ `mesh.su2`（ハッシュが一致）ができた。
- 計算時間は観察値で、精度の結論とは別である。
- **確かめていないこと**：この計算機には podman と PBS がないので、イメージの中での実行と `qsub` による投入は動かしていない。確かめたのは、ジョブスクリプトと podman のコマンドの生成（`--dry-run` と試験）までである。ワークステーションでは、最初に `submit --only <case>` で 1 ケースだけ試す。
