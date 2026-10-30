"""Split the official AirfRANS force coefficients into pressure and friction parts, true vs predicted.

Uses the same inference (utils.metrics.Infer_test), field reconstruction (Airfoil_test) and integration
(as in Compute_coefficients) as the official scoring, but keeps the two force contributions separate.
Also reports velocity errors in bands of wall distance.

Usage (from Airfoil-Design-AirfRANS/):
  MPLBACKEND=Agg python diag_cd_airfrans.py <task> <dataset_dir> <model_file> <out_csv>
"""
import csv, json, random, sys, os.path as osp
import numpy as np
import pyvista as pv
import torch
import yaml
from torch_geometric.loader import DataLoader

sys.path.insert(0, '.')
import utils.metrics as metrics
from utils.reorganize import reorganize
from dataset.dataset import Dataset

random.seed(0); np.random.seed(0); torch.manual_seed(0)
task, my_path, model_file, out_csv = sys.argv[1:5]
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

with open(osp.join(my_path, 'manifest.json')) as f:
    manifest = json.load(f)
manifest_train = manifest[task + '_train']
n = int(.1 * len(manifest_train))
_, coef_norm = Dataset(manifest_train[:-n], norm=True, sample=None, my_path=my_path)  # same as main.py
with open('params.yaml') as f:
    hparams = yaml.safe_load(f)['Transolver']
model = torch.load(model_file, map_location=device, weights_only=False)

s = task + '_test' if task != 'scarce' else 'full_test'
names = manifest[s]
loader = DataLoader(Dataset(names, sample=None, coef_norm=coef_norm, my_path=my_path), shuffle=False)

BANDS = [(0, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, np.inf)]  # wall distance, chord = 1


def components(internal, airfoil, bool_surf, Uinf, angle):
    """Same steps as metrics.Compute_coefficients, returning (friction, pressure) coefficients in (drag, lift)."""
    intern, aerofoil = internal.copy(), airfoil.copy()
    point_mesh, point_surf = intern.points[bool_surf, :2], aerofoil.points[:, :2]
    intern = intern.compute_derivative(scalars='U', gradient='pred_grad')
    surf_grad = reorganize(point_mesh, point_surf, intern.point_data['pred_grad'].reshape(-1, 3, 3)[bool_surf, :2, :2])
    surf_p = reorganize(point_mesh, point_surf, intern.point_data['p'][bool_surf])
    aerofoil.point_data['wallShearStress'] = metrics.WallShearStress(surf_grad, -aerofoil.point_data['Normals'])
    aerofoil.point_data['p'] = surf_p
    aerofoil = aerofoil.ptc(pass_point_data=True)
    L = aerofoil.cell_data['Length'].reshape(-1, 1)
    wss = (aerofoil.cell_data['wallShearStress'] * L).sum(axis=0)
    wp = (-aerofoil.cell_data['p'][:, None] * aerofoil.cell_data['Normals'][:, :2] * L).sum(axis=0)
    a = angle * np.pi / 180
    basis = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    fric = 2 * (basis @ wss) / Uinf ** 2
    pres = 2 * (basis @ -wp) / Uinf ** 2
    return fric, pres


rows = []
for j, data in enumerate(loader):
    name = names[j]
    Uinf, angle = float(name.split('_')[2]), float(name.split('_')[3])
    outs, _ = metrics.Infer_test(device, [model], [hparams], data, coef_norm=coef_norm)
    intern = pv.read(osp.join(my_path, name, name + '_internal.vtu'))
    aerofoil = pv.read(osp.join(my_path, name, name + '_aerofoil.vtp'))
    surf = data.surf
    f_t, p_t = components(intern, aerofoil, surf, Uinf, angle)
    # predicted field on the mesh, built as in metrics.Airfoil_test (wall values set to 0)
    pi = intern.copy()
    out = (outs[0] * (coef_norm[3] + 1e-8) + coef_norm[2]).numpy()
    out[surf.numpy(), :2] = 0
    out[surf.numpy(), 3] = 0
    pi.point_data['U'][:, :2] = out[:, :2]
    pi.point_data['p'] = out[:, 2]
    f_p, p_p = components(pi, aerofoil, surf, Uinf, angle)
    # velocity error by wall distance (relative to Uinf), excluding wall points
    d = np.abs(intern.point_data['implicit_distance'])
    err = np.linalg.norm(out[:, :2] - intern.point_data['U'][:, :2], axis=1) / Uinf
    row = dict(case=name, Uinf=Uinf, aoa=angle,
               cd_fric_true=f_t[0], cd_pres_true=p_t[0], cd_fric_pred=f_p[0], cd_pres_pred=p_p[0],
               cl_fric_true=f_t[1], cl_pres_true=p_t[1], cl_fric_pred=f_p[1], cl_pres_pred=p_p[1])
    for lo, hi in BANDS:
        m = (d > lo) & (d <= hi) & (~surf.numpy())
        row[f'u_err_rms_{lo:g}_{hi:g}'] = float(np.sqrt((err[m] ** 2).mean()))
        row[f'n_pts_{lo:g}_{hi:g}'] = int(m.sum())
    rows.append(row)
    print(j, name, {k: round(v, 5) for k, v in row.items() if k.startswith('cd_')}, flush=True)

with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print('done', len(rows))
