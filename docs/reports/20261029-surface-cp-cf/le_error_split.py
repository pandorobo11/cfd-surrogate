"""Cp/Cf relative L2 with and without points near the leading edge, for saved surfcpcf runs."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '.')
from surfcpcf.data import Normalization, SurfaceDataset, load_case, read_splits, to_device
from surfcpcf.model import build_model, select_device

surface, tdir = sys.argv[1], sys.argv[2]
for run in sys.argv[3:]:
    run = Path(run); c = json.loads((run / 'config.json').read_text())
    names = read_splits(surface, c['task'])['test']
    norm = Normalization.load(run / 'norm.npz'); ds = SurfaceDataset(surface, names, norm)
    dev = select_device(None)
    m = build_model(c.get('model', 'transolver'), tdir, c['n_hidden'], c['n_layers'], c['slice_num']).to(dev)
    m.load_state_dict(torch.load(run / 'model.pt', map_location=dev, weights_only=True)); m.eval()
    out = {k: [] for k in ('cp_all', 'cp_out', 'cf_all', 'cf_out', 'cp_le_share')}
    with torch.no_grad():
        for name, (d, _) in zip(names, ds):
            case = load_case(surface, name)
            pr = norm.inverse_targets(m(to_device(d, dev)).cpu().numpy())
            x = case['pos'][:, 0]; far = x > x.min() + 0.02 * np.ptp(x)
            for j, f in enumerate(('cp', 'cf')):
                e, t = pr[:, j] - case[f], case[f]
                out[f + '_all'].append(np.linalg.norm(e) / np.linalg.norm(t))
                out[f + '_out'].append(np.linalg.norm(e[far]) / np.linalg.norm(t[far]))
            e = pr[:, 0] - case['cp']; out['cp_le_share'].append(float((e[~far] ** 2).sum() / (e ** 2).sum()))
    print(run.name, c.get('model', 'transolver'), {k: (round(float(np.median(v)), 4), round(float(np.mean(v)), 4)) for k, v in out.items()})
