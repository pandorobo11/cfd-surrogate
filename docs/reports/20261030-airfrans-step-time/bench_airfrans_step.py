"""Time one training step / inference of the official AirfRANS Transolver config on MPS with dummy inputs."""
import sys, time, types
import numpy as np
import torch

sys.path.insert(0, sys.argv[1])  # path to Airfoil-Design-AirfRANS
from models.Transolver import Transolver

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


class TransolverDev(Transolver):
    def get_grid(self, my_pos):  # same as upstream, without the hard-coded .cuda()
        b = my_pos.shape[0]
        gx = torch.tensor(np.linspace(-2, 4, self.ref), dtype=torch.float).reshape(1, self.ref, 1, 1).repeat([b, 1, self.ref, 1])
        gy = torch.tensor(np.linspace(-1.5, 1.5, self.ref), dtype=torch.float).reshape(1, 1, self.ref, 1).repeat([b, self.ref, 1, 1])
        g = torch.cat((gx, gy), dim=-1).to(my_pos.device).reshape(b, self.ref ** 2, 2)
        return torch.sqrt(torch.sum((my_pos[:, :, None, :] - g[:, None, :, :]) ** 2, dim=-1)).reshape(b, my_pos.shape[1], self.ref ** 2)


def sync():
    if DEVICE.type == 'mps':
        torch.mps.synchronize()


def bench(name, cfg, n_train, n_infer, steps=30, warmup=5):
    torch.manual_seed(0)
    model = TransolverDev(space_dim=7, fun_dim=0, out_dim=4, **cfg).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    nparam = sum(p.numel() for p in model.parameters())

    def data(n):
        return types.SimpleNamespace(x=torch.randn(n, 7, device=DEVICE), pos=torch.rand(n, 2, device=DEVICE),
                                     y=torch.randn(n, 4, device=DEVICE))

    d = data(n_train)
    for i in range(warmup + steps):
        if i == warmup:
            sync(); t0 = time.time()
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(d), d.y)
        loss.backward()
        opt.step()
    sync(); t_train = (time.time() - t0) / steps
    mem_train = torch.mps.driver_allocated_memory() / 2**30 if DEVICE.type == 'mps' else float('nan')

    model.eval()
    d = data(n_infer)
    with torch.no_grad():
        model(d); sync(); t0 = time.time()
        for _ in range(5):
            model(d)
        sync(); t_inf = (time.time() - t0) / 5
    mem = torch.mps.driver_allocated_memory() / 2**30 if DEVICE.type == 'mps' else float('nan')
    print(f"{name}: params {nparam:,}  train step (N={n_train}) {t_train*1000:.0f} ms  "
          f"infer (N={n_infer}) {t_inf*1000:.0f} ms  mps_mem after train {mem_train:.2f} GiB, after infer {mem:.2f} GiB", flush=True)
    del model, opt
    if DEVICE.type == 'mps':
        torch.mps.empty_cache()


print('device:', DEVICE, 'torch', torch.__version__)
official = dict(n_hidden=256, n_layers=8, n_head=8, mlp_ratio=2, slice_num=32, unified_pos=1)
bench('official (256x8, slice32)', official, 32000, 180000)
reduced = dict(n_hidden=128, n_layers=4, n_head=8, mlp_ratio=2, slice_num=32, unified_pos=1)
bench('reduced (128x4, slice32), N=32k', reduced, 32000, 180000)
bench('reduced (128x4, slice32), N=16k', reduced, 16000, 180000)
