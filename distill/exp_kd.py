"""Sanity check before the NEAT runs: does distillation help a ~160-parameter network at all, and at which
temperature? Trains the nm160 topology (kept weights / fresh random weights) with labels vs. distillation.
  python3 exp_kd.py seed      # prints one line per setting"""
import sys, os, math, importlib
sys.dont_write_bytecode = True
import numpy as np, torch
import distill as D

seed = int(sys.argv[1]); base = D.NM.base_net(seed); search, val, full = D.make_data(base, seed)
g160 = D.cached_nm160(base, seed); print(seed, 'nm160', g160.count(), round(D.test_acc(g160), 4), flush=True)


def reinit(g, rng):
    g = g.copy(); fan = {}
    for (a, b) in g.conn: fan[b] = fan.get(b, 0) + 1
    for k in g.conn: g.conn[k] = float(rng.normal() * math.sqrt(2 / fan[k[1]]))
    for n in g.bias: g.bias[n] = 0.0
    return g


for init in ('keep', 'fresh'):
    for mode, T in (('hard', 0), ('kd', 1), ('kd', 2), ('kd', 4)):
        accs = []
        for r in range(2):
            D.TEMP = T or D.TEMP; rng = np.random.default_rng(r)
            g = g160 if init == 'keep' else reinit(g160, rng)
            lr = 1e-3 if init == 'keep' else D.LR
            torch.manual_seed(r); g = g.copy(); D.Compiled(g).train(full, D.FINAL_STEPS, lr=lr, mode=mode, decay=True)
            accs.append(round(D.test_acc(g), 4))
            if init == 'keep': break
        print(seed, init, mode, T, accs, flush=True)
