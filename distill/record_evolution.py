"""Re-run the best NEAT run (neat_kd_seed, seed 1, cap 150) and keep the best genome of every generation, so the
3D view can replay the evolution. The run is deterministic: the torch RNG is put in the state it had in the original
run (after the from-scratch NEAT winner's final 6000 training steps), and the final network is checked against
run_seed1.json.
  python3 record_evolution.py [seed]          # writes evolution_seed<seed>.json"""
import json, os, sys, time
sys.dont_write_bytecode = True
import numpy as np, torch
import distill as D

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
base = D.NM.base_net(seed); search, val, full = D.make_data(base, seed)
g160 = D.cached_nm160(base, seed)
torch.manual_seed(seed)                                   # RNG state of the original run at this point
for _ in range(D.FINAL_STEPS): torch.randint(len(full['X']), (256,))
seeded = g160.copy(); D.repair(seeded, search, D.CAP); D.Compiled(seeded).train(search, 200, lr=D.WARM_LR)
frames = []; t = time.time()
g, hist = D.neat([seeded], search, val, seed, mode='kd', log=lambda s: print(s, flush=True), frames=frames)
final = D.polish(g, full, 'kd', seed, lr=D.WARM_LR)
ref = json.load(open(os.path.join(D.HERE, f'run_seed{seed}.json')))['runs'][str(seed)]['neat_kd_seed']
print('final', final.count(), D.test_acc(final), 'original run', ref['params'], ref['acc'], f'{time.time() - t:.0f}s')
out = dict(seed=seed, start=D.describe(g160)['genome'], cut=D.describe(seeded)['genome'],
           frames=[dict(gen=k, fitness=round(f[0], 4), val_acc=round(f[1], 4), genome=D.describe(x)['genome']) for k, x, f in frames],
           final=D.describe(final)['genome'], final_acc=D.test_acc(final), ref=dict(params=ref['params'], acc=ref['acc']))
json.dump(out, open(os.path.join(D.HERE, f'evolution_seed{seed}.json'), 'w'), separators=(',', ':'))
