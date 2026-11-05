"""Turn evolution_seed<seed>.json into view_evo.json for the 3D replay: one frame per stage
(starting network, cut to the cap, every NEAT generation, final training), each with importance and test accuracy.
  python3 export_evolution.py [seed]"""
import json, os, sys
sys.dont_write_bytecode = True
import torch
import distill as D
from export_view import importance

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
E = json.load(open(os.path.join(D.HERE, f'evolution_seed{seed}.json')))
start = D.to_genome(E['start']); start_ids = dict(start.depth)


def frame(g, label, kind, **extra):
    imp, pix = importance(g)
    nodes = [dict(id=h, d=round(g.depth[h], 4), b=round(g.bias[h], 4), imp=round(imp[h], 5),
                  new=h not in start_ids or abs(start_ids[h] - g.depth[h]) > 1e-9) for h in g.hidden()]
    return dict(label=label, kind=kind, acc=round(D.test_acc(g), 4), params=g.count(), nodes=nodes,
                edges=[[a, b, round(w, 4)] for (a, b), w in g.conn.items()], pix=[round(v, 5) for v in pix],
                obias=[round(g.bias[o], 4) for o in D.OUT_IDS], **extra)


frames = [frame(start, 'Starting network', 'start'), frame(D.to_genome(E['cut']), 'Cut to 150', 'cut')]
for f in E['frames']:
    if f['gen'] == 0: continue
    frames.append(frame(D.to_genome(f['genome']), f'Generation {f["gen"]}', 'gen', gen=f['gen'], val_acc=f['val_acc']))
frames.append(frame(D.to_genome(E['final']), 'Final training', 'final'))
base = D.NM.base_net(seed)
idx = [int((D.C.yte == c).nonzero()[0]) for c in range(10)]
out = dict(seed=seed, gens=max(f.get('gen', 0) for f in frames), teacher_acc=round(D.C.acc(base), 4), frames=frames,
           samples=[dict(x=[round(float(v), 3) for v in D.C.Xte[i]], y=int(D.C.yte[i])) for i in idx])
json.dump(out, open(os.path.join(D.HERE, 'view_evo.json'), 'w'), separators=(',', ':'))
print(len(frames), 'frames', [(f['label'], f['params'], f['acc']) for f in frames[:3] + frames[-2:]])
html = open(os.path.join(D.HERE, 'evo_template.html')).read().replace('__DATA__', json.dumps(out, separators=(',', ':')))
open(os.path.join(D.HERE, 'best-student.html'), 'w').write(html)
print('wrote best-student.html', len(html))
