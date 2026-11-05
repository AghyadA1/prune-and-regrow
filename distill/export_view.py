"""Export the best distilled student (and the network it started from) for the 3D view: view_data.json.
  python3 export_view.py [seed]      # default seed 1, the best run (148 params, 93.7%)"""
import json, os, sys
sys.dont_write_bytecode = True
import numpy as np, torch, torch.nn.functional as F
import distill as D



def acts(g, X):
    """Activations of every node (inputs, hidden, outputs) for inputs X."""
    cg = D.Compiled(g)
    with torch.no_grad():
        z, A, _ = cg.forward(X, keep=True)
    cols = {n: c for n, c in cg.col.items()}
    return A, z, cols


def kl(p_log, q_log): return float((p_log.exp() * (p_log - q_log)).sum(1).mean())


def importance(g):
    """Mean-ablation KL on the training set (same measure as the pruning 3D view)."""
    X = D.C.Xtr; cg = D.Compiled(g)
    with torch.no_grad():
        p = F.log_softmax(cg.forward(X), 1); out = {}
        hid = g.hidden()
        for h in hid:   # replace h's activation by its mean: run the graph by hand
            A = X; mean = None
            for ncol, m, ci, rows, cols, bi, lv in cg.plan:
                W = torch.zeros(ncol, m).index_put((rows, cols), cg.w[ci]); zz = A @ W + cg.b[bi]
                if lv is D.OUT_IDS: break
                a = F.relu(zz)
                if h in lv: j = lv.index(h); a[:, j] = a[:, j].mean()
                A = torch.cat([A, a], 1)
            out[h] = kl(p, F.log_softmax(zz, 1))
        pix = []
        for i in range(64):
            Xa = X.clone(); Xa[:, i] = X[:, i].mean(); pix.append(kl(p, F.log_softmax(cg.forward(Xa), 1)))
    return out, pix


def export(g, label, new_ids):
    imp, pix = importance(g)
    nodes = [dict(id=h, d=round(g.depth[h], 4), b=round(g.bias[h], 4), imp=round(imp[h], 5), new=h in new_ids) for h in g.hidden()]
    edges = [[a, b, round(w, 4)] for (a, b), w in g.conn.items()]
    return dict(label=label, acc=round(D.test_acc(g), 4), params=g.count(), nodes=nodes, edges=edges,
                pix=[round(v, 5) for v in pix], obias=[round(g.bias[o], 4) for o in D.OUT_IDS])



if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    HERE = D.HERE
    base = D.NM.base_net(seed)
    start = D.cached_nm160(base, seed)
    best = D.to_genome(json.load(open(os.path.join(HERE, f'run_seed{seed}.json')))['runs'][str(seed)]['neat_kd_seed']['genome'])
    start_ids = {h: d for h, d in start.depth.items()}
    new = {h for h, d in best.depth.items() if h not in start_ids or abs(start_ids[h] - d) > 1e-9}
    data = dict(seed=seed, start=export(start, 'start', set()), best=export(best, 'best', new))
    # a few test digits (one per class) for the "feed a digit" mode
    idx = [int((D.C.yte == c).nonzero()[0]) for c in range(10)]
    data['samples'] = [dict(x=[round(float(v), 3) for v in D.C.Xte[i]], y=int(D.C.yte[i])) for i in idx]
    json.dump(data, open(os.path.join(HERE, 'view_data.json'), 'w'), separators=(',', ':'))
    print('start', data['start']['params'], data['start']['acc'], len(data['start']['nodes']), 'hidden')
    print('best', data['best']['params'], data['best']['acc'], len(data['best']['nodes']), 'hidden', len(new), 'new',
          sorted({n['d'] for n in data['best']['nodes']}))
