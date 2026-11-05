"""NEAT-deep: NEAT-merge on a general feed-forward graph, so new neurons can also start new hidden layers.

Same criterion as neat_merge.py: make the change that loses the least per parameter saved, measured after the
neurons that stay are re-fitted (least squares) to cover for the ones that left.

What is new: every neuron has a depth (inputs 0, hidden 1 and 2, outputs 3) and a link may go from any neuron to
any deeper one, as in NEAT. A merge can put its new neuron
  same   in the layer of the neurons it replaces, or
  after  in a NEW hidden layer just behind that layer (depth halfway to the next one). There it can read from the
         neurons that stay in that layer, so the network gets deeper while it shrinks.

Variants (all use the look-ahead of neat_merge's winner, full3_look):
  flat    no skip links, no new layers: the same search as full3_look (sanity check)
  skip    links may skip layers (NEAT style), no new layers (ablation)
  deep    skip links + new layers
Result (3 seeds, mean test accuracy at ~460 / ~285 / ~160 parameters):
  flat 95.6 / 94.6 / 90.3   skip 96.2 / 94.6 / 90.2   deep 95.7 / 94.8 / 90.7   (neat_merge full3_look 96.5 / 95.1 / 91.0)
  New layers get used (the recorded run ends with 5 hidden layers) but do not beat the control beyond seed noise.

  python3 neat_deep.py [variant ...] [seed ...]      # writes neat_deep.json next to it
  env: ND_OUT=<file> output json (merged into if it exists), ND_REC=<variant> records seed 0 for the 3D view
"""
import json, os, sys, time, copy
import torch, torch.nn.functional as F
import numpy as np
import neat_merge as NM

HERE = os.path.dirname(os.path.abspath(__file__))
Xtr, ytr, Xte, yte = NM.Xtr, NM.ytr, NM.Xte, NM.yte
TARGETS, FT_EPOCHS = NM.TARGETS, NM.FT_EPOCHS
if os.environ.get('ND_TARGETS'): TARGETS = [int(t) for t in os.environ['ND_TARGETS'].split(',')]   # short test runs
OUT = 3.0
MAX_NEW_LAYERS = 3


class DNet:
    """Feed-forward graph. Nodes are kept sorted by depth; W[i, j] is the link i -> j (only if depth i < depth j)."""
    def __init__(self, base):
        Ws = [w.double().numpy() for w in base.W]; bs = [b.double().numpy() for b in base.b]
        self.key = [f'i{p}' for p in range(64)] + [f'h0_{j}' for j in range(32)] + [f'h1_{j}' for j in range(32)] + [f'o{d}' for d in range(10)]
        self.depth = np.array([0.] * 64 + [1.] * 32 + [2.] * 32 + [OUT] * 10)
        N = len(self.key); self.W = np.zeros((N, N)); self.M = np.zeros((N, N))
        for (r0, c0), w in zip([(0, 64), (64, 96), (96, 128)], Ws):
            self.W[r0:r0 + w.shape[0], c0:c0 + w.shape[1]] = w; self.M[r0:r0 + w.shape[0], c0:c0 + w.shape[1]] = 1
        self.b = np.r_[np.zeros(64), bs[0], bs[1], bs[2]]
        self.next_id = 100

    def copy(self): return copy.deepcopy(self)
    def idx(self): return {k: i for i, k in enumerate(self.key)}
    def hidden(self): return [i for i in range(len(self.key)) if 0 < self.depth[i] < OUT]
    def depths(self): return sorted(set(self.depth.tolist()))

    def groups(self):
        d = self.depth; cuts = np.flatnonzero(np.diff(d)) + 1
        return list(zip(np.r_[0, cuts].tolist(), np.r_[cuts, len(d)].tolist()))

    def forward(self, X, fix=None):
        N = len(self.key); A = np.zeros((len(X), N)); Z = np.zeros((len(X), N)); A[:, :64] = X; WM = self.W * self.M
        for s, e in self.groups()[1:]:
            rows = np.flatnonzero(self.M[:s, s:e].any(1))
            z = A[:, rows] @ WM[rows, s:e] + self.b[s:e]; Z[:, s:e] = z
            A[:, s:e] = z if self.depth[s] == OUT else np.maximum(z, 0)
            if fix:
                for i, v in fix.items():
                    if s <= i < e: A[:, i] = v
        return A, Z

    def logits(self, X): return self.forward(X)[0][:, -10:]

    def count(self): return int(self.M.sum()) + len(self.hidden()) + 10

    def remove(self, idxs):
        keep = [i for i in range(len(self.key)) if i not in set(idxs)]
        self.W, self.M = self.W[np.ix_(keep, keep)], self.M[np.ix_(keep, keep)]
        self.b, self.depth = self.b[keep], self.depth[keep]; self.key = [self.key[i] for i in keep]

    def insert(self, key, depth):
        pos = int(np.searchsorted(self.depth, depth, side='right'))
        self.W = np.insert(np.insert(self.W, pos, 0, axis=0), pos, 0, axis=1)
        self.M = np.insert(np.insert(self.M, pos, 0, axis=0), pos, 0, axis=1)
        self.b = np.insert(self.b, pos, 0.); self.depth = np.insert(self.depth, pos, depth); self.key.insert(pos, key)
        return pos


def loss_of(net, X=Xtr, y=ytr):
    z = net.logits(X); z = z - z.max(1, keepdims=True)
    return float(-(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean())


def acc_of(net, X=Xte, y=yte): return float((net.logits(X).argmax(1) == y).mean())


class Ctx:
    """The network before a change: its activations and the pre-activations every node should keep receiving."""
    def __init__(self, net):
        self.A, self.Z = net.forward(Xtr); self.at = net.idx()
    def z(self, keys): return self.Z[:, [self.at[k] for k in keys]]


def refit(net, T, A, Zt):
    """Re-solve the incoming links (and bias) of target nodes T so that they receive ~Zt, keeping the link masks."""
    U = np.flatnonzero(net.M[:, T].any(1)) if len(T) else np.array([], int); pos = {int(u): i for i, u in enumerate(U)}
    A1 = np.c_[A[:, U], np.ones(len(A))]; G = A1.T @ A1; R = A1.T @ Zt; d = len(U); lam = 1e-3 * np.trace(G) / (d + 1)
    for c, j in enumerate(T):
        act = np.r_[[pos[int(i)] for i in np.flatnonzero(net.M[:, j])], d].astype(int); Gs = G[np.ix_(act, act)].copy()
        Gs[np.diag_indices_from(Gs)] += lam; Gs[-1, -1] -= lam
        sol = np.linalg.solve(Gs, R[act, c]); net.W[:, j] = 0; net.W[U[act[:-1]], j] = sol[:-1]; net.b[j] = sol[-1]


def refit_down(net, keys, ctx):
    """Re-fit the given nodes in depth order (each layer sees the already re-fitted layers before it)."""
    at = net.idx(); T = sorted(at[k] for k in set(keys) if k in at)
    for dep in sorted(set(net.depth[T].tolist())):
        Tg = [j for j in T if net.depth[j] == dep]; A, _ = net.forward(Xtr)
        refit(net, Tg, A, ctx.z([net.key[j] for j in Tg]))


def children(net, idxs): return sorted({int(j) for i in idxs for j in np.flatnonzero(net.M[i])})


def next_layer(net, d): nd = min(x for x in net.depths() if x > d); return [i for i in range(len(net.key)) if net.depth[i] == nd]


def cleanup(net):
    """Hidden nodes with no input links are constants (folded into their children's bias); no output links -> gone."""
    while True:
        H = net.hidden()
        dead_in = [i for i in H if net.M[:, i].sum() == 0]
        for i in dead_in: net.b += max(net.b[i], 0) * net.W[i] * net.M[i]
        dead_out = [i for i in H if net.M[i].sum() == 0]
        gone = sorted(set(dead_in + dead_out))
        if not gone: return
        net.remove(gone)


def m_delete(net, i, ctx):
    n = net.copy(); ch = [n.key[j] for j in sorted(set(children(n, [i])) | set(next_layer(n, n.depth[i])))]; gone = n.key[i]
    n.remove([i]); refit_down(n, ch, ctx); cleanup(n)
    return n, dict(type='delete', id=gone, depth=float(net.depth[i]))


def m_links(net, g, ctx, frac=0.1):
    """Cut the weakest 10% of the links into depth group g (strength = |w| * spread of the input)."""
    s, e = net.groups()[g]; n = net.copy(); act = np.argwhere(n.M[:, s:e] > 0); act[:, 1] += s
    if len(act) < 12: return None
    sc = np.abs(n.W[act[:, 0], act[:, 1]]) * (ctx.A[:, [ctx.at[net.key[i]] for i in act[:, 0]]].std(0) + 1e-9)
    kill = act[np.argsort(sc)[:max(4, int(frac * len(act)))]]
    n.M[kill[:, 0], kill[:, 1]] = 0; n.W[kill[:, 0], kill[:, 1]] = 0
    refit(n, list(range(s, e)), n.forward(Xtr)[0], ctx.z(n.key[s:e]))
    rec = dict(type='links', depth=float(net.depth[s]), removed=[[net.key[i], net.key[j]] for i, j in kill])
    cleanup(n); return n, rec


def placement(net, dS, place, skip):
    """Depth of the new neuron, the nodes it may read from, and the nodes it may feed (besides the replaced ones' children)."""
    ds = net.depths(); k = ds.index(dS)
    if place == 'same': dn = dS
    else:
        if len([d for d in ds if d not in (0., 1., 2., OUT)]) >= MAX_NEW_LAYERS: return None
        dn = (dS + ds[k + 1]) / 2
    prev = max(d for d in ds if d < dn) if place == 'same' else dS
    nxt = min(d for d in ds if d > dn)
    ins = [i for i in range(len(net.key)) if (net.depth[i] < dn if skip else net.depth[i] == prev)]
    outs = [i for i in range(len(net.key)) if net.depth[i] == nxt]
    return dn, ins, outs


def m_merge_multi(net, S, ctx, place, skip, m_list=(1, 2, 4, 8, 16)):
    """Delete neurons S (one layer) and grow ONE new neuron in their place, NEAT add-connection style: it starts
    from its best single input link and adds the next best one at a time. Every prefix of that growth is a candidate."""
    dS = float(net.depth[S[0]]); pl = placement(net, dS, place, skip)
    if pl is None: return []
    dn, ins, outs = pl
    gone = [net.key[i] for i in S]; outk = sorted({net.key[j] for j in children(net, S)} | {net.key[j] for j in outs})
    ink = [net.key[i] for i in ins if i not in S]
    n0 = net.copy(); n0.remove(S)
    A2, _ = n0.forward(Xtr); tmp = n0.copy(); refit_down(tmp, outk, ctx)
    at = tmp.idx(); _, Z3 = tmp.forward(Xtr)
    R = ctx.z(outk) - Z3[:, [at[k] for k in outk]]
    Rc = R - R.mean(0); U, s, _ = np.linalg.svd(Rc, full_matrices=False); t = U[:, 0] * s[0]
    at0 = n0.idx(); X = A2[:, [at0[k] for k in ink]]; Xc = X - X.mean(0); nrm = np.linalg.norm(Xc, axis=0) + 1e-9
    I, r = [], t.copy()
    while len(I) < max(m_list):
        c = np.abs(Xc.T @ r) / nrm; c[I] = -1; j = int(np.argmax(c))
        if c[j] <= 1e-9: break
        I.append(j); w = np.linalg.lstsq(Xc[:, I], t, rcond=None)[0]; r = t - Xc[:, I] @ w
    new_layer = dn not in net.depths()
    out = []
    for m in m_list:
        if m > len(I): break
        J = I[:m]; n = n0.copy(); w = np.linalg.lstsq(Xc[:, J], t, rcond=None)[0]; z = X[:, J] @ w
        bias = -X.mean(0)[J] @ w; bias += max(0.0, -np.quantile(z + bias, 0.05))
        key = add_neuron(n, dn, [ink[j] for j in J], w, bias, outk, ctx)
        out.append((n, dict(type='merge', depth=dn, removed=gone, new=key if key in n.idx() else None,
                            links=[ink[j] for j in J], new_layer=bool(new_layer))))
    return out


def add_neuron(n, dn, src, w, bias, outk, ctx):
    key = f'g{n.next_id}'; n.next_id += 1; p = n.insert(key, dn); at = n.idx()
    n.W[[at[k] for k in src], p] = w; n.M[[at[k] for k in src], p] = 1; n.b[p] = bias
    oi = [at[k] for k in outk]; n.M[p, oi] = 1
    refit_down(n, outk, ctx)
    A3, _ = n.forward(Xtr); at = n.idx(); p = at[key]                      # keep the output links that carry 95%
    oi = np.flatnonzero(n.M[p]); e = n.W[p, oi] ** 2 * A3[:, p].var(); order = np.argsort(-e)
    keep = oi[order[:int(np.searchsorted(np.cumsum(e[order]) / (e.sum() + 1e-12), 0.95)) + 1]]
    n.M[p] = 0; n.M[p, keep] = 1; refit_down(n, outk, ctx); cleanup(n)
    return key


def refine(net, rec, ctx):
    """Post-hoc: let the chosen new neuron use its ReLU (fit its input weights to what its children need)."""
    key = rec.get('new'); at = net.idx()
    if key is None or key not in at: return net
    p = at[key]; n = net.copy(); A, _ = n.forward(Xtr)
    J = np.flatnonzero(n.M[:, p]); ch = sorted(set(children(n, [p])) | set(next_layer(n, n.depth[p])))
    if not len(J) or not ch: return net
    src = sorted({int(i) for j in ch for i in np.flatnonzero(n.M[:, j])} - {p})
    Tz = ctx.z([n.key[j] for j in ch]); XI = A[:, J]; Ar = A[:, src]; w = n.W[J, p].copy(); bias = float(n.b[p])
    mw, vw, mb, vb = 0 * w, 0 * w, 0.0, 0.0
    for it in range(1, 81):
        zz = XI @ w + bias; a = np.maximum(zz, 0); B = np.c_[Ar, a, np.ones(len(a))]
        V = np.linalg.solve(B.T @ B + 1e-6 * np.eye(B.shape[1]), B.T @ Tz)
        E = B @ V - Tz; ga = (E @ V[-2]) * (zz > 0) / len(a); gw, gb = XI.T @ ga, ga.sum()
        mw = .9 * mw + .1 * gw; vw = .999 * vw + .001 * gw ** 2; mb = .9 * mb + .1 * gb; vb = .999 * vb + .001 * gb ** 2
        w = w - 0.03 * (mw / (1 - .9 ** it)) / (np.sqrt(vw / (1 - .999 ** it)) + 1e-8)
        bias = bias - 0.03 * (mb / (1 - .9 ** it)) / (np.sqrt(vb / (1 - .999 ** it)) + 1e-8)
    n.W[J, p] = w; n.b[p] = bias; refit_down(n, [n.key[j] for j in ch], ctx)
    return n if loss_of(n) < loss_of(net) else net


def groups(net, s, e, A, del_scores):
    """Candidate groups to merge in one layer: the weakest neurons plus the neurons most correlated with them,
    and every single neuron (re-grow it sparsely)."""
    H = A[:, s:e]; nn = e - s
    if nn < 3: return [[s + j] for j in range(nn)]
    C_ = np.abs(np.nan_to_num(np.corrcoef(H.T))); order = np.argsort(del_scores)
    out = set()
    for seed in order[:3]:
        near = [int(j) for j in np.argsort(-C_[seed]) if j != seed]
        for k in (2, 3, 4, 6):
            if k <= nn - 1: out.add(tuple(sorted([int(seed)] + near[:k - 1])))
    for k in (2, 3, 4, 6):
        if k <= nn - 1: out.add(tuple(sorted(int(j) for j in order[:k])))
    out.update((int(j),) for j in range(nn))
    return [[s + j for j in g] for g in out]


VARIANTS = dict(flat=(('same',), False), skip=(('same',), True), deep=(('same', 'after'), True))


def search(net, target, variant, log):
    places, skip = VARIANTS[variant]
    while net.count() > target:
        ctx = Ctx(net); L0 = loss_of(net); P0 = net.count(); cands = []
        gr = net.groups()
        for g, (s, e) in enumerate(gr):
            if not 0 < net.depth[s] < OUT: continue
            ds = []
            for i in range(s, e):
                c = m_delete(net, i, ctx); d = loss_of(c[0]) - L0; ds.append(d)
                if len(net.hidden()) > 2: cands.append((d, P0 - c[0].count(), c))
            for S in groups(net, s, e, ctx.A, np.array(ds)):
                for place in places:
                    for c in m_merge_multi(net, S, ctx, place, skip): cands.append((loss_of(c[0]) - L0, P0 - c[0].count(), c))
        for g in range(1, len(gr)):
            c = m_links(net, g, ctx)
            if c: cands.append((loss_of(c[0]) - L0, P0 - c[0].count(), c))
        cands = [c for c in cands if c[1] > 0]
        if not cands: break
        top = sorted(cands, key=lambda c: c[0] / c[1])[:6]; rescored = []      # look ahead: 1 epoch of training each
        for d, dp, (cn, rec) in top:
            if rec['type'] == 'merge': cn = refine(cn, rec, ctx); d = loss_of(cn) - L0
            probe = cn.copy(); finetune(probe, 1, 7); rescored.append(((loss_of(probe) - L0) / dp, d, dp, cn, rec))
        _, d, dp, net, rec = min(rescored, key=lambda c: c[0])
        if rec['type'] == 'merge' and rec.get('new') in net.idx():          # keep the new neuron's weights for the replay
            at = net.idx(); p = at[rec['new']]
            rec['win'] = [[net.key[i], round(float(net.W[i, p]), 4)] for i in np.flatnonzero(net.M[:, p])]
            rec['wout'] = [[net.key[j], round(float(net.W[p, j]), 4)] for j in np.flatnonzero(net.M[p])]
        rec.update(dloss=round(d, 5), saved=int(dp)); log.append(rec)
    return net


def finetune(net, epochs, seed):
    W = torch.tensor(net.W, requires_grad=True); b = torch.tensor(net.b, requires_grad=True)
    M = torch.tensor(net.M); X = torch.tensor(Xtr); y = torch.tensor(ytr)
    opt = torch.optim.Adam([W, b], lr=1e-3); g = torch.Generator().manual_seed(seed); gr = net.groups()
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for s0 in range(0, len(perm), 64):
            bi = perm[s0:s0 + 64]; opt.zero_grad(); WM = W * M; A = [X[bi]]
            for s, e in gr[1:]:
                z = torch.cat(A, 1) @ WM[:s, s:e] + b[s:e]
                A.append(z if net.depth[s] == OUT else F.relu(z))
            F.cross_entropy(A[-1], y[bi]).backward(); opt.step()
    net.W = W.detach().numpy() * net.M; net.b = b.detach().numpy().copy()


def importance(net):
    A, _ = net.forward(Xtr); z = A[:, -10:]; p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True); lp = np.log(p + 1e-12)
    def kl(z2):
        q = np.exp(z2 - z2.max(1, keepdims=True)); q /= q.sum(1, keepdims=True); return float((p * (lp - np.log(q + 1e-12))).sum(1).mean())
    imp = {net.key[i]: round(kl(net.forward(Xtr, fix={i: A[:, i].mean()})[0][:, -10:]), 5) for i in net.hidden()}
    pix = []
    for i in range(64):
        Xa = Xtr.copy(); Xa[:, i] = Xtr[:, i].mean(); pix.append(round(kl(net.logits(Xa)), 5))
    return imp, pix


def edges(net):
    return [[net.key[i], net.key[j], round(float(net.W[i, j]), 4)] for i, j in np.argwhere(net.M > 0)]


def state(net):
    imp, pix = importance(net)
    return dict(acc=round(acc_of(net), 4), params=net.count(), nodes=[[net.key[i], float(net.depth[i])] for i in net.hidden()],
                imp=imp, pix=pix, edges=edges(net))


def run(base, seed, variant, record=False):
    net = DNet(base); out = []
    for r, target in enumerate(TARGETS):
        t0 = time.time(); log = []
        net = search(net, target, variant, log)
        step = dict(round=r + 1, log=log, after_cut=dict(acc=round(acc_of(net), 4), params=net.count()))
        if record: step['cut'] = dict(edges=edges(net), nodes=[[net.key[i], float(net.depth[i])] for i in net.hidden()])
        finetune(net, FT_EPOCHS, seed * 100 + r)
        step.update(state(net) if record else dict(acc=round(acc_of(net), 4), params=net.count()))
        step['layers'] = sorted({float(net.depth[i]) for i in net.hidden()})
        out.append(step)
        kinds = {t: sum(1 for x in log if x['type'] == t) for t in ('merge', 'delete', 'links')}
        newl = sum(1 for x in log if x['type'] == 'merge' and x.get('new_layer'))
        print(f'  {variant} seed {seed} round {r + 1}: params {step["params"]} acc {step["acc"]:.4f} '
              f'(cut {step["after_cut"]["acc"]:.4f}) {kinds} new layers {newl} depths {step["layers"]} {time.time() - t0:.1f}s', flush=True)
    return out


if __name__ == '__main__':
    variants = [a for a in sys.argv[1:] if not a.isdigit()] or ['deep']
    seeds = [int(a) for a in sys.argv[1:] if a.isdigit()] or [0, 1, 2]
    path = os.path.join(HERE, os.environ.get('ND_OUT', 'neat_deep.json'))
    for seed in seeds:
        base = NM.base_net(seed); b0 = [DNet(base).count(), round(acc_of(DNet(base)), 4)]
        print(f'seed {seed}: base acc {b0[1]}', flush=True)
        for v in variants:
            rec = seed == 0 and v == os.environ.get('ND_REC', '')
            steps = run(base, seed, v, record=rec)
            res = json.load(open(path)) if os.path.exists(path) else dict(curves={}, record=None)   # re-read: other runs may write too
            res['curves'].setdefault(v, {})[str(seed)] = [b0] + [[s['params'], s['acc']] for s in steps]
            res.setdefault('depths', {}).setdefault(v, {})[str(seed)] = [s['layers'] for s in steps]
            if rec:
                d0 = DNet(base); res['record'] = dict(variant=v, base=state(d0), steps=steps)
            json.dump(res, open(path, 'w'), separators=(',', ':'))
    print('wrote', path)
