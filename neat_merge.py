"""Merge-style NEAT pruning ("NEAT-merge").

The network keeps its layers, but every hidden layer can lose neurons, gain NEAT-grown neurons, and lose
links. Each step tries these mutations and applies the cheapest one:

  merge k   delete k old neurons and add ONE new neuron in their place. The new neuron starts with a single
            input link and grows links one at a time (NEAT add-connection) while each link helps.
  delete    delete one neuron, old or newly grown (so weak new neurons get removed again).
  links     delete the weakest 10% of links in one weight matrix.

After every mutation the surviving outgoing weights are re-solved (least squares) so the next layer receives
nearly the same signal as before. That is what makes the remaining neurons carry more of the work.

Criterion (one line): make the change that loses the least per parameter saved, measured after the neurons
that stay are re-fitted (least squares) to cover for the ones that left.

Variants (each adds one thing to the row above; mean test accuracy over 3 seeds in summary.json):
  merge_only     merge + delete grown neurons only (closest to the original idea)
  merge_del      + delete any neuron
  full           + trim links
  full2          + new neurons fitted through their ReLU, sparse outputs
  full3          + try new neurons with 1/2/4/8/16 links, polish the chosen one with Adam
  full3_look     + look ahead: the 6 best changes get 1 epoch of training before choosing  <- winner, 3D view
  no_merge       delete + trim only, no new neurons (ablation: is growing worth it?)
  no_merge_look  same with the look-ahead
  python3 neat_merge.py [variant ...] [seed ...]   # needs compare.py next to it; writes neat_merge.json
  env: NM_OUT=<file> output json (merged into if it exists), NM_REC=<variant> records seed 0 for the 3D view
"""
import json, os, sys, time, copy
import torch, torch.nn.functional as F
import numpy as np
import compare as C

HERE = os.path.dirname(os.path.abspath(__file__))
Xtr, ytr, Xte, yte = C.Xtr.double().numpy(), C.ytr.numpy(), C.Xte.double().numpy(), C.yte.numpy()
TARGETS = [1767, 1086, 740, 462, 289, 163]   # the parameter counts "delete small weights" reaches each round
FT_EPOCHS = C.FT_EPOCHS


class ENet:
    def __init__(self, base):
        self.W = [w.double().numpy().copy() for w in base.W]
        self.b = [b.double().numpy().copy() for b in base.b]
        self.M = [np.ones_like(w) for w in self.W]
        self.ids = [list(range(32)), list(range(32))]     # node ids per hidden layer; grown ones are >= 100
        self.next_id = [100, 100]

    def copy(self): return copy.deepcopy(self)

    def forward(self, X):
        A, Z = [X], []
        for k in range(3):
            z = A[-1] @ (self.W[k] * self.M[k]) + self.b[k]; Z.append(z)
            A.append(np.maximum(z, 0) if k < 2 else z)
        return A, Z

    def count(self):
        return int(sum(m.sum() for m in self.M)) + len(self.ids[0]) + len(self.ids[1]) + 10


def loss_of(net, X=Xtr, y=ytr):
    z = net.forward(X)[0][-1]; z = z - z.max(1, keepdims=True)
    return float(-(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean())


def acc_of(net, X=Xte, y=yte): return float((net.forward(X)[0][-1].argmax(1) == y).mean())


def refit(net, k, Ain, Zt):
    """Re-solve matrix k (and bias) so that Ain @ W + b ~ Zt, keeping the link mask."""
    n, d = Ain.shape; A1 = np.c_[Ain, np.ones(n)]
    G = A1.T @ A1; lam = 1e-3 * (np.trace(G) / (d + 1)); G += lam * np.eye(d + 1); G[-1, -1] -= lam
    rhs = A1.T @ Zt
    if net.M[k].all():
        sol = np.linalg.solve(G, rhs); net.W[k] = sol[:-1]; net.b[k] = sol[-1]; return
    for c in range(Zt.shape[1]):
        act = np.r_[np.where(net.M[k][:, c] > 0)[0], d]
        sol = np.linalg.solve(G[np.ix_(act, act)], rhs[act, c])
        net.W[k][:, c] = 0; net.W[k][act[:-1], c] = sol[:-1]; net.b[k][c] = sol[-1]


def remove_nodes(net, l, idx):
    keep = [i for i in range(len(net.ids[l])) if i not in set(idx)]
    net.W[l], net.M[l], net.b[l] = net.W[l][:, keep], net.M[l][:, keep], net.b[l][keep]
    net.W[l + 1], net.M[l + 1] = net.W[l + 1][keep], net.M[l + 1][keep]
    net.ids[l] = [net.ids[l][i] for i in keep]


def cleanup(net):
    """Neurons left with no input links become constants (folded into the next bias); no output links -> gone."""
    for l in (0, 1):
        dead_in = [i for i in range(len(net.ids[l])) if net.M[l][:, i].sum() == 0]
        for i in dead_in: net.b[l + 1] += max(net.b[l][i], 0) * net.W[l + 1][i] * net.M[l + 1][i]
        dead_out = [i for i in range(len(net.ids[l])) if net.M[l + 1][i].sum() == 0]
        if dead_in or dead_out: remove_nodes(net, l, sorted(set(dead_in + dead_out)))


def m_delete(net, l, i, A, Z):
    n = net.copy(); gone = n.ids[l][i]; remove_nodes(n, l, [i])
    A2, _ = n.forward(Xtr); refit(n, l + 1, A2[l + 1], Z[l + 1]); cleanup(n)
    return n, dict(type='delete', layer=l, id=gone)


def m_links(net, k, A, Z, frac=0.1):
    n = net.copy(); act = np.argwhere(n.M[k] > 0)
    if len(act) < 12: return None
    s = np.abs(n.W[k][act[:, 0], act[:, 1]]) * (A[k][:, act[:, 0]].std(0) + 1e-9)
    kill = act[np.argsort(s)[:max(4, int(frac * len(act)))]]
    n.M[k][kill[:, 0], kill[:, 1]] = 0; n.W[k][kill[:, 0], kill[:, 1]] = 0
    refit(n, k, A[k], Z[k])
    src = (lambda i: int(i)) if k == 0 else (lambda i: net.ids[k - 1][i])
    dst = (lambda j: int(j)) if k == 2 else (lambda j: net.ids[k][j])
    rec = dict(type='links', matrix=k, removed=[[src(i), dst(j)] for i, j in kill])
    cleanup(n); return n, rec


def m_merge(net, l, S, A, Z, m_max=16, gain=0.02, nonlin=False, sparse_out=False):
    """Delete neurons S of hidden layer l; grow ONE new neuron from a single link to replace them."""
    n = net.copy(); gone = [n.ids[l][i] for i in S]; remove_nodes(n, l, S)
    A2, _ = n.forward(Xtr)
    tmp = n.copy(); refit(tmp, l + 1, A2[l + 1], Z[l + 1])
    R = Z[l + 1] - (A2[l + 1] @ (tmp.W[l + 1] * tmp.M[l + 1]) + tmp.b[l + 1])
    Rc = R - R.mean(0); U, s, _ = np.linalg.svd(Rc, full_matrices=False); t = U[:, 0] * s[0]
    X = A2[l]; Xc = X - X.mean(0); nrm = np.linalg.norm(Xc, axis=0) + 1e-9
    I, r, err = [], t.copy(), float(t @ t)
    while len(I) < m_max:                       # NEAT add-connection, greedy: keep a link only if it helps
        c = np.abs(Xc.T @ r) / nrm; c[I] = -1; j = int(np.argmax(c))
        w = np.linalg.lstsq(Xc[:, I + [j]], t, rcond=None)[0]; r2 = t - Xc[:, I + [j]] @ w; e2 = float(r2 @ r2)
        if I and (err - e2) < gain * float(t @ t): break
        I.append(j); r, err = r2, e2
    w = np.linalg.lstsq(Xc[:, I], t, rcond=None)[0]
    z = X[:, I] @ w; bias = -X.mean(0)[I] @ w; bias += max(0.0, -np.quantile(z + bias, 0.05))
    if nonlin:   # let the new neuron use its ReLU: fit its input weights to the real target, re-solving outputs each step
        Ar = A2[l + 1]; XI = X[:, I]; mw, vw, mb, vb = 0 * w, 0 * w, 0.0, 0.0
        for it in range(1, 61):
            zz = XI @ w + bias; a = np.maximum(zz, 0); B = np.c_[Ar, a, np.ones(len(a))]
            V = np.linalg.solve(B.T @ B + 1e-6 * np.eye(B.shape[1]), B.T @ Z[l + 1])
            E = B @ V - Z[l + 1]; ga = (E @ V[-2]) * (zz > 0) / len(a)
            gw, gb = XI.T @ ga, ga.sum()
            mw = .9 * mw + .1 * gw; vw = .999 * vw + .001 * gw ** 2; mb = .9 * mb + .1 * gb; vb = .999 * vb + .001 * gb ** 2
            w = w - 0.03 * (mw / (1 - .9 ** it)) / (np.sqrt(vw / (1 - .999 ** it)) + 1e-8)
            bias = bias - 0.03 * (mb / (1 - .9 ** it)) / (np.sqrt(vb / (1 - .999 ** it)) + 1e-8)
    col = np.zeros(X.shape[1]); col[I] = w; mcol = np.zeros(X.shape[1]); mcol[I] = 1
    n.W[l] = np.c_[n.W[l], col]; n.M[l] = np.c_[n.M[l], mcol]; n.b[l] = np.r_[n.b[l], bias]
    nn = n.W[l + 1].shape[1]
    n.W[l + 1] = np.r_[n.W[l + 1], np.zeros((1, nn))]; n.M[l + 1] = np.r_[n.M[l + 1], np.ones((1, nn))]
    nid = n.next_id[l]; n.next_id[l] += 1; n.ids[l].append(nid)
    A3, _ = n.forward(Xtr); refit(n, l + 1, A3[l + 1], Z[l + 1])
    if sparse_out:  # keep only the output links that carry 95% of what the new neuron sends
        e = n.W[l + 1][-1] ** 2 * A3[l + 1][:, -1].var(); order = np.argsort(-e)
        keep = order[:int(np.searchsorted(np.cumsum(e[order]) / (e.sum() + 1e-12), 0.95)) + 1]
        n.M[l + 1][-1] = 0; n.M[l + 1][-1, keep] = 1; refit(n, l + 1, A3[l + 1], Z[l + 1])
    cleanup(n)
    src = (lambda i: int(i)) if l == 0 else (lambda i: net.ids[0][i])
    return n, dict(type='merge', layer=l, removed=gone, new=nid, links=[src(i) for i in I])


def m_merge_multi(net, l, S, A, Z, m_list=(1, 2, 4, 8, 16), sparse_out=True):
    """Like m_merge, but returns one candidate per link budget: the new neuron grows links one at a time
    (best link first) and every prefix of that growth is a candidate, so the criterion picks how far it grows."""
    n0 = net.copy(); gone = [n0.ids[l][i] for i in S]; remove_nodes(n0, l, S)
    A2, _ = n0.forward(Xtr)
    tmp = n0.copy(); refit(tmp, l + 1, A2[l + 1], Z[l + 1])
    R = Z[l + 1] - (A2[l + 1] @ (tmp.W[l + 1] * tmp.M[l + 1]) + tmp.b[l + 1])
    Rc = R - R.mean(0); U, s, _ = np.linalg.svd(Rc, full_matrices=False); t = U[:, 0] * s[0]
    X = A2[l]; Xc = X - X.mean(0); nrm = np.linalg.norm(Xc, axis=0) + 1e-9
    I, r = [], t.copy()
    while len(I) < max(m_list):
        c = np.abs(Xc.T @ r) / nrm; c[I] = -1; j = int(np.argmax(c))
        if c[j] <= 1e-9: break
        I.append(j); w = np.linalg.lstsq(Xc[:, I], t, rcond=None)[0]; r = t - Xc[:, I] @ w
    src = (lambda i: int(i)) if l == 0 else (lambda i: net.ids[0][i])
    out = []
    for m in m_list:
        if m > len(I): break
        J = I[:m]; n = n0.copy()
        w = np.linalg.lstsq(Xc[:, J], t, rcond=None)[0]; z = X[:, J] @ w
        bias = -X.mean(0)[J] @ w; bias += max(0.0, -np.quantile(z + bias, 0.05))
        add_neuron(n, l, J, w, bias, Z, sparse_out)
        out.append((n, dict(type='merge', layer=l, removed=gone, new=n.ids[l][-1] if n.ids[l] and n.ids[l][-1] >= 100 else None, links=[src(i) for i in J])))
    return out


def add_neuron(n, l, J, w, bias, Z, sparse_out):
    X = n.forward(Xtr)[0][l]
    col = np.zeros(X.shape[1]); col[J] = w; mcol = np.zeros(X.shape[1]); mcol[J] = 1
    n.W[l] = np.c_[n.W[l], col]; n.M[l] = np.c_[n.M[l], mcol]; n.b[l] = np.r_[n.b[l], bias]
    nn = n.W[l + 1].shape[1]
    n.W[l + 1] = np.r_[n.W[l + 1], np.zeros((1, nn))]; n.M[l + 1] = np.r_[n.M[l + 1], np.ones((1, nn))]
    nid = n.next_id[l]; n.next_id[l] += 1; n.ids[l].append(nid)
    A3, _ = n.forward(Xtr); refit(n, l + 1, A3[l + 1], Z[l + 1])
    if sparse_out:
        e = n.W[l + 1][-1] ** 2 * A3[l + 1][:, -1].var(); order = np.argsort(-e)
        keep = order[:int(np.searchsorted(np.cumsum(e[order]) / (e.sum() + 1e-12), 0.95)) + 1]
        n.M[l + 1][-1] = 0; n.M[l + 1][-1, keep] = 1; refit(n, l + 1, A3[l + 1], Z[l + 1])
    cleanup(n)


def refine(net, rec, Z):
    """Post-hoc: let the chosen new neuron use its ReLU (fit input weights to the real target)."""
    l = rec['layer']; ids = net.ids[l]
    if rec.get('new') is None or rec['new'] not in ids: return net
    i = ids.index(rec['new']); n = net.copy(); A, _ = n.forward(Xtr)
    J = np.where(n.M[l][:, i] > 0)[0]; XI = A[l][:, J]; w = n.W[l][J, i].copy(); bias = float(n.b[l][i])
    others = [k for k in range(len(ids)) if k != i]; Ar = A[l + 1][:, others]
    mw, vw, mb, vb = 0 * w, 0 * w, 0.0, 0.0
    for it in range(1, 81):
        zz = XI @ w + bias; a = np.maximum(zz, 0); B = np.c_[Ar, a, np.ones(len(a))]
        V = np.linalg.solve(B.T @ B + 1e-6 * np.eye(B.shape[1]), B.T @ Z[l + 1])
        E = B @ V - Z[l + 1]; ga = (E @ V[-2]) * (zz > 0) / len(a); gw, gb = XI.T @ ga, ga.sum()
        mw = .9 * mw + .1 * gw; vw = .999 * vw + .001 * gw ** 2; mb = .9 * mb + .1 * gb; vb = .999 * vb + .001 * gb ** 2
        w = w - 0.03 * (mw / (1 - .9 ** it)) / (np.sqrt(vw / (1 - .999 ** it)) + 1e-8)
        bias = bias - 0.03 * (mb / (1 - .9 ** it)) / (np.sqrt(vb / (1 - .999 ** it)) + 1e-8)
    n.W[l][J, i] = w; n.b[l][i] = bias
    A3, _ = n.forward(Xtr); refit(n, l + 1, A3[l + 1], Z[l + 1])
    return n if loss_of(n) < loss_of(net) else net


def groups(net, l, A, del_scores, singles=False):
    """Candidate groups to merge: the weakest neurons plus the neurons most correlated with them."""
    H = A[l + 1]; nn = H.shape[1]
    if nn < 3: return []
    C_ = np.abs(np.nan_to_num(np.corrcoef(H.T))); order = np.argsort(del_scores)
    out = set()
    for seed in order[:3]:
        near = [int(j) for j in np.argsort(-C_[seed]) if j != seed]
        for k in (2, 3, 4, 6):
            if k <= nn - 1: out.add(tuple(sorted([int(seed)] + near[:k - 1])))
    for k in (2, 3, 4, 6):
        if k <= nn - 1: out.add(tuple(sorted(int(j) for j in order[:k])))
    if singles: out.update((int(j),) for j in range(nn))   # re-grow any one neuron sparsely from scratch
    return [list(g) for g in out]


def search(net, target, variant, log):
    """Greedy NEAT search: apply the cheapest mutation until the parameter count reaches the target."""
    while net.count() > target:
        A, Z = net.forward(Xtr); L0 = loss_of(net); P0 = net.count(); cands = []
        del_scores = [[], []]
        for l in (0, 1):
            for i, nid in enumerate(net.ids[l]):
                c = m_delete(net, l, i, A, Z); d = loss_of(c[0]) - L0; del_scores[l].append(d)
                if len(net.ids[l]) > 2 and (variant != 'merge_only' or nid >= 100): cands.append((d, P0 - c[0].count(), c))
        if variant.startswith('full3'):
            for l in (0, 1):
                for g in groups(net, l, A, np.array(del_scores[l]), singles=True):
                    for c in m_merge_multi(net, l, g, A, Z): cands.append((loss_of(c[0]) - L0, P0 - c[0].count(), c))
        elif not variant.startswith('no_merge'):
            for l in (0, 1):
                for g in groups(net, l, A, np.array(del_scores[l])):
                    c = m_merge(net, l, g, A, Z, nonlin=variant.startswith('full2'), sparse_out=variant.startswith('full2')); cands.append((loss_of(c[0]) - L0, P0 - c[0].count(), c))
        if variant.startswith(('full', 'no_merge')):
            for k in range(3):
                c = m_links(net, k, A, Z)
                if c: cands.append((loss_of(c[0]) - L0, P0 - c[0].count(), c))
        cands = [c for c in cands if c[1] > 0]
        if not cands: break
        if variant.endswith('_look'):   # look ahead: try 1 epoch of training on the 6 best-looking changes, keep the best
            top = sorted(cands, key=lambda c: c[0] / c[1])[:6]; rescored = []
            for d, dp, (cn, rec) in top:
                if variant.startswith('full3') and rec['type'] == 'merge': cn = refine(cn, rec, Z); d = loss_of(cn) - L0
                probe = cn.copy(); finetune(probe, 1, 7); rescored.append(((loss_of(probe) - L0) / dp, d, dp, cn, rec))
            _, d, dp, net, rec = min(rescored, key=lambda c: c[0])
        else:
            d, dp, (net, rec) = min(cands, key=lambda c: c[0] / c[1])
            if variant == 'full3' and rec['type'] == 'merge': net = refine(net, rec, Z); d = loss_of(net) - L0
        if rec['type'] == 'merge' and rec.get('new') in net.ids[rec['layer']]:   # keep the new neuron's weights for the 3D replay
            l = rec['layer']; j = net.ids[l].index(rec['new'])
            src = list(range(64)) if l == 0 else net.ids[0]; dst = net.ids[1] if l == 0 else list(range(10))
            wi = net.W[l][:, j] * net.M[l][:, j]; wo = net.W[l + 1][j] * net.M[l + 1][j]
            rec['win'] = [[int(src[i]), round(float(wi[i]), 4)] for i in np.flatnonzero(wi)]
            rec['wout'] = [[int(dst[i]), round(float(wo[i]), 4)] for i in np.flatnonzero(wo)]
        rec.update(dloss=round(d, 5), saved=int(dp)); log.append(rec)
    return net


def finetune(net, epochs, seed):
    W = [torch.tensor(w, requires_grad=True) for w in net.W]; b = [torch.tensor(v, requires_grad=True) for v in net.b]
    M = [torch.tensor(m) for m in net.M]; X = torch.tensor(Xtr); y = torch.tensor(ytr)
    opt = torch.optim.Adam(W + b, lr=1e-3); g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for s in range(0, len(perm), 64):
            bi = perm[s:s + 64]; h = X[bi]; opt.zero_grad()
            for k in range(3):
                h = h @ (W[k] * M[k]) + b[k]
                if k < 2: h = F.relu(h)
            F.cross_entropy(h, y[bi]).backward(); opt.step()
    net.W = [w.detach().numpy() * m for w, m in zip(W, net.M)]; net.b = [v.detach().numpy() for v in b]


def importance(net):
    A, _ = net.forward(Xtr); z = A[-1]; p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True); lp = np.log(p + 1e-12)
    def kl(z2):
        q = np.exp(z2 - z2.max(1, keepdims=True)); q /= q.sum(1, keepdims=True); return float((p * (lp - np.log(q + 1e-12))).sum(1).mean())
    out = []
    for l in (0, 1):
        row = {}
        for i, nid in enumerate(net.ids[l]):
            h = A[l + 1].copy(); h[:, i] = h[:, i].mean(); z2 = h
            for k in range(l + 1, 3):
                z2 = z2 @ (net.W[k] * net.M[k]) + net.b[k]
                if k < 2: z2 = np.maximum(z2, 0)
            row[nid] = round(kl(z2), 5)
        out.append(row)
    pix = []
    for i in range(64):
        Xa = Xtr.copy(); Xa[:, i] = Xtr[:, i].mean(); pix.append(round(kl(net.forward(Xa)[0][-1]), 5))
    return out, pix


def edges(net):
    """Sparse edge list per matrix: [from_id, to_id, q] with q = w / scale * 99."""
    res = []
    for k in range(3):
        W = net.W[k] * net.M[k]; s = float(np.abs(W).max()) or 1
        src = list(range(64)) if k == 0 else net.ids[k - 1]; dst = list(range(10)) if k == 2 else net.ids[k]
        nz = np.argwhere(net.M[k] > 0)
        res.append(dict(scale=round(s, 3), e=[[int(src[i]), int(dst[j]), int(round(W[i, j] / s * 99))] for i, j in nz]))
    return res


def state(net):
    imp, pix = importance(net)
    return dict(acc=round(acc_of(net), 4), params=net.count(), ids=[list(map(int, i)) for i in net.ids],
                imp=[{str(k): v for k, v in r.items()} for r in imp], pix=pix, edges=edges(net))


def run(base, seed, variant, record=False):
    net = ENet(base); out = []
    for r, target in enumerate(TARGETS):
        t0 = time.time(); log = []
        net = search(net, target, variant, log)
        step = dict(round=r + 1, log=log, after_cut=dict(acc=round(acc_of(net), 4), params=net.count()))
        if record: step['cut'] = dict(edges=edges(net), ids=[list(map(int, i)) for i in net.ids])
        finetune(net, FT_EPOCHS, seed * 100 + r)
        step.update(state(net) if record else dict(acc=round(acc_of(net), 4), params=net.count()))
        out.append(step)
        kinds = {t: sum(1 for x in log if x['type'] == t) for t in ('merge', 'delete', 'links')}
        print(f'  {variant} seed {seed} round {r + 1}: params {step["params"]} acc {step["acc"]:.4f} '
              f'(cut {step["after_cut"]["acc"]:.4f}) {kinds} {time.time() - t0:.1f}s', flush=True)
    return out


def base_net(seed):
    torch.manual_seed(seed); base = C.Net(seed); C.train(base, 150, seed=seed); return base


if __name__ == '__main__':
    variants = [a for a in sys.argv[1:] if not a.isdigit()] or ['merge_only', 'merge_del', 'full', 'no_merge']
    seeds = [int(a) for a in sys.argv[1:] if a.isdigit()] or [0, 1, 2]
    path = os.path.join(HERE, os.environ.get('NM_OUT', 'neat_merge.json'))
    res = json.load(open(path)) if os.path.exists(path) else dict(curves={}, record=None)
    for seed in seeds:
        base = base_net(seed); b0 = [base.count(), round(C.acc(base), 4)]
        print(f'seed {seed}: base acc {b0[1]}', flush=True)
        for v in variants:
            rec = seed == 0 and v == os.environ.get('NM_REC', 'full3')
            steps = run(base, seed, v, record=rec)
            res['curves'].setdefault(v, {})[str(seed)] = [b0] + [[s['params'], s['acc']] for s in steps]
            if rec:
                ENet0 = ENet(base); res['record'] = dict(base=state(ENet0), steps=steps)
            json.dump(res, open(path, 'w'), separators=(',', ':'))
    print('wrote', path)
