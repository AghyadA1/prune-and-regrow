"""Toy demo (needs numpy, scipy, scikit-learn; matplotlib for the heatmap):
  python3 demo.py
Train an MLP on sklearn digits, score each hidden neuron and each
cluster of neurons, then try replacing the weakest cluster with a ladder of
simpler surrogates (mean, linear, small grown net). Writes results.json."""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import linkage, fcluster

rng = np.random.default_rng(0)
X, y = load_digits(return_X_y=True)
X = X / 16.0
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
sizes = [64, 32, 32, 10]

def init():
    return [(rng.normal(0, np.sqrt(2 / a), (a, b)), np.zeros(b)) for a, b in zip(sizes[:-1], sizes[1:])]

def forward(P, X, hook=None):
    acts, h = [X], X
    for i, (W, b) in enumerate(P):
        z = h @ W + b
        h = np.maximum(z, 0) if i < len(P) - 1 else z
        if hook and i in hook: h = hook[i](acts[-1], h)
        acts.append(h)
    return acts

def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def train(P, epochs=300, lr=3e-3):
    m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in P]; v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in P]
    Y = np.eye(10)[ytr]; t = 0
    for ep in range(epochs):
        idx = rng.permutation(len(Xtr))
        for s in range(0, len(idx), 64):
            bi = idx[s:s + 64]; acts = forward(P, Xtr[bi]); g = (softmax(acts[-1]) - Y[bi]) / len(bi); t += 1
            for i in range(len(P) - 1, -1, -1):
                W, b = P[i]; gW = acts[i].T @ g + 1e-4 * W; gb = g.sum(0)
                if i > 0: g = (g @ W.T) * (acts[i] > 0)
                for k, gr in ((0, gW), (1, gb)):
                    mm = list(m[i]); vv = list(v[i]); mm[k] = .9 * mm[k] + .1 * gr; vv[k] = .999 * vv[k] + .001 * gr ** 2
                    m[i] = tuple(mm); v[i] = tuple(vv)
                    upd = lr * (mm[k] / (1 - .9 ** t)) / (np.sqrt(vv[k] / (1 - .999 ** t)) + 1e-8)
                    P[i] = (P[i][0] - upd, P[i][1]) if k == 0 else (P[i][0], P[i][1] - upd)
    return P

P = train(init())
acts_te = forward(P, Xte); acts_tr = forward(P, Xtr)
base_logits = acts_te[-1]; base_p = softmax(base_logits)
acc = lambda logits: float((logits.argmax(1) == yte).mean())
base_acc = acc(base_logits)
kl = lambda p, q: float(np.mean(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)), 1)))

def ablate(layer, units, fn=None):
    """layer = index into P whose output is replaced; units = neuron ids."""
    means = acts_tr[layer + 1][:, units].mean(0)
    def hook(hin, h):
        h = h.copy(); h[:, units] = fn(hin) if fn else means; return h
    out = forward(P, Xte, {layer: hook})[-1]
    return acc(out), kl(base_p, softmax(out))

def entropy(a, bins=16):
    if a.max() - a.min() < 1e-9: return 0.0
    c, _ = np.histogram(a, bins=bins); p = c / c.sum(); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

# per-neuron scores
neurons = []
for L in (0, 1):
    A = acts_te[L + 1]; W_out = P[L + 1][0]
    # first-order effect: E|sum_j (a - abar) * dlogit/da| using gradient of base KL ~ use weight-norm proxy x deviation
    for j in range(A.shape[1]):
        a_kl = ablate(L, [j])
        neurons.append(dict(layer=L + 1, idx=j, entropy=entropy(A[:, j]), var=float(A[:, j].var()),
                            dead=float((A[:, j] <= 0).mean()), kl=a_kl[1], acc_drop=base_acc - a_kl[0],
                            proxy=float(np.abs(A[:, j] - A[:, j].mean()).mean() * np.linalg.norm(W_out[j]))))

def rank(x): return np.argsort(np.argsort(x))
def spearman(a, b): return float(np.corrcoef(rank(np.array(a)), rank(np.array(b)))[0, 1])
corr = dict(entropy_vs_kl=spearman([n['entropy'] for n in neurons], [n['kl'] for n in neurons]),
            var_vs_kl=spearman([n['var'] for n in neurons], [n['kl'] for n in neurons]),
            proxy_vs_kl=spearman([n['proxy'] for n in neurons], [n['kl'] for n in neurons]))

# clusters: per layer, agglomerative on activation correlation
clusters = []
for L in (0, 1):
    A = acts_tr[L + 1]; std = A.std(0); alive = np.where(std > 1e-6)[0]; deadu = np.where(std <= 1e-6)[0]
    C = np.corrcoef(A[:, alive].T); D = 1 - np.abs(np.nan_to_num(C))
    Z = linkage(D[np.triu_indices(len(alive), 1)], 'average'); lab = fcluster(Z, t=6, criterion='maxclust')
    groups = [list(map(int, alive[lab == k])) for k in np.unique(lab)]
    if len(deadu): groups.append(list(map(int, deadu)))
    for g in groups:
        a_, k_ = ablate(L, g)
        clusters.append(dict(layer=L + 1, units=g, kl=k_, acc_drop=base_acc - a_))
for i, c in enumerate(clusters): c['id'] = i
for n in neurons:
    n['cluster'] = next(c['id'] for c in clusters if c['layer'] == n['layer'] and n['idx'] in c['units'])

# replacement ladder on the most "replaceable" sizeable cluster: lowest KL per unit among clusters with >=4 units
cands = [c for c in clusters if len(c['units']) >= 4 and c['kl'] > 1e-3]
target = min(cands, key=lambda c: c['kl'] / len(c['units']))
L, U = target['layer'] - 1, target['units']
Hin_tr, Hin_te = acts_tr[L], acts_te[L]; T_tr = acts_tr[L + 1][:, U]
orig_params = (P[L][0].shape[0] + 1) * len(U)
ladder = []
a_, k_ = ablate(L, U); ladder.append(dict(name='Constant (mean)', params=len(U), acc=a_, kl=k_))
# linear least squares surrogate (post-ReLU target, clipped at 0)
Xa = np.c_[Hin_tr, np.ones(len(Hin_tr))]; Wl = np.linalg.lstsq(Xa, T_tr, rcond=None)[0]
lin = lambda h: np.maximum(np.c_[h, np.ones(len(h))] @ Wl, 0)
a_, k_ = ablate(L, U, lin); ladder.append(dict(name='Linear map', params=int(Wl.size), acc=a_, kl=k_))
# low-rank linear (rank r)
for r in (1, 2):
    Uu, S, Vt = np.linalg.svd(Wl, full_matrices=False); Wr = (Uu[:, :r] * S[:r]) @ Vt[:r]
    f = (lambda Wr: (lambda h: np.maximum(np.c_[h, np.ones(len(h))] @ Wr, 0)))(Wr)
    a_, k_ = ablate(L, U, f); ladder.append(dict(name=f'Rank-{r} linear', params=int(r * (Xa.shape[1] + len(U))), acc=a_, kl=k_))

# constructive growth (cascade-style, NEAT-flavoured): start from nothing, add one hidden unit at a time
# connected to only the k most useful inputs, keep the unit if it lowers local error.
def grow(max_units=8, k_in=8, tries=20):
    resid = T_tr - T_tr.mean(0); units = []; bias = T_tr.mean(0); hist = []
    def feats(H): return np.stack([np.maximum(H[:, s] @ w + b0, 0) for s, w, b0 in [(u[0], u[1], u[2]) for u in units]], 1) if units else np.zeros((len(H), 0))
    for _ in range(max_units):
        best = None
        for _ in range(tries):  # mutation: random sparse unit, refined by a few gradient steps on the residual
            s = rng.choice(Hin_tr.shape[1], k_in, replace=False); w = rng.normal(0, .5, k_in); b0 = 0.0
            for _ in range(60):
                z = Hin_tr[:, s] @ w + b0; h = np.maximum(z, 0)
                o = np.linalg.lstsq(np.c_[h], resid, rcond=None)[0]; e = h[:, None] * o - resid
                gh = (e @ o.T).ravel() * (z > 0); w -= .05 * Hin_tr[:, s].T @ gh / len(z); b0 -= .05 * gh.mean()
            h = np.maximum(Hin_tr[:, s] @ w + b0, 0)[:, None]; o = np.linalg.lstsq(h, resid, rcond=None)[0]
            err = float(((h @ o - resid) ** 2).mean())
            if best is None or err < best[0]: best = (err, s, w.copy(), b0)
        units.append(best[1:]); F = feats(Hin_tr); Wo = np.linalg.lstsq(np.c_[F, np.ones(len(F))], T_tr, rcond=None)[0]
        resid = T_tr - np.c_[F, np.ones(len(F))] @ Wo
        fn = (lambda Wo, us: (lambda H: np.maximum(np.c_[np.stack([np.maximum(H[:, s] @ w + b0, 0) for s, w, b0 in us], 1), np.ones(len(H))] @ Wo, 0)))(Wo, list(units))
        a_, k_ = ablate(L, U, fn)
        hist.append(dict(name=f'Grown net, {len(units)} hidden', params=int(len(units) * (k_in + 1) + Wo.size), acc=a_, kl=k_))
    return hist
ladder += grow()

out = dict(sizes=sizes, base_acc=base_acc, neurons=neurons, clusters=clusters, corr=corr,
           target=dict(id=target['id'], layer=target['layer'], units=U, orig_params=orig_params), ladder=ladder,
           weights=[np.round(W, 3).tolist() for W, _ in P[1:]] , w0_absmean=np.round(np.abs(P[0][0]).mean(0), 4).tolist())
json.dump(out, open(os.path.join(HERE, 'results.json'), 'w'))
print('base acc', base_acc, 'corr', corr)
print('target', target['layer'], U, 'orig params', orig_params)
for r in ladder: print(r)
print(sorted([(c['layer'], len(c['units']), round(c['kl'], 4), round(c['acc_drop'], 4)) for c in clusters]))

# heatmap of the network, one panel per score
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit('matplotlib not installed; skipping heatmap.png')
scores = [('kl', 'Effect on output (sqrt of ablation KL)'), ('proxy', 'Deviation x outgoing weight'), ('entropy', 'Activation entropy (bits)'), ('var', 'Activation variance')]
fig, axes = plt.subplots(1, 4, figsize=(18, 6))
W1, W2 = P[1][0], P[2][0]
for ax, (key, title) in zip(axes, scores):
    for W, x0, x1, n_out in ((W1, 1, 2, 32), (W2, 2, 3, 10)):
        m = np.abs(W).max()
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                a = abs(W[i, j]) / m
                if a > .3:
                    yj = j if n_out == 32 else 3 + j * 2.8
                    ax.plot([x0, x1], [i, yj], color='gray', alpha=(a - .3) * .7, lw=.6, zorder=1)
    v = np.array([n[key] for n in neurons]); v = np.sqrt(v) if key == 'kl' else v
    for L in (1, 2):
        sel = [k for k, n in enumerate(neurons) if n['layer'] == L]
        sc = ax.scatter([L] * len(sel), [neurons[k]['idx'] for k in sel], c=v[sel], cmap='viridis', vmin=v.min(), vmax=v.max(), s=60, zorder=2, edgecolors='k', linewidths=.3)
    ax.scatter([3] * 10, [3 + j * 2.8 for j in range(10)], c='white', edgecolors='k', s=80, zorder=2)
    ax.set_xticks([1, 2, 3], ['Hidden 1', 'Hidden 2', 'Output']); ax.set_yticks([]); ax.invert_yaxis(); ax.set_title(title)
    fig.colorbar(sc, ax=ax, fraction=.04)
plt.tight_layout(); plt.savefig(os.path.join(HERE, 'heatmap.png'), dpi=120)
print('wrote heatmap.png and results.json')
