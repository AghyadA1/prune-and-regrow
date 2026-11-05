"""Compare three ways of pruning a trained MLP on 8x8 digits, round by round:

  magnitude  delete the smallest weights, then fine-tune
  neuron     delete the least important neurons (mean-ablation effect on output), then fine-tune
  neat       delete dead neurons (no effect on output), replace the least important cluster in each
             hidden layer with a small subnetwork grown NEAT-style from one node, then fine-tune
  neat_lean  same, but growth stops at a looser error target and a smaller size budget

Every round fine-tunes for the same number of epochs. Writes compare.json (curves plus a
step-by-step record of seed 0 for the 3D viewer).

  python3 compare.py            # 3 seeds, ~10 minutes on a laptop CPU
Needs torch, numpy, scipy, scikit-learn.
"""
import json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import linkage, fcluster

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(4)
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X / 16.0, y, test_size=0.3, random_state=0, stratify=y)
Xtr, Xte = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xte, dtype=torch.float32)
ytr, yte = torch.tensor(ytr), torch.tensor(yte)
SIZES = [64, 32, 32, 10]
ROUNDS, FT_EPOCHS = 6, 20
MAG_SPARSITY = [0.5, 0.7, 0.8, 0.88, 0.93, 0.96]
NEURONS_PER_ROUND = 8


class Module:
    """A grown replacement for a cluster of hidden units. Genome = sparse input->hidden and hidden->unit links."""
    def __init__(self, layer, units, d_in):
        self.layer, self.units, self.d_in, self.k = layer, list(units), d_in, len(units)
        self.H = 0; self.A = torch.zeros(d_in, 0); self.Am = torch.zeros(d_in, 0)
        self.bh = torch.zeros(0); self.B = torch.zeros(0, self.k); self.Bm = torch.zeros(0, self.k); self.bB = torch.zeros(self.k)
        self.events = []

    def params(self):
        return [self.A, self.bh, self.B, self.bB]

    def forward(self, h):
        hid = F.relu(h @ (self.A * self.Am) + self.bh)
        return hid @ (self.B * self.Bm) + self.bB

    def count(self):
        return int(self.Am.sum() + self.Bm.sum()) + self.H + self.k

    def clone(self):
        m = Module(self.layer, self.units, self.d_in); m.H = self.H
        for n in ('A', 'Am', 'bh', 'B', 'Bm', 'bB'): setattr(m, n, getattr(self, n).detach().clone())
        m.events = list(self.events); return m


class Net:
    def __init__(self, seed):
        g = torch.Generator().manual_seed(seed)
        self.W = [torch.randn(a, b, generator=g) * np.sqrt(2 / a) for a, b in zip(SIZES[:-1], SIZES[1:])]
        self.b = [torch.zeros(b) for b in SIZES[1:]]
        self.M = [torch.ones_like(w) for w in self.W]
        self.alive = [torch.ones(32), torch.ones(32)]
        self.mods = []

    def forward(self, x, ablate=None, return_acts=False):
        """ablate = (layer, unit_list, values) replaces those hidden activations."""
        h, acts = x, [x]
        for l in range(3):
            z = h @ (self.W[l] * self.M[l]) + self.b[l]
            if l < 2:
                mods = [m for m in self.mods if m.layer == l]
                if mods:
                    z = z.clone()
                    for m in mods: z[:, m.units] = m.forward(h)
                z = F.relu(z) * self.alive[l]
                if ablate and ablate[0] == l:
                    z = z.clone(); z[:, ablate[1]] = ablate[2]
            h = z; acts.append(h)
        return (h, acts) if return_acts else h

    def tensors(self):
        return self.W + self.b + [p for m in self.mods for p in m.params()]

    def count(self):
        n = sum(int(m.sum()) for m in self.M) + 10
        in_mod = [set(u for m in self.mods if m.layer == l for u in m.units) for l in (0, 1)]
        for l in (0, 1):
            n += sum(1 for j in range(32) if self.alive[l][j] > 0 and j not in in_mod[l] and self.M[l][:, j].sum() > 0)
        return n + sum(m.count() for m in self.mods)


def acc(net):
    with torch.no_grad(): return float((net.forward(Xte).argmax(1) == yte).float().mean())


def train(net, epochs, lr=3e-3, seed=0):
    ps = net.tensors()
    for p in ps: p.requires_grad_(True)
    opt = torch.optim.Adam(ps, lr=lr, weight_decay=0)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), generator=g)
        for s in range(0, len(perm), 64):
            bi = perm[s:s + 64]; opt.zero_grad()
            F.cross_entropy(net.forward(Xtr[bi]), ytr[bi]).backward(); opt.step()
    for p in ps: p.requires_grad_(False)


def importance(net):
    """Mean-ablation KL on the training set, per hidden unit and per input pixel."""
    with torch.no_grad():
        logits, acts = net.forward(Xtr, return_acts=True); p = F.log_softmax(logits, 1)
        out = []
        for l in (0, 1):
            A = acts[l + 1]; row = []
            for j in range(32):
                if net.alive[l][j] == 0: row.append(0.0); continue
                q = F.log_softmax(net.forward(Xtr, ablate=(l, [j], A[:, j].mean())), 1)
                row.append(float((p.exp() * (p - q)).sum(1).mean()))
            out.append(row)
        pix = []
        for i in range(64):
            Xa = Xtr.clone(); Xa[:, i] = Xtr[:, i].mean()
            q = F.log_softmax(net.forward(Xa), 1); pix.append(float((p.exp() * (p - q)).sum(1).mean()))
    return out, pix


def delete_unit(net, l, j, acts):
    mean = acts[l + 1][:, j].mean()
    net.b[l + 1] += mean * net.W[l + 1][j] * net.M[l + 1][j]
    net.M[l][:, j] = 0; net.M[l + 1][j] = 0; net.alive[l][j] = 0


def grow(Xin, T, d_in, layer, units, budget, rng, tol=0.03, max_gens=25, cands=8, steps=150):
    """NEAT-style growth: start with one hidden node, mutate (add input link, add node, add output link),
    keep the best mutant per generation while it lowers the local error."""
    k = T.shape[1]; var = float(T.var(0).sum()) + 1e-8
    def fit(m, steps):
        ps = m.params()
        for p in ps: p.requires_grad_(True)
        opt = torch.optim.Adam(ps, lr=0.02)
        for _ in range(steps):
            opt.zero_grad(); loss = ((F.relu(m.forward(Xin)) - T) ** 2).sum(1).mean(); loss.backward(); opt.step()
        for p in ps: p.requires_grad_(False)
        with torch.no_grad(): return float(((F.relu(m.forward(Xin)) - T) ** 2).sum(1).mean()) / var
    def top_inputs(m, n=16):
        with torch.no_grad(): R = T - F.relu(m.forward(Xin))
        Xc = Xin - Xin.mean(0); Rc = R - R.mean(0)
        c = (Xc.T @ Rc).abs().sum(1) / (Xc.norm(dim=0) + 1e-8)
        return [int(i) for i in torch.argsort(c, descending=True)[:n]]
    def add_node(m, i):
        m.A = torch.cat([m.A, torch.zeros(d_in, 1)], 1); m.Am = torch.cat([m.Am, torch.zeros(d_in, 1)], 1)
        m.A[i, m.H] = float(rng.normal(0, .5)); m.Am[i, m.H] = 1
        m.bh = torch.cat([m.bh, torch.tensor([0.1])])
        m.B = torch.cat([m.B, torch.tensor(rng.normal(0, .3, (1, k)), dtype=torch.float32)]); m.Bm = torch.cat([m.Bm, torch.ones(1, k)])
        m.events.append(['node', m.H, i]); m.H += 1
    m = Module(layer, units, d_in); m.bB = T.mean(0).clone()
    add_node(m, top_inputs(m)[0]); err = fit(m, steps)
    fails = 0
    for gen in range(max_gens):
        if err < tol or fails >= 3: break
        best = None; tops = top_inputs(m)
        for _ in range(cands):
            c = m.clone(); r = rng.random()
            if r < 0.6:  # add input -> hidden link
                free = [(i, h) for i in tops for h in range(c.H) if c.Am[i, h] == 0]
                if not free: continue
                i, h = free[rng.integers(len(free))]; c.Am[i, h] = 1; c.A[i, h] = float(rng.normal(0, .3)); c.events.append(['in', h, i])
            elif r < 0.85:  # add node
                add_node(c, tops[rng.integers(len(tops))])
            else:  # add hidden -> unit link (re-enable)
                free = [(h, o) for h in range(c.H) for o in range(k) if c.Bm[h, o] == 0]
                if not free: continue
                h, o = free[rng.integers(len(free))]; c.Bm[h, o] = 1; c.events.append(['out', h, o])
            if c.count() > budget: continue
            e = fit(c, steps // 2)
            if best is None or e < best[0]: best = (e, c)
        if best and best[0] < err * 0.98: err, m = best; fails = 0
        else: fails += 1
    # prune output links that carry almost nothing (NEAT's "disable gene"), then refit
    with torch.no_grad():
        small = (m.B.abs() < 0.02) & (m.Bm > 0); m.Bm[small] = 0
    err = fit(m, steps)
    return m, err


def edges_snapshot(net):
    out = []
    for l in range(3):
        W = (net.W[l] * net.M[l]); s = float(W.abs().max()) or 1
        out.append(dict(scale=round(s, 3), q=[int(v) for v in torch.round(W / s * 99).flatten().tolist()]))
    return out


def mods_snapshot(net):
    res = []
    for m in net.mods:
        A, B = m.A * m.Am, m.B * m.Bm
        res.append(dict(layer=m.layer, units=m.units, H=m.H, events=m.events,
                        ins=[[int(i), int(h), round(float(A[i, h]), 3)] for i, h in zip(*torch.nonzero(m.Am, as_tuple=True))],
                        outs=[[int(h), int(o), round(float(B[h, o]), 3)] for h, o in zip(*torch.nonzero(m.Bm, as_tuple=True))]))
    return res


def state(net, label):
    imp, pix = importance(net)
    return dict(label=label, acc=round(acc(net), 4), params=net.count(), imp=[[round(v, 5) for v in r] for r in imp],
                pix=[round(v, 5) for v in pix], alive=[[int(v) for v in a.tolist()] for a in net.alive],
                edges=edges_snapshot(net), mods=mods_snapshot(net))


def run(method, base, seed, record):
    net = Net(0); net.W = [w.clone() for w in base.W]; net.b = [b.clone() for b in base.b]
    rng = np.random.default_rng(seed)
    steps = []
    for r in range(ROUNDS):
        t0 = time.time()
        with torch.no_grad(): _, acts = net.forward(Xtr, return_acts=True)
        imp, _ = importance(net)
        step = dict(round=r + 1)
        if method == 'magnitude':
            live = torch.cat([(w * m).abs()[m > 0] for w, m in zip(net.W, net.M)])
            total = sum(m.numel() for m in net.M); n_keep = int(round(total * (1 - MAG_SPARSITY[r])))
            thr = torch.sort(live, descending=True).values[min(n_keep, len(live) - 1)]
            marked = []
            for l in range(3):
                kill = ((net.W[l] * net.M[l]).abs() <= thr) & (net.M[l] > 0)
                marked.append([int(i) * SIZES[l + 1] + int(j) for i, j in zip(*torch.nonzero(kill, as_tuple=True))])
                net.M[l][kill] = 0
            # units with no incoming or outgoing links are gone too
            for l in (0, 1):
                for j in range(32):
                    if net.alive[l][j] and (net.M[l][:, j].sum() == 0 or net.M[l + 1][j].sum() == 0): delete_unit(net, l, j, acts)
            step['marked_edges'] = marked
        elif method == 'neuron':
            cand = sorted([(imp[l][j], l, j) for l in (0, 1) for j in range(32) if net.alive[l][j]])
            # keep at least 4 units per layer
            chosen, left = [], [int(net.alive[0].sum()), int(net.alive[1].sum())]
            for v, l, j in cand:
                if len(chosen) == NEURONS_PER_ROUND: break
                if left[l] > 4: chosen.append([l, j]); left[l] -= 1
            for l, j in chosen: delete_unit(net, l, j, acts)
            step['marked_units'] = chosen
        else:
            in_mod = [set(u for m in net.mods if m.layer == l for u in m.units) for l in (0, 1)]
            dead = [[l, j] for l in (0, 1) for j in range(32) if net.alive[l][j] and j not in in_mod[l] and imp[l][j] < 1e-4]
            for l, j in dead: delete_unit(net, l, j, acts)
            replaced = []
            for l in (0, 1):
                elig = [j for j in range(32) if net.alive[l][j] and j not in in_mod[l]]
                if len(elig) < 3: continue
                A = acts[l + 1][:, elig].numpy(); C = np.nan_to_num(np.corrcoef(A.T)); D = 1 - np.abs(C)
                Z = linkage(D[np.triu_indices(len(elig), 1)], 'average')
                lab = fcluster(Z, t=max(2, len(elig) // 5), criterion='maxclust')
                groups = [[elig[i] for i in np.where(lab == c)[0]] for c in np.unique(lab)]
                groups = [g for g in groups if len(g) >= 3] or [sorted(elig, key=lambda j: imp[l][j])[:3]]
                g = min(groups, key=lambda g: np.mean([imp[l][j] for j in g]))[:8]
                Xin = acts[l]; T = acts[l + 1][:, g]
                lean = method == 'neat_lean'  # leaner growth: stop earlier, smaller budget
                budget = int((0.25 if lean else 0.5) * (SIZES[l] * len(g)))  # must be much cheaper than the dense block
                mod, err = grow(Xin, T, SIZES[l], l, g, budget, rng, tol=0.15 if lean else 0.03)
                net.M[l][:, g] = 0; net.mods.append(mod)
                replaced.append(dict(layer=l, units=g, H=mod.H, local_err=round(err, 4), params_before=SIZES[l] * len(g) + len(g), params_after=mod.count()))
            step['marked_units'] = dead; step['replaced'] = replaced
        step['after_cut'] = dict(acc=round(acc(net), 4), params=net.count())
        if record: step['cut_edges'] = edges_snapshot(net)
        train(net, FT_EPOCHS, lr=1e-3, seed=seed * 100 + r)
        st = state(net, f'round {r + 1}') if record else dict(acc=round(acc(net), 4), params=net.count())
        step.update(st); steps.append(step)
        print(f'  {method} seed {seed} round {r + 1}: params {step["params"]} acc {step["acc"]:.4f} (cut {step["after_cut"]["acc"]:.4f}) {time.time() - t0:.1f}s', flush=True)
    return steps


def main(seeds):
    out = dict(sizes=SIZES, rounds=ROUNDS, ft_epochs=FT_EPOCHS, curves={}, record=None)
    for seed in seeds:
        torch.manual_seed(seed)
        base = Net(seed); train(base, 150, seed=seed)
        b0 = dict(acc=round(acc(base), 4), params=base.count())
        print(f'seed {seed}: trained, acc {b0["acc"]}, params {b0["params"]}', flush=True)
        rec = dict(base=state(base, 'trained')) if seed == seeds[0] else None
        for method in ('magnitude', 'neuron', 'neat', 'neat_lean'):
            steps = run(method, base, seed, record=seed == seeds[0] and method != 'neat_lean')
            out['curves'].setdefault(method, []).append([[b0['params'], b0['acc']]] + [[s['params'], s['acc']] for s in steps])
            if rec is not None and method != 'neat_lean': rec[method] = steps
        if rec is not None: out['record'] = rec
        json.dump(out, open(os.path.join(HERE, 'compare.json'), 'w'), separators=(',', ':'))
    print('wrote compare.json')


if __name__ == '__main__':
    main([int(s) for s in sys.argv[1:]] or [0, 1, 2])
