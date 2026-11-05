"""Distill the original trained network (teacher, 64-32-32-10) into a small student whose topology NEAT builds.

Goal: a student SMALLER than the smallest NEAT-merge result (~160 parameters, 91.0% mean test accuracy, 3 seeds)
and MORE accurate. Same data split, same parameter counting (active links + hidden biases + 10 output biases).

How NEAT is pushed there
  hard cap      a genome never has more than CAP parameters: any mutation that goes over is repaired by deleting
                its weakest links (|w| * activity std) until it fits. So every network NEAT ever sees is small.
  fitness       distillation loss on held-out training images (KL to the teacher's softened outputs + a little
                cross-entropy) plus LAMBDA per parameter, so of two equally good students the smaller one wins.
  Lamarckian    each child is trained a few dozen Adam steps on the distillation loss and keeps its trained
                weights; NEAT only decides the topology.
  mutations     add link (mostly gradient-guided: the missing link whose gradient is largest), add neuron (split a
                link, NEAT style), delete weakest link, delete neuron; crossover by innovation number; speciation.
  transfer set  the training images, noisy copies and mixups, all labelled by the teacher. Test set unused.

Students (per seed; test accuracy):
  nm160          the smallest NEAT-merge network (full3_look, ~160 params), re-built here     <- the bar to beat
  nm160_hard/kd  the same network trained longer on labels / distilled                        (no NEAT search)
  nm160_cut_kd   nm160 with its weakest links cut to fit the cap, then distilled              (pruning alone, no NEAT)
  dense2_hard/kd plain fixed 64-2-10 network (160 params), trained on labels / distilled      (fixed-size baseline)
  sparse_hard/kd fixed 64-H-10 network with random links, same size and H as the NEAT winner  (fixed-size baseline)
  neat_hard      NEAT from scratch, trained on labels only                                    (NEAT alone, no KD)
  neat_kd        NEAT from scratch, distilled                                                 (the idea)
  neat_kd_seed   NEAT started from nm160 (pruned under the cap), distilled                    (the idea, warm start)

  neat_hard_seed NEAT started from nm160, trained on labels only                              (warm start, no KD)

  python3 distill.py [seed ...]          # env: DS_CAP (default 150), DS_GENS, DS_POP, DS_OUT, DS_ONLY=name,name
"""
import json, os, sys, time, copy, math
import numpy as np, torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.dont_write_bytecode = True
import compare as C
import neat_merge as NM

torch.set_num_threads(int(os.environ.get('DS_THREADS', '1')))
CAP = int(os.environ.get('DS_CAP', '150'))
ONLY = set(os.environ['DS_ONLY'].split(',')) if os.environ.get('DS_ONLY') else None   # subset of the students below
GENS = int(os.environ.get('DS_GENS', '40'))
POP = int(os.environ.get('DS_POP', '30'))
LAMBDA = 2e-4          # fitness cost per parameter (loss units)
TEMP, ALPHA = float(os.environ.get('DS_TEMP', '2')), float(os.environ.get('DS_ALPHA', '0.2'))  # distillation temperature, weight of the hard-label term
LR = float(os.environ.get('DS_LR', '0.005'))
WARM_LR = 1e-3
CHILD_STEPS, FINAL_STEPS = int(os.environ.get('DS_CHILD', '300')), int(os.environ.get('DS_FINAL', '6000'))
N_NOISE, N_MIX = 4, 8
NI, NO = 64, 10
OUT_IDS = list(range(NI, NI + NO))


# ---------------------------------------------------------------- data
def teacher_logits(teacher, X):
    with torch.no_grad(): return teacher.forward(X)


def make_data(teacher, seed):
    """Split the training images 80/20 (search / selection); build the teacher-labelled transfer sets."""
    g = torch.Generator().manual_seed(1000 + seed)
    perm = torch.randperm(len(C.Xtr), generator=g); nv = len(perm) // 5
    vi, ti = perm[:nv], perm[nv:]

    def build(idx):
        """Originals (real labels) + noisy copies + mixups, all labelled by the teacher. No shifts: the test digits
        are centred, and a 150-parameter student cannot afford to spend capacity on shift invariance."""
        X, y = C.Xtr[idx], C.ytr[idx]
        Xn = (X.repeat(N_NOISE, 1) + torch.randn(len(X) * N_NOISE, 64, generator=g) * 0.08).clamp(0, 1)
        Xr = X.repeat(N_MIX, 1); k = torch.randperm(len(Xr), generator=g); lam = torch.rand(len(Xr), 1, generator=g) * 0.5
        Xm = (1 - lam) * Xr + lam * Xr[k]
        Xk = torch.cat([X, Xn, Xm])
        yk = torch.cat([y, y.repeat(N_NOISE), torch.full((len(Xm),), -100)])
        return dict(X=Xk, T=teacher_logits(teacher, Xk), Xh=X, yh=y, y=yk)
    search = build(ti)
    val = dict(X=C.Xtr[vi], y=C.ytr[vi], T=teacher_logits(teacher, C.Xtr[vi]))
    full = build(perm)
    return search, val, full


def kd_loss(z, T, y=None):
    kl = F.kl_div(F.log_softmax(z / TEMP, 1), F.log_softmax(T / TEMP, 1), log_target=True, reduction='batchmean') * TEMP ** 2
    return kl + ALPHA * F.cross_entropy(z, y, ignore_index=-100) if y is not None and (y >= 0).any() else kl


# ---------------------------------------------------------------- NEAT genome
class Innov:
    """Global innovation registry: one number per (src, dst) link, one node id per split link."""
    def __init__(self): self.conn, self.split, self.next_node = {}, {}, NI + NO

    def c(self, a, b):
        if (a, b) not in self.conn: self.conn[(a, b)] = len(self.conn)
        return self.conn[(a, b)]

    def node(self, a, b, taken):
        k = 0
        while True:
            key = (a, b, k)
            if key not in self.split: self.split[key] = self.next_node; self.next_node += 1
            if self.split[key] not in taken: return self.split[key]
            k += 1


class Genome:
    def __init__(self):
        self.depth = {}      # hidden node id -> depth in (0, 1); inputs 0, outputs 1
        self.conn = {}       # (src, dst) -> weight
        self.bias = {o: 0.0 for o in OUT_IDS}

    def copy(self): return copy.deepcopy(self)

    def d(self, n): return 0.0 if n < NI else 1.0 if n < NI + NO else self.depth[n]

    def count(self): return len(self.conn) + len(self.depth) + NO

    def hidden(self): return sorted(self.depth, key=lambda n: (self.depth[n], n))

    def cleanup(self):
        """Hidden neurons without inputs become constants folded into their targets; without outputs they go."""
        changed = True
        while changed:
            changed = False
            for h in list(self.depth):
                ins = [k for k in self.conn if k[1] == h]; outs = [k for k in self.conn if k[0] == h]
                if not ins:
                    for k in outs: self.bias[k[1]] += max(self.bias[h], 0) * self.conn.pop(k)
                if not ins or not outs:
                    for k in ins: self.conn.pop(k)
                    del self.depth[h]; del self.bias[h]; changed = True


class Compiled:
    """Torch view of a genome: hidden neurons grouped by depth, each group one masked matmul."""
    def __init__(self, g):
        self.g = g; self.keys = list(g.conn); self.nodes = list(g.bias)
        self.w = torch.tensor([g.conn[k] for k in self.keys], dtype=torch.float32, requires_grad=True)
        self.b = torch.tensor([g.bias[n] for n in self.nodes], dtype=torch.float32, requires_grad=True)
        bpos = {n: i for i, n in enumerate(self.nodes)}
        hid = g.hidden(); levels = []
        for h in hid:
            if levels and g.depth[levels[-1][0]] == g.depth[h]: levels[-1].append(h)
            else: levels.append([h])
        levels.append(OUT_IDS)
        col = {i: i for i in range(NI)}; ncol = NI; self.plan = []
        for lv in levels:
            loc = {n: j for j, n in enumerate(lv)}
            ci = [i for i, k in enumerate(self.keys) if k[1] in loc]
            rows = torch.tensor([col[self.keys[i][0]] for i in ci], dtype=torch.long)
            cols = torch.tensor([loc[self.keys[i][1]] for i in ci], dtype=torch.long)
            self.plan.append((ncol, len(lv), torch.tensor(ci, dtype=torch.long), rows, cols,
                              torch.tensor([bpos[n] for n in lv], dtype=torch.long), lv))
            if lv is not OUT_IDS:
                for n in lv: col[n] = ncol; ncol += 1
        self.col = col

    def forward(self, X, keep=False):
        A = X; pre = []
        for ncol, m, ci, rows, cols, bi, lv in self.plan:
            W = torch.zeros(ncol, m).index_put((rows, cols), self.w[ci])
            z = A @ W + self.b[bi]
            if keep: z.retain_grad(); pre.append(z)
            if lv is OUT_IDS: return (z, A, pre) if keep else z
            A = torch.cat([A, F.relu(z)], 1)

    def train(self, data, steps, lr=LR, mode='kd', decay=False, bs=256):
        if len(self.keys) == 0: return
        opt = torch.optim.Adam([self.w, self.b], lr=lr); n = len(data['Xh'] if mode == 'hard' else data['X'])
        for s in range(steps):
            if decay:
                for pg in opt.param_groups: pg['lr'] = lr * 0.5 * (1 + math.cos(math.pi * s / steps))
            opt.zero_grad(); self.loss(data, mode, torch.randint(n, (bs,))).backward(); opt.step()
        self.write_back()

    def loss(self, data, mode, i=slice(None)):
        if mode == 'hard': return F.cross_entropy(self.forward(data['Xh'][i]), data['yh'][i])
        return kd_loss(self.forward(data['X'][i]), data['T'][i], data['y'][i])

    def write_back(self):
        w, b = self.w.detach().tolist(), self.b.detach().tolist()
        self.g.conn = dict(zip(self.keys, w)); self.g.bias = dict(zip(self.nodes, b))


def evaluate(g, val, mode):
    with torch.no_grad():
        z = Compiled(g).forward(val['X'])
        if mode == 'hard': L = float(F.cross_entropy(z, val['y']))
        else: L = float(kd_loss(z, val['T']) + ALPHA * F.cross_entropy(z, val['y']))
        return L + LAMBDA * g.count(), float((z.argmax(1) == val['y']).float().mean())


def test_acc(g):
    with torch.no_grad(): return float((Compiled(g).forward(C.Xte).argmax(1) == C.yte).float().mean())


# ---------------------------------------------------------------- mutations
def link_scores(g, data, mode):
    """|d loss / d w| for every link that could be added (src shallower than dst, not present yet)."""
    cg = Compiled(g); cg.w.requires_grad_(True)
    X = data['Xh'] if mode == 'hard' else data['X'][::5]
    z, A, pre = cg.forward(X, keep=True)
    L = F.cross_entropy(z, data['yh']) if mode == 'hard' else kd_loss(z, data['T'][::5])
    L.backward()
    cands = []
    for (ncol, m, ci, rows, cols, bi, lv), zz in zip(cg.plan, pre):
        G = (A[:, :ncol].T @ zz.grad).abs() / len(X)
        srcs = [n for n, c in cg.col.items() if c < ncol]
        for j, dst in enumerate(lv):
            for s in srcs:
                if (s, dst) not in g.conn and g.d(s) < g.d(dst): cands.append((float(G[cg.col[s], j]), s, dst))
    return cands


def weakest(g, data):
    """Link importance = |w| * std of the source activation (training data)."""
    cg = Compiled(g)
    with torch.no_grad():
        acts = data['X'][::5]
        for ncol, m, ci, rows, cols, bi, lv in cg.plan[:-1]:
            W = torch.zeros(ncol, m).index_put((rows, cols), cg.w[ci]); acts = torch.cat([acts, F.relu(acts @ W + cg.b[bi])], 1)
        sd = acts.std(0)
    return sorted(g.conn, key=lambda k: abs(g.conn[k]) * float(sd[cg.col[k[0]]]) + 1e-12)


def repair(g, data, cap):
    while g.count() > cap:
        order = weakest(g, data)
        over = g.count() - cap
        for k in order[:max(1, over // 2)]: g.conn.pop(k)
        g.cleanup()


def mutate(g, inn, data, rng, mode, cap):
    g = g.copy(); room = cap - g.count()
    p_add = 0.7 if room > 4 else 0.35
    if rng.random() < p_add:
        n_add = 1 + int(rng.integers(0, 3))
        if rng.random() < 0.7:
            c = link_scores(g, data, mode); c.sort(reverse=True)
            top = c[:12]
            for i in rng.choice(len(top), size=min(n_add, len(top)), replace=False): g.conn[(top[i][1], top[i][2])] = 0.0
        else:
            nodes = list(range(NI)) + list(g.depth)
            for _ in range(n_add):
                s = int(rng.choice(nodes)); dsts = [n for n in list(g.depth) + OUT_IDS if g.d(n) > g.d(s)]
                dd = int(rng.choice(dsts))
                if (s, dd) not in g.conn: g.conn[(s, dd)] = 0.0
    if rng.random() < 0.25 and g.conn:   # add neuron: split a link a->b into a->new->b (function unchanged)
        keys = list(g.conn); a, b = keys[int(rng.integers(len(keys)))]
        n = inn.node(a, b, set(g.depth)); w = g.conn.pop((a, b))
        g.depth[n] = (g.d(a) + g.d(b)) / 2; g.bias[n] = 0.0; g.conn[(a, n)] = 1.0; g.conn[(n, b)] = w
    if rng.random() < 0.35 and len(g.conn) > 10:
        for k in weakest(g, data)[:1 + int(rng.integers(0, 3))]: g.conn.pop(k)
    if rng.random() < 0.08 and g.depth:
        h = int(rng.choice(list(g.depth)))
        for k in [k for k in g.conn if h in k]: g.conn.pop(k)
    g.cleanup(); repair(g, data, cap); return g


def crossover(a, fa, b, fb, rng):
    """NEAT crossover: matching links inherit randomly, disjoint/excess from the fitter parent."""
    if fb < fa: a, b = b, a
    c = a.copy()
    for k in c.conn:
        if k in b.conn and rng.random() < 0.5: c.conn[k] = b.conn[k]
    for n in c.bias:
        if n in b.bias and rng.random() < 0.5: c.bias[n] = b.bias[n]
    return c


def distance(a, b):
    ka, kb = set(a.conn), set(b.conn); m = ka & kb; N = max(len(ka), len(kb), 1)
    wd = np.mean([abs(a.conn[k] - b.conn[k]) for k in m]) if m else 0.0
    return len(ka ^ kb) / N + 0.4 * wd


# ---------------------------------------------------------------- NEAT search
def neat(init, data, val, seed, mode='kd', cap=CAP, gens=GENS, pop=POP, log=None, frames=None):
    """frames: optional list; gets (generation, best genome of that generation, its fitness) for replaying the run."""
    rng = np.random.default_rng(seed); torch.manual_seed(seed); inn = Innov()
    for g in init:
        for k in g.conn: inn.c(*k)
    P = []
    for i in range(pop):
        g = mutate(init[i % len(init)], inn, data, rng, mode, cap) if i >= len(init) else init[i].copy()
        repair(g, data, cap); cg = Compiled(g); cg.train(data, CHILD_STEPS * 2, mode=mode); P.append(g)
    fit = [evaluate(g, val, mode) for g in P]
    if frames is not None: i0 = int(np.argmin([f[0] for f in fit])); frames.append((0, P[i0].copy(), fit[i0]))
    thr, species_target = 0.5, 5; best = None; history = []
    for gen in range(gens):
        t0 = time.time()
        reps, species = [], []
        for i, g in enumerate(P):
            for s, r in enumerate(reps):
                if distance(g, r) < thr: species[s].append(i); break
            else: reps.append(g); species.append([i])
        thr *= 1.1 if len(species) > species_target else 0.9 if len(species) < species_target else 1.0
        order = sorted(range(len(P)), key=lambda i: fit[i][0])
        if best is None or fit[order[0]][0] < best[1][0]: best = (P[order[0]].copy(), fit[order[0]])
        # offspring per species ~ shared fitness (mean of exp(-(loss - best)/0.05))
        L0 = fit[order[0]][0]
        share = np.array([np.mean([math.exp(-(fit[i][0] - L0) / 0.05) for i in sp]) for sp in species])
        quota = np.floor(share / share.sum() * (pop - 2)).astype(int)
        while quota.sum() < pop - 2: quota[int(np.argmax(share))] += 1
        nxt = [P[order[0]].copy(), P[order[1]].copy()]       # global elites
        for sp, q in zip(species, quota):
            sp = sorted(sp, key=lambda i: fit[i][0]); par = sp[:max(1, (len(sp) + 1) // 2)]
            for _ in range(q):
                i = par[int(rng.integers(len(par)))]
                if len(par) > 1 and rng.random() < 0.25:
                    j = par[int(rng.integers(len(par)))]; c = crossover(P[i], fit[i][0], P[j], fit[j][0], rng)
                else: c = P[i]
                nxt.append(mutate(c, inn, data, rng, mode, cap))
        for g in nxt[2:]: Compiled(g).train(data, CHILD_STEPS, mode=mode)
        P = nxt; fit = [evaluate(g, val, mode) for g in P]
        bi = int(np.argmin([f[0] for f in fit])); b = P[bi]
        if frames is not None: frames.append((gen + 1, b.copy(), fit[bi]))
        rec = dict(gen=gen + 1, params=b.count(), hidden=len(b.depth), val_acc=round(fit[bi][1], 4),
                   fitness=round(fit[bi][0], 4), species=len(species), sec=round(time.time() - t0, 1))
        history.append(rec)
        if log: log(f'    gen {gen + 1:3d}  best {rec["params"]:3d} params {rec["hidden"]:2d} hidden  val {rec["val_acc"]:.4f}  fit {rec["fitness"]:.4f}  species {len(species)}  {rec["sec"]}s')
    i = int(np.argmin([f[0] for f in fit]))
    if fit[i][0] < best[1][0]: best = (P[i].copy(), fit[i])
    return best[0], history


# ---------------------------------------------------------------- converting the existing networks
def from_enet(net):
    """NEAT-merge's layered network (64 -> h0 -> h1 -> 10, masked) as a genome."""
    g = Genome(); ids = {}
    for l in (0, 1):
        for j, nid in enumerate(net.ids[l]):
            n = NI + NO + 1000 * l + int(nid); ids[(l, j)] = n; g.depth[n] = (l + 1) / 3; g.bias[n] = float(net.b[l][j])
    for o in range(NO): g.bias[NI + o] = float(net.b[2][o])
    src = [lambda i: i, lambda i: ids[(0, i)], lambda i: ids[(1, i)]]
    dst = [lambda j: ids[(0, j)], lambda j: ids[(1, j)], lambda j: NI + j]
    for k in range(3):
        for i, j in np.argwhere(net.M[k] > 0): g.conn[(src[k](int(i)), dst[k](int(j)))] = float(net.W[k][i, j])
    g.cleanup(); return g


def nm160(base, seed):
    """Re-run neat_merge's winner (full3_look) to get its smallest network (same code, same seed)."""
    net = NM.ENet(base)
    for r, target in enumerate(NM.TARGETS):
        net = NM.search(net, target, 'full3_look', []); NM.finetune(net, NM.FT_EPOCHS, seed * 100 + r)
    return net


def to_genome(d):
    g = Genome(); g.depth = {int(k): v for k, v in d['depth'].items()}; g.bias = {int(k): v for k, v in d['bias'].items()}
    g.conn = {(int(a), int(b)): w for a, b, w in d['conn']}; return g


def cached_nm160(base, seed):
    path = os.path.join(HERE, f'nm160_seed{seed}.json')
    if os.path.exists(path): return to_genome(json.load(open(path)))
    enet = nm160(base, seed); g = from_enet(enet)
    json.dump(dict(acc=NM.acc_of(enet), params=enet.count(), **describe(g)['genome']), open(path, 'w')); return g


def minimal(rng, k=3):
    g = Genome()
    for o in OUT_IDS:
        for i in rng.choice(NI, size=k, replace=False): g.conn[(int(i), o)] = 0.0
    return g


def fixed_sparse(H, P, rng):
    """64-H-10 with random links, exactly P parameters (every hidden neuron has >= 1 in and out link)."""
    g = Genome(); hs = [NI + NO + i for i in range(H)]
    for h in hs: g.depth[h] = 0.5; g.bias[h] = 0.1
    nl = P - H - NO; n_in = max(H, int(round(nl * 0.6))); n_out = nl - n_in
    pool_in = [(i, h) for i in range(NI) for h in hs]; pool_out = [(h, o) for h in hs for o in OUT_IDS]
    for h in hs: g.conn[(int(rng.integers(NI)), h)] = 0.0; g.conn[(h, int(rng.choice(OUT_IDS)))] = 0.0
    while sum(1 for k in g.conn if k[1] in hs) < n_in: g.conn.setdefault(pool_in[int(rng.integers(len(pool_in)))], 0.0)
    while len(g.conn) < nl: g.conn.setdefault(pool_out[int(rng.integers(len(pool_out)))], 0.0)
    for k in g.conn: g.conn[k] = float(rng.normal() * 0.3)
    return g


def dense(H, rng):
    g = Genome(); hs = [NI + NO + i for i in range(H)]
    for h in hs: g.depth[h] = 0.5; g.bias[h] = 0.1
    for i in range(NI):
        for h in hs: g.conn[(i, h)] = float(rng.normal() * math.sqrt(2 / NI))
    for h in hs:
        for o in OUT_IDS: g.conn[(h, o)] = float(rng.normal() * math.sqrt(2 / H))
    return g


def polish(g, data, mode, seed, lr=LR):
    """Final training of a fixed topology (same schedule for every student), on all training images.
    Students that arrive with trained weights (NEAT winners, nm160) use a gentler lr so they are not knocked out."""
    torch.manual_seed(seed); g = g.copy(); Compiled(g).train(data, FINAL_STEPS, lr=lr, mode=mode, decay=True); return g


def describe(g):
    used = sorted({k[0] for k in g.conn if k[0] < NI})
    return dict(params=g.count(), hidden=len(g.depth), links=len(g.conn), pixels_used=len(used),
                depths=sorted({round(v, 3) for v in g.depth.values()}),
                genome=dict(depth={str(k): v for k, v in g.depth.items()}, bias={str(k): v for k, v in g.bias.items()},
                            conn=[[a, b, w] for (a, b), w in g.conn.items()]))


def run_seed(seed, out):
    t0 = time.time(); log = lambda s: print(f'[seed {seed}] {s}', flush=True)
    base = NM.base_net(seed); log(f'teacher {base.count()} params, test {C.acc(base):.4f}')
    search, val, full = make_data(base, seed)
    R = out.setdefault(str(seed), {}); R['teacher'] = dict(params=base.count(), acc=round(C.acc(base), 4))
    g160 = cached_nm160(base, seed)
    R['nm160'] = dict(params=g160.count(), acc=round(test_acc(g160), 4)); log(f'nm160 {R["nm160"]}  ({time.time() - t0:.0f}s)')
    want = lambda n: ONLY is None or n in ONLY
    for mode in ('hard', 'kd') if want('nm160_long') else ():   # the same network, just trained longer (labels / distillation): the honest bar
        a = polish(g160, full, mode, seed, lr=WARM_LR); R[f'nm160_{mode}'] = dict(params=a.count(), acc=round(test_acc(a), 4)); log(f'nm160_{mode} {R[f"nm160_{mode}"]}')
    rng = np.random.default_rng(seed)
    for mode in ('hard', 'kd') if want('dense2') else ():
        a = polish(dense(2, rng), full, mode, seed); R[f'dense2_{mode}'] = dict(params=a.count(), acc=round(test_acc(a), 4)); log(f'dense2_{mode} {R[f"dense2_{mode}"]}')

    def neat_run(name, init, mode):
        t = time.time(); log(f'{name}: NEAT, cap {CAP}, {GENS} generations x {POP}')
        g, hist = neat(init, search, val, seed, mode=mode, log=log)
        a = polish(g, full, mode, seed, lr=WARM_LR)
        R[name] = dict(acc=round(test_acc(a), 4), **describe(a), history=hist, sec=round(time.time() - t))
        log(f'{name}: params {a.count()} hidden {len(a.depth)} test {R[name]["acc"]:.4f}  ({R[name]["sec"]}s)')
        return a

    rng = np.random.default_rng(seed)
    if want('neat_kd'): best = neat_run('neat_kd', [minimal(rng) for _ in range(POP)], 'kd')
    seeded = g160.copy(); repair(seeded, search, CAP); Compiled(seeded).train(search, 200, lr=WARM_LR)
    if want('nm160_cut_kd'):
        a = polish(seeded, full, 'kd', seed, lr=WARM_LR); R['nm160_cut_kd'] = dict(params=a.count(), acc=round(test_acc(a), 4)); log(f'nm160_cut_kd {R["nm160_cut_kd"]}')
    if want('neat_kd_seed'): neat_run('neat_kd_seed', [seeded], 'kd')
    if want('neat_hard_seed'): neat_run('neat_hard_seed', [seeded], 'hard')
    if want('neat_hard'): neat_run('neat_hard', [minimal(rng) for _ in range(POP)], 'hard')
    for mode in ('hard', 'kd') if want('neat_kd') else ():
        H = max(1, len(best.depth)); Pm = best.count()
        a = polish(fixed_sparse(H, Pm, rng), full, mode, seed)
        R[f'sparse_{mode}'] = dict(params=a.count(), hidden=H, acc=round(test_acc(a), 4)); log(f'sparse_{mode} {R[f"sparse_{mode}"]}')
    log(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    seeds = [int(a) for a in sys.argv[1:]] or [0, 1, 2]
    path = os.path.join(HERE, os.environ.get('DS_OUT', 'distill.json'))
    for seed in seeds:
        out = json.load(open(path)) if os.path.exists(path) else {}
        out.setdefault('config', dict(cap=CAP, gens=GENS, pop=POP, lam=LAMBDA, temp=TEMP, alpha=ALPHA))
        run_seed(seed, out.setdefault('runs', {}))
        json.dump(out, open(path, 'w'), separators=(',', ':'))
    print('wrote', path)
