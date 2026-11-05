"""Mean / min / max test accuracy and parameter count per student over seeds, from run_seed*.json (and cap*_seed*.json).
  python3 summarize.py          # prints a table, writes summary.json"""
import json, glob, os
HERE = os.path.dirname(os.path.abspath(__file__))
rows = {}
for p in sorted(glob.glob(os.path.join(HERE, '*seed*.json'))):
    if os.path.basename(p).startswith('nm160_'): continue
    d = json.load(open(p)); cap = d['config']['cap']
    for seed, R in d['runs'].items():
        for name, r in R.items():
            key = name if name in ('teacher', 'nm160', 'nm160_hard', 'nm160_kd', 'dense2_hard', 'dense2_kd') else f'{name}@cap{cap}'
            rows.setdefault(key, {})[seed] = (r['params'], r['acc'])
out = {}
print(f'{"student":28s} {"params":>7s} {"acc %":>6s} {"min":>6s} {"max":>6s}  n')
for k, v in rows.items():
    P = [p for p, _ in v.values()]; A = [a * 100 for _, a in v.values()]
    out[k] = dict(n=len(v), params=round(sum(P) / len(P), 1), acc=round(sum(A) / len(A), 2), min=round(min(A), 2), max=round(max(A), 2),
                  per_seed={s: dict(params=p, acc=a) for s, (p, a) in v.items()})
    print(f'{k:28s} {out[k]["params"]:7.1f} {out[k]["acc"]:6.2f} {out[k]["min"]:6.2f} {out[k]["max"]:6.2f}  {len(v)}')
json.dump(out, open(os.path.join(HERE, 'summary.json'), 'w'), indent=1)
