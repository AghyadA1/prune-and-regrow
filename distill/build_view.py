"""Build best-student.html from view_template.html + view_data.json (+ the teacher's accuracy)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, 'view_data.json')))
r = json.load(open(os.path.join(HERE, f'run_seed{d["seed"]}.json')))['runs'][str(d['seed'])]
d['teacher_acc'] = r['teacher']['acc']
html = open(os.path.join(HERE, 'view_template.html')).read().replace('__DATA__', json.dumps(d, separators=(',', ':')))
open(os.path.join(HERE, 'best-student.html'), 'w').write(html); print(len(html))
