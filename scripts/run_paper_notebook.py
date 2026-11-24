"""Execute the supplied reproducibility notebook in a fresh process."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--repo', required=True, type=Path)
p.add_argument('--data', required=True, type=Path)
p.add_argument('--output', required=True, type=Path)
a = p.parse_args()
repo, data, output = a.repo.resolve(), a.data.resolve(), a.output.resolve()
output.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(output / '.mplconfig')
os.environ['OUTDATED_IGNORE'] = '1'
sys.path.insert(0, str(repo))
os.chdir(repo)
nb = json.loads((repo / 'paper_reproducibility_analysis.ipynb').read_text())
ctx = {'display': lambda *args: None}
started = time.time()
executed = []
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    code = ''.join(c['source'])
    code = code.replace('u.ensure_output_dirs("paper_repro_outputs")', f'u.ensure_output_dirs({str(output)!r})')
    code = code.replace('u.load_paper_data("data")', f'u.load_paper_data({str(data)!r})')
    exec(compile(code, f'notebook cell {i}', 'exec'), ctx)
    executed.append(i)
    print(f'Completed cell {i} after {time.time()-started:.1f}s', flush=True)
import numpy, pandas, scipy, statsmodels, pingouin
manifest = {
    'python': sys.version,
    'versions': {m.__name__: m.__version__ for m in [numpy, pandas, scipy, statsmodels, pingouin]},
    'data_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(data.glob('*.csv'))},
    'scoring_sha256': hashlib.sha256((repo/'analysisfunctions.py').read_bytes()).hexdigest(),
    'code_cells_executed': executed,
    'elapsed_seconds': time.time()-started,
}
(output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
