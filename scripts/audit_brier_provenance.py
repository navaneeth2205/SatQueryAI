import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import inspect
import scripts.benchmark_phase1_candidates as bpc

print('--- compute_multiclass_brier implementation ---')
print(inspect.getsource(bpc.compute_multiclass_brier))

with open(r'c:\Users\B NAVANEETH\SatQueryAI\scripts\evaluate_phase1_locked_validation.py', 'r') as f:
    text = f.read()

print('--- evaluate_phase1_locked_validation.py Brier usage ---')
for line in text.split('\n'):
    if 'brier' in line.lower():
        print(line)
