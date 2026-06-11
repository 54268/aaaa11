from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[3]
subprocess.run([sys.executable, str(root / 'ablations' / 'run_ablation.py'), '--category', 'km', '--dataset', 'oracle'], check=True, cwd=root)
