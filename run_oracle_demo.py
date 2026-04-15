from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TF_GPU_PYTHON = Path(r"E:\anaconda3\envs\tf_gpu\python.exe")


def main() -> None:
    python_exe = TF_GPU_PYTHON if TF_GPU_PYTHON.exists() else Path(sys.executable)
    cmd = [
        str(python_exe),
        str(ROOT / "run_osr_experiment.py"),
        "--config",
        str(ROOT / "configs" / "oracle_osr_main.yaml"),
        *sys.argv[1:],
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
