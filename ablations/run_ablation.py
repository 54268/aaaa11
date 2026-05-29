from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = PROJECT_ROOT / "outputs" / "wisig_singleday_osr_k16_u12" / "best_closed_set.pt"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sei_osr.utils.config import load_config
from sei_osr.utils.io import ensure_dir, load_json, save_json
from sei_osr.utils.python_env import resolve_python_executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    return parser.parse_args()


def _run(script: Path, config_path: str, ckpt_path: str) -> None:
    python_exe = resolve_python_executable()
    cmd = [str(python_exe), str(script), "--config", config_path, "--ckpt", ckpt_path]
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = ensure_dir(config["project"]["output_dir"])
    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    copied_ckpt = output_dir / "best_closed_set.pt"
    shutil.copy2(ckpt_path, copied_ckpt)

    source_train_summary = ckpt_path.parent / "train_summary.json"
    if source_train_summary.exists():
        shutil.copy2(source_train_summary, output_dir / "train_summary_from_base.json")

    save_json(
        output_dir / "ablation_manifest.json",
        {
            "config_path": str(Path(args.config).resolve()),
            "base_checkpoint": str(ckpt_path),
            "base_output_dir": str(ckpt_path.parent),
        },
    )

    _run(PROJECT_ROOT / "tools" / "mine_boundary.py", args.config, str(ckpt_path))
    python_exe = resolve_python_executable()
    cmd = [str(python_exe), str(PROJECT_ROOT / "tools" / "generate_pseudo_unknown.py"), "--config", args.config]
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    _run(PROJECT_ROOT / "tools" / "fit_openmax.py", args.config, str(ckpt_path))
    _run(PROJECT_ROOT / "tools" / "calibrate_fusion.py", args.config, str(ckpt_path))
    _run(PROJECT_ROOT / "eval" / "evaluate_open_set.py", args.config, str(ckpt_path))

    metrics = load_json(output_dir / "open_set_metrics.json")
    print("\n[Ablation Metrics]")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()

