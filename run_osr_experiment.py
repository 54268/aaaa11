from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sei_osr.utils.config import load_config
from sei_osr.utils.io import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--skip-prepare", action="store_true")
    return parser.parse_args()


def _run(script: Path, config_path: str) -> None:
    cmd = [sys.executable, str(script), "--config", config_path]
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    args = parse_args()
    config_path = str(Path(args.config).resolve())
    config = load_config(config_path)

    prep_kind = config.get("prep", {}).get("kind", "")
    if not args.skip_prepare and prep_kind == "wisig_compact":
        _run(ROOT / "tools" / "prepare_wisig_compact.py", config_path)
    elif not args.skip_prepare and prep_kind == "oracle_sigmf":
        _run(ROOT / "tools" / "prepare_oracle_sigmf.py", config_path)

    _run(ROOT / "trainers" / "train_closed.py", config_path)
    _run(ROOT / "tools" / "mine_boundary.py", config_path)
    _run(ROOT / "tools" / "generate_pseudo_unknown.py", config_path)
    _run(ROOT / "tools" / "fit_openmax.py", config_path)
    _run(ROOT / "tools" / "calibrate_fusion.py", config_path)
    _run(ROOT / "eval" / "evaluate_open_set.py", config_path)

    metrics = load_json(Path(config["project"]["output_dir"]) / "open_set_metrics.json")
    print("\n[Final Metrics]")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    print(f"\nSummary report: {Path(config['project']['output_dir']) / 'final_report.md'}")
    print(f"Latest summary shortcut: {ROOT / 'RESULT_SUMMARY.md'}")


if __name__ == "__main__":
    main()
