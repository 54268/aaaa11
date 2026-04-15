from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ORACLE_LINKS = {
    "dataset1": "http://hdl.handle.net/2047/D20324547",
    "dataset2": "http://hdl.handle.net/2047/D20324548",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset1", choices=sorted(ORACLE_LINKS))
    parser.add_argument("--output-dir", type=str, default="data/raw/oracle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.dataset}.download"
    cmd = ["curl.exe", "-L", "-A", "Mozilla/5.0", ORACLE_LINKS[args.dataset], "-o", str(output_path)]
    subprocess.run(cmd, check=True)
    print(f"Downloaded {args.dataset} to {output_path}")


if __name__ == "__main__":
    main()
