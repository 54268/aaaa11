from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


WISIG_LINKS = {
    "ManySig": "https://drive.google.com/file/d/1szuns8MhcYocdbipK9t9TM9MLgEMklxk/view?usp=sharing",
    "ManyRx": "https://drive.google.com/file/d/1TtdydJCuhkvDQ1RWb3PkxakkWo2-X5Uv/view?usp=sharing",
    "ManyTx": "https://drive.google.com/file/d/17EnvGFoflJEh1xhFC8wx5fhCuPYhWt2l/view?usp=sharing",
    "SingleDay": "https://drive.google.com/file/d/1lWf9BuUZTSNcABVFWYoBT_-EH8ctXEcZ/view?usp=sharing",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default="ManySig", choices=sorted(WISIG_LINKS))
    parser.add_argument("--output-dir", type=str, default="data/raw/wisig")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.subset}.pkl"
    cmd = ["gdown", "--fuzzy", WISIG_LINKS[args.subset], "-O", str(output_path)]
    subprocess.run(cmd, check=True)
    print(f"Downloaded {args.subset} to {output_path}")


if __name__ == "__main__":
    main()
