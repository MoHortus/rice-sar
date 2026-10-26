"""Download parcel labels (crop declarations) for one year of a study region.

Usage (from the project root):
    python scripts/download_labels.py --year 2025
    python scripts/download_labels.py --config configs/camargue.toml --year 2024
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ricesar.config import load_config  # noqa: E402
from ricesar.labels import download_labels, load_labels, summary  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/camargue.toml")
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    path = download_labels(config, args.year)
    labels = load_labels(config, args.year)
    print(f"Saved {len(labels)} parcels ({labels['is_rice'].sum()} rice) to {path}")
    print(summary(labels).head(10).to_string())


if __name__ == "__main__":
    main()
