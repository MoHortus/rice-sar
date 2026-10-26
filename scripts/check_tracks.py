"""Count Sentinel-1 passes per track for every study-area square with parcels, and recommend one track.

Only the Catalog is queried (dates, no images), so this costs no processing units.
Use a training year: choosing the track must not depend on the test year.

Usage (from the project root):
    python scripts/check_tracks.py --year 2024
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from eolearn.io import get_available_timestamps  # noqa: E402
from sentinelhub.time_utils import filter_times  # noqa: E402

from ricesar.config import get_sh_config, load_config  # noqa: E402
from ricesar.download import patch_name, patches_with_parcels, s1_collection, season_interval, track_slot  # noqa: E402
from ricesar.labels import load_labels  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/camargue.toml")
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    sh_config = get_sh_config()
    bboxes = patches_with_parcels(config, load_labels(config, args.year))

    rows = []
    for bbox in bboxes:
        timestamps = get_available_timestamps(
            bbox=bbox,
            time_interval=season_interval(config, args.year),
            data_collection=s1_collection(config),
            config=sh_config,
        )
        passes = filter_times(timestamps, pd.Timedelta(hours=1).to_pytimedelta())
        rows.append({"patch": patch_name(bbox), **Counter(track_slot(t) for t in passes)})

    table = pd.DataFrame(rows).set_index("patch").fillna(0).astype(int)
    print(f"{config['s1']['orbit_direction'].lower()} passes per track, {args.year}, {len(table)} squares with parcels:")
    print(table.to_string())

    summary = pd.DataFrame({"squares_covered": (table > 0).sum(), "min_passes": table.min(), "total": table.sum()})
    print("\n" + summary.to_string())
    full = summary[summary["squares_covered"] == len(table)]
    if full.empty:
        print("\nNo single track covers every square.")
    else:
        print(f"\nRecommended track (covers all squares, most passes): {full['min_passes'].idxmax()}")


if __name__ == "__main__":
    main()
