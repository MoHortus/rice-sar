"""Download Sentinel-1 EOPatches for one season of a study region (labels must be downloaded first).

Usage (from the project root):
    python scripts/download_s1.py --year 2025 --area feasibility
    python scripts/download_s1.py --year 2024 --area full

Needs CDSE credentials in .env (see .env.example). Prints the request/PU estimate and asks before downloading.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from eolearn.io import get_available_timestamps  # noqa: E402

from ricesar.config import DATA_DIR, get_sh_config, load_config  # noqa: E402
from ricesar.download import (  # noqa: E402
    estimate_cost,
    feasibility_bbox,
    patches_with_parcels,
    run_download,
    s1_collection,
    season_interval,
    track_filter,
)
from ricesar.labels import load_labels  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/camargue.toml")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--area", choices=["feasibility", "full"], default="feasibility")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = parser.parse_args()

    config = load_config(args.config)
    sh_config = get_sh_config()
    labels = load_labels(config, args.year)
    bboxes = [feasibility_bbox(config)] if args.area == "feasibility" else patches_with_parcels(config, labels)

    timestamps = get_available_timestamps(
        bbox=bboxes[0], time_interval=season_interval(config, args.year), data_collection=s1_collection(config),
        config=sh_config,
    )
    dates = track_filter(config["s1"].get("track"))(timestamps, pd.Timedelta(hours=1).to_pytimedelta())
    cost = estimate_cost(bboxes[0], len(dates), config["s1"]["resolution"])
    print(f"{len(bboxes)} patch(es), track {config['s1'].get('track', 'all')}, about {len(dates)} passes each")
    print(f"Estimate: {cost['requests'] * len(bboxes)} requests, {cost['pu_total'] * len(bboxes)} PU")
    if not args.yes and input("Download? [y/N] ").strip().lower() != "y":
        return

    out_dir = DATA_DIR / "raw" / f"eopatches_{config['name']}_{args.area}_{args.year}"
    executor = run_download(config, sh_config, labels, bboxes, args.year, out_dir)
    if executor is None:
        print("All patches already exist in", out_dir)
        return
    executor.make_report()
    print(f"Done: {len(executor.get_successful_executions())} ok, {len(executor.get_failed_executions())} failed")
    print("Report:", executor.get_report_path())


if __name__ == "__main__":
    main()
