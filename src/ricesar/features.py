"""From EOPatches to per-parcel time series, a fixed time grid and per-parcel features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from eolearn.core import EOPatch
from sentinelhub import BBox
from shapely.ops import unary_union

from .download import BANDS, MASK_FEATURE, PARCEL_FEATURE, S1_FEATURE, track_slot
from .labels import project

SERIES_COLUMNS = ["VV_db", "VH_db", "VH_VV_db"]


def parcel_sums(eopatch: EOPatch) -> pd.DataFrame:
    """Per parcel and date: sum of each band (linear power) and the number of valid pixels.

    Sums (not means) are returned so that parcels split across several patches can be combined exactly.
    """
    pids = eopatch[PARCEL_FEATURE][..., 0]
    data = eopatch[S1_FEATURE]
    valid_data = eopatch[MASK_FEATURE][..., 0].astype(bool)
    n_ids = int(pids.max()) + 1

    rows = []
    for t_idx, timestamp in enumerate(eopatch.timestamps):
        # dataMask = 0 outside the swath; value 0 also marks missing data (log of 0 is undefined)
        valid = valid_data[t_idx] & (pids > 0) & np.all(data[t_idx] > 0, axis=-1)
        ids = pids[valid]
        counts = np.bincount(ids, minlength=n_ids)
        present = np.nonzero(counts)[0]
        frame = {"pid": present, "timestamp": timestamp, "n_pixels": counts[present]}
        for b_idx, band in enumerate(BANDS):
            frame[f"{band}_sum"] = np.bincount(ids, weights=data[t_idx][..., b_idx][valid], minlength=n_ids)[present]
        rows.append(pd.DataFrame(frame))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def parcel_timeseries(eopatch_dirs: list[str | Path], min_pixels: int = 5, track: str | None = None) -> pd.DataFrame:
    """Per-parcel mean backscatter over all given EOPatches, in linear power and dB.

    Averaging is done in linear units first; averaging dB values would give a biased (geometric) mean.
    Parcels with fewer than `min_pixels` valid pixels on a date are dropped for that date.
    If `track` is given (e.g. "05:50"), only passes of that track are used.
    """
    sums = pd.concat([parcel_sums(EOPatch.load(str(path))) for path in eopatch_dirs], ignore_index=True)
    sums["timestamp"] = pd.to_datetime(sums["timestamp"])
    if track is not None:
        sums = sums[sums["timestamp"].map(track_slot) == track]

    # One pass gets slightly different timestamps in neighbouring patches (seconds apart); group by the hour so that
    # the pieces of a parcel split across patches are combined
    sums["pass"] = sums["timestamp"].dt.floor("h")
    sum_cols = ["n_pixels"] + [f"{band}_sum" for band in BANDS]
    aggregations = {"timestamp": ("timestamp", "min"), **{col: (col, "sum") for col in sum_cols}}
    series = sums.groupby(["pid", "pass"]).agg(**aggregations).reset_index()

    series = series[series["n_pixels"] >= min_pixels].copy()
    for band in BANDS:
        series[band] = series.pop(f"{band}_sum") / series["n_pixels"]
        series[f"{band}_db"] = to_db(series[band])
    series["VH_VV_db"] = series["VH_db"] - series["VV_db"]  # ratio in linear = difference in dB
    return series


def to_db(linear: np.ndarray | pd.Series) -> np.ndarray | pd.Series:
    return 10 * np.log10(linear)


def grid_dates(config: dict[str, Any], year: int, step_days: int = 12) -> pd.DatetimeIndex:
    """Start dates of the fixed time bins covering the season."""
    season = config["season"]
    return pd.date_range(f"{year}-{season['start']}", f"{year}-{season['end']}", freq=f"{step_days}D")


def to_time_grid(series: pd.DataFrame, config: dict[str, Any], year: int, step_days: int = 12) -> pd.DataFrame:
    """One row per parcel, one column per (measure, time bin), e.g. `VH_db_05`.

    Passes are averaged per bin (in linear units), so years with different revisit (one pass vs two per bin) give
    the same columns. Empty bins are filled by linear interpolation along time.
    """
    dates = grid_dates(config, year, step_days)
    frame = series.copy()
    frame["bin"] = ((pd.to_datetime(frame["timestamp"]) - dates[0]).dt.days // step_days).astype(int)
    frame = frame[(frame["bin"] >= 0) & (frame["bin"] < len(dates))]

    means = frame.groupby(["pid", "bin"])[BANDS].mean()
    measures = pd.DataFrame(
        {"VV_db": to_db(means["VV"]), "VH_db": to_db(means["VH"]), "VH_VV_db": to_db(means["VH"] / means["VV"])}
    )

    columns = []
    for measure in SERIES_COLUMNS:
        wide = measures[measure].unstack("bin").reindex(columns=range(len(dates)))
        wide = wide.interpolate(axis=1, limit_direction="both")
        wide.columns = [f"{measure}_{b:02d}" for b in wide.columns]
        columns.append(wide)
    return pd.concat(columns, axis=1)


def _window_bins(config: dict[str, Any], year: int, window: str, step_days: int) -> list[int]:
    start, end = config["features"][window]
    dates = grid_dates(config, year, step_days)
    inside = (dates >= pd.Timestamp(f"{year}-{start}")) & (dates <= pd.Timestamp(f"{year}-{end}"))
    return list(np.nonzero(inside)[0])


def season_features(grid: pd.DataFrame, config: dict[str, Any], year: int, step_days: int = 12) -> pd.DataFrame:
    """Interpretable per-parcel features from the physics of flooded rice, using the season windows of the config."""

    def cols(measure: str, window: str) -> list[str]:
        return [f"{measure}_{b:02d}" for b in _window_bins(config, year, window, step_days)]

    features = pd.DataFrame(index=grid.index)
    features["vh_flood_min"] = grid[cols("VH_db", "flood")].min(axis=1)  # how dark during flooding
    features["vh_peak_max"] = grid[cols("VH_db", "peak")].max(axis=1)  # how bright at full growth
    features["vh_rise"] = features["vh_peak_max"] - features["vh_flood_min"]  # dip followed by growth
    features["vv_flood_min"] = grid[cols("VV_db", "flood")].min(axis=1)
    features["ratio_flood_min"] = grid[cols("VH_VV_db", "flood")].min(axis=1)
    features["ratio_peak_mean"] = grid[cols("VH_VV_db", "peak")].mean(axis=1)
    return features


def spatial_blocks(labels: gpd.GeoDataFrame, crs: Any, size_m: int = 5000) -> pd.Series:
    """Block id per parcel (index `pid`): the `size_m` square of the parcel centroid, e.g. "126_965"."""
    centroids = project(labels, crs).geometry.centroid
    block_x = (centroids.x // size_m).astype(int).astype(str)
    block_y = (centroids.y // size_m).astype(int).astype(str)
    return pd.Series((block_x + "_" + block_y).values, index=labels["pid"].values, name="block")


def build_dataset(
    config: dict[str, Any],
    year: int,
    eopatch_dirs: list[str | Path],
    labels: gpd.GeoDataFrame,
    bboxes: list[BBox],
    min_coverage: float = 0.75,
    step_days: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-parcel table: id, label, crop, area, spatial block, time grid and season features.

    Parcels observed (≥ 5 valid pixels) on fewer than `min_coverage` of the passes are left out, mostly small or narrow
    parcels after the edge buffer. Returns (dataset, excluded labels).
    """
    series = parcel_timeseries(eopatch_dirs, track=config["s1"].get("track"))
    n_passes = series["pass"].nunique()
    coverage = series.groupby("pid").size() / n_passes
    kept = coverage[coverage >= min_coverage].index

    grid = to_time_grid(series[series["pid"].isin(kept)], config, year, step_days)
    features = season_features(grid, config, year, step_days)
    meta = labels.set_index("pid")[["is_rice", "code_cultu", "group_name", "surf_parc"]]
    crs = bboxes[0].crs.pyproj_crs()
    blocks = spatial_blocks(labels, crs, config["features"]["block_size_m"])

    dataset = meta.loc[grid.index].join(blocks).join(features).join(grid)
    dataset.index.name = "pid"
    area = unary_union([bbox.geometry for bbox in bboxes])
    inside = project(labels, crs).within(area).values  # parcels fully inside the downloaded squares
    excluded = labels[inside & ~labels["pid"].isin(kept).values]
    return dataset, excluded
