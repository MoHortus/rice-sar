"""Sentinel-1 download with eo-learn: one EOPatch per square patch, with S1 backscatter and rasterised parcel ids."""

from __future__ import annotations

import datetime as dt
from functools import partial
from pathlib import Path
from typing import Any, Callable

import geopandas as gpd
import numpy as np
import pandas as pd
from eolearn.core import EOExecutor, EONode, EOWorkflow, FeatureType, OverwritePermission, SaveTask
from eolearn.core.utils.fs import get_filesystem
from eolearn.geometry import VectorToRasterTask
from eolearn.io import SentinelHubInputTask
from sentinelhub import CRS, BBox, DataCollection, SHConfig, UtmZoneSplitter
from sentinelhub.time_utils import filter_times
from shapely.geometry import Point, box

from .config import CDSE_BASE_URL
from .labels import project

S1_FEATURE = (FeatureType.DATA, "S1")  # bands: VV, VH (linear power)
MASK_FEATURE = (FeatureType.MASK, "dataMask")
PARCEL_FEATURE = (FeatureType.MASK_TIMELESS, "PARCEL_ID")  # 0 = no parcel
BANDS = ["VV", "VH"]


def s1_collection(config: dict[str, Any]) -> DataCollection:
    """Sentinel-1 IW (VV+VH) of one orbit direction, defined for the CDSE endpoint."""
    direction = config["s1"]["orbit_direction"].upper()
    base = {"ASCENDING": DataCollection.SENTINEL1_IW_ASC, "DESCENDING": DataCollection.SENTINEL1_IW_DES}[direction]
    return base.define_from(f"s1iw_{direction.lower()}_cdse", service_url=CDSE_BASE_URL)


def track_slot(timestamp: dt.datetime) -> str:
    """Satellite track label: the 5-minute UTC slot of the pass, e.g. "05:40". Passes of one track over a region
    always come at about the same time of day, and neighbouring tracks are several minutes apart."""
    return pd.Timestamp(timestamp).floor("5min").strftime("%H:%M")


def _keep_track(timestamps: list[dt.datetime], time_difference: dt.timedelta, slot: str) -> list[dt.datetime]:
    return filter_times([t for t in timestamps if track_slot(t) == slot], time_difference)


def track_filter(slot: str | None) -> Callable[[list[dt.datetime], dt.timedelta], list[dt.datetime]]:
    """Timestamp filter for `SentinelHubInputTask` that keeps only one track (all tracks if `slot` is None)."""
    return filter_times if slot is None else partial(_keep_track, slot=slot)


def season_interval(config: dict[str, Any], year: int) -> tuple[str, str]:
    season = config["season"]
    return f"{year}-{season['start']}", f"{year}-{season['end']}"


def feasibility_bbox(config: dict[str, Any]) -> BBox:
    """The UTM-aligned square (size from the config) that contains the feasibility centre point."""
    lon, lat = config["feasibility"]["center"]
    splitter = UtmZoneSplitter([Point(lon, lat).buffer(1e-6)], CRS.WGS84, config["feasibility"]["size_m"])
    return splitter.get_bbox_list()[0]


def study_area_bboxes(config: dict[str, Any], size_m: int = 10000) -> list[BBox]:
    """UTM-aligned squares covering the whole study area."""
    splitter = UtmZoneSplitter([box(*config["aoi"]["bbox"])], CRS.WGS84, size_m)
    return splitter.get_bbox_list()


def patches_with_parcels(config: dict[str, Any], labels: gpd.GeoDataFrame, size_m: int = 10000) -> list[BBox]:
    """The study-area squares that contain at least one labelled parcel (no need to download empty land or sea)."""
    bboxes = study_area_bboxes(config, size_m)
    parcels = project(labels, bboxes[0].crs.pyproj_crs())
    return [bbox for bbox in bboxes if parcels.intersects(bbox.geometry).any()]


def patch_name(bbox: BBox) -> str:
    """Stable folder name from the lower-left UTM corner, e.g. `utm31n_640000_4820000`."""
    return f"{bbox.crs.name.lower().replace('_', '')}_{int(bbox.min_x)}_{int(bbox.min_y)}"


def estimate_cost(bbox: BBox, n_dates: int, resolution: float, orthorectify: bool = True) -> dict[str, float]:
    """Rough Process API cost: one request per date. PU = area/512² × bands/3 × 2 (FLOAT32) × 2 (orthorectification)."""
    width = (bbox.max_x - bbox.min_x) / resolution
    height = (bbox.max_y - bbox.min_y) / resolution
    pu_per_request = max(width * height / 512**2, 0.01) * (len(BANDS) / 3) * 2 * (2 if orthorectify else 1)
    return {"requests": n_dates, "pu_per_request": round(pu_per_request, 2), "pu_total": round(pu_per_request * n_dates)}


def build_workflow(
    config: dict[str, Any], sh_config: SHConfig, parcels: gpd.GeoDataFrame, out_dir: str | Path
) -> tuple[EOWorkflow, EONode, EONode]:
    """S1 download → parcel-id rasterisation → save. Returns the workflow and the nodes that take per-patch arguments.

    `parcels` must have a `pid` column and should already be in the patches' UTM CRS (saves re-projecting per patch).
    """
    s1 = config["s1"]
    input_task = SentinelHubInputTask(
        data_collection=s1_collection(config),
        resolution=s1["resolution"],
        bands_feature=S1_FEATURE,
        bands=BANDS,
        additional_data=[MASK_FEATURE],
        bands_dtype=np.float32,
        # Scenes of one satellite pass are seconds apart; merge them into one date
        time_difference=dt.timedelta(hours=1),
        timestamp_filter=track_filter(s1.get("track")),
        aux_request_args={"processing": {"backCoeff": s1["back_coeff"], "orthorectify": s1["orthorectify"]}},
        config=sh_config,
        max_threads=4,
    )
    rasterize_task = VectorToRasterTask(
        parcels[["pid", "geometry"]],
        PARCEL_FEATURE,
        values_column="pid",
        raster_shape=S1_FEATURE,
        raster_dtype=np.int32,
        no_data_value=0,
        buffer=s1["edge_buffer_m"],
    )
    save_task = SaveTask(str(out_dir), overwrite_permission=OverwritePermission.OVERWRITE_FEATURES)

    input_node = EONode(input_task)
    rasterize_node = EONode(rasterize_task, inputs=[input_node])
    save_node = EONode(save_task, inputs=[rasterize_node])
    return EOWorkflow([input_node, rasterize_node, save_node]), input_node, save_node


def run_download(
    config: dict[str, Any],
    sh_config: SHConfig,
    parcels: gpd.GeoDataFrame,
    bboxes: list[BBox],
    year: int,
    out_dir: str | Path,
    workers: int = 2,
) -> EOExecutor | None:
    """Downloads one EOPatch per bbox into `out_dir/<patch_name>`. Patches that already exist are skipped
    (returns None if there is nothing to do)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [bbox for bbox in bboxes if not (out_dir / patch_name(bbox)).exists()]
    if not todo:
        return None

    parcels = project(parcels, bboxes[0].crs.pyproj_crs())
    workflow, input_node, save_node = build_workflow(config, sh_config, parcels, out_dir)
    interval = season_interval(config, year)
    execution_kwargs = [
        {input_node: {"bbox": bbox, "time_interval": interval}, save_node: {"eopatch_folder": patch_name(bbox)}}
        for bbox in todo
    ]
    # eo-learn 1.5.7 on Windows nests the logs folder wrongly when given a Windows path; a filesystem rooted at
    # out_dir with logs_folder="/" avoids it. Threads (multiprocess=False) because downloads wait on the network.
    executor = EOExecutor(
        workflow,
        execution_kwargs,
        execution_names=[patch_name(bbox) for bbox in todo],
        save_logs=True,
        filesystem=get_filesystem(str(out_dir), create=True),
        logs_folder="/",
    )
    executor.run(workers=workers, multiprocess=False)
    return executor
