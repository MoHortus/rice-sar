"""Parcel labels. Each label source is an adapter that returns parcels with a crop code and an `is_rice` flag.

Implemented: the French RPG (Registre Parcellaire Graphique, CAP parcel declarations) from the IGN WFS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests

from .config import DATA_DIR

RPG_WFS_URL = "https://data.geopf.fr/wfs/ows"
RPG_COLUMNS = ["code_cultu", "code_group", "surf_parc"]

# RPG crop groups (code_group), shortened English names
RPG_GROUPS = {
    "1": "soft wheat", "2": "maize", "3": "barley", "4": "other cereals", "5": "rapeseed", "6": "sunflower",
    "7": "other oilseeds", "8": "protein crops", "9": "fibre plants", "11": "fallow", "14": "rice",
    "15": "grain legumes", "16": "fodder", "17": "summer pasture", "18": "permanent grassland",
    "19": "temporary grassland", "20": "orchards", "21": "vineyards", "22": "nuts", "23": "olives",
    "24": "other industrial", "25": "vegetables/flowers", "26": "sugar cane", "28": "miscellaneous",
}


def fetch_rpg(layer: str, bbox: list[float], page_size: int = 1000, timeout: int = 120) -> gpd.GeoDataFrame:
    """Downloads all RPG parcels intersecting `bbox` ([lon_min, lat_min, lon_max, lat_max], WGS84) from the IGN WFS."""
    lon_min, lat_min, lon_max, lat_max = bbox
    params = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetFeature",
        "TYPENAMES": layer,
        "OUTPUTFORMAT": "application/json",
        "COUNT": page_size,
        # In EPSG:4326 this WFS expects latitude first inside the CQL BBOX filter
        "CQL_FILTER": f"BBOX(geom,{lat_min},{lon_min},{lat_max},{lon_max})",
    }
    features: list[dict[str, Any]] = []
    while True:
        response = requests.get(RPG_WFS_URL, params={**params, "STARTINDEX": len(features)}, timeout=timeout)
        response.raise_for_status()
        page = response.json()["features"]
        features.extend(page)
        if len(page) < page_size:
            break

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    gdf["fid"] = [feature["id"] for feature in features]
    return gdf.drop_duplicates("fid").reset_index(drop=True)


def rpg_labels(gdf: gpd.GeoDataFrame, rice_codes: list[str]) -> gpd.GeoDataFrame:
    """Keeps the useful RPG columns, adds `group_name`, `is_rice` and an integer parcel id `pid` (1..N)."""
    labels = gdf[RPG_COLUMNS + ["geometry"]].copy()
    labels["group_name"] = labels["code_group"].map(RPG_GROUPS).fillna("unknown")
    labels["is_rice"] = labels["code_cultu"].isin(rice_codes)
    labels.insert(0, "pid", range(1, len(labels) + 1))
    return labels


def labels_path(config: dict[str, Any], year: int) -> Path:
    return DATA_DIR / "raw" / f"labels_{config['name']}_{year}.gpkg"


def download_labels(config: dict[str, Any], year: int) -> Path:
    """Downloads the labels of one year for the whole study area and saves them as a GeoPackage."""
    label_config = config["labels"]
    if label_config["source"] != "rpg":
        raise NotImplementedError(f"Label source {label_config['source']!r} is not implemented")

    raw = fetch_rpg(label_config["layers"][str(year)], config["aoi"]["bbox"])
    labels = rpg_labels(raw, label_config["rice_codes"])

    path = labels_path(config, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_file(path, driver="GPKG")
    return path


def load_labels(config: dict[str, Any], year: int) -> gpd.GeoDataFrame:
    """Loads the saved labels (run `scripts/download_labels.py` first)."""
    path = labels_path(config, year)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/download_labels.py --year {year}")
    return repair(gpd.read_file(path))


def repair(labels: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """A few declared polygons are self-intersecting (or become so after re-projection); buffer(0) repairs them so
    that clipping, overlays and spatial predicates work."""
    invalid = ~labels.geometry.is_valid
    if invalid.any():
        labels = labels.copy()
        labels.loc[invalid, "geometry"] = labels.geometry[invalid].buffer(0)
    return labels


def project(labels: gpd.GeoDataFrame, crs: Any) -> gpd.GeoDataFrame:
    """Re-projects labels (e.g. to the UTM zone of the EOPatches) and repairs any polygon that became invalid."""
    return repair(labels.to_crs(crs))


def summary(labels: pd.DataFrame) -> pd.DataFrame:
    """Parcel count and area per crop group, largest first."""
    return (
        labels.groupby("group_name")
        .agg(parcels=("pid", "size"), area_ha=("surf_parc", "sum"))
        .sort_values("area_ha", ascending=False)
    )
