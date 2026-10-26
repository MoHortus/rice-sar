"""Project paths, region config and Sentinel Hub (CDSE) credentials."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import tomli
from sentinelhub import SHConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

CDSE_BASE_URL = "https://sh.dataspace.copernicus.eu"
CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


def load_config(path: str | Path) -> dict[str, Any]:
    """Reads a region config, e.g. `configs/camargue.toml`. Relative paths are resolved from the project root."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "rb") as file:
        return tomli.load(file)


def load_dotenv(path: str | Path = PROJECT_ROOT / ".env") -> None:
    """Minimal `.env` reader: `KEY=value` lines go into `os.environ` unless the variable is already set."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


def get_sh_config() -> SHConfig:
    """SHConfig for the Copernicus Data Space Ecosystem.

    Credentials come from the environment (or `.env`): either `SH_CLIENT_ID` + `SH_CLIENT_SECRET`, or `SH_PROFILE`
    naming a profile saved with sentinelhub-py. `SHConfig` reads these variables itself.
    """
    load_dotenv()
    config = SHConfig()
    if not config.sh_client_id or not config.sh_client_secret:
        raise RuntimeError(
            "No Sentinel Hub credentials found. Copy .env.example to .env and set SH_CLIENT_ID and SH_CLIENT_SECRET "
            "(or SH_PROFILE)."
        )
    config.sh_base_url = CDSE_BASE_URL
    config.sh_token_url = CDSE_TOKEN_URL
    return config
