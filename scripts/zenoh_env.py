"""Local Zenoh endpoint defaults shared by robot scripts."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_ZENOH_CONNECT = "tcp/127.0.0.1:7447"
CONNECT_ENV_NAMES = ("ZENOH_CONNECT", "XENO_CONNECT")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    env_path = path if path is not None else repo_root() / ".env"
    values: dict[str, str] = {}
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return values

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _clean_dotenv_value(value.strip())
        if key:
            values[key] = value
    return values


def default_zenoh_connect(fallback: str = DEFAULT_ZENOH_CONNECT) -> str:
    for name in CONNECT_ENV_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            return value

    dotenv_values = load_dotenv()
    for name in CONNECT_ENV_NAMES:
        value = dotenv_values.get(name, "").strip()
        if value:
            return value

    return fallback


def _clean_dotenv_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value
