"""JSON configuration. One snapshot per attempt is passed to every stage and bound into the approval (spec §5.3)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
CONFIG_NAMES = ("verifier", "judge", "physics")


def load_config(name: str) -> dict:
    """Load config/<name>.json."""
    with open(CONFIG_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def config_snapshot() -> dict[str, dict]:
    """All configs, loaded once. Callers pass this dict down instead of re-reading disk mid-attempt."""
    return {name: load_config(name) for name in CONFIG_NAMES}


def config_hash(cfg: dict) -> str:
    """Stable sha256 of a config dict (sorted keys, no whitespace)."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
