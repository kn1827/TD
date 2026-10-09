"""Configs and the model registry. Paths are relative to the repository root."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REPLACED_KEYS = ("setups", "tasks")    # a child config replaces these whole instead of merging


def resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _merge(base: dict, over: dict, top: bool = True) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and not (top and k in REPLACED_KEYS):
            out[k] = _merge(out[k], v, top=False)
        else:
            out[k] = v
    return out


def load_config(path) -> dict:
    with open(resolve(path), encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "extends" in cfg:
        parent = load_config(cfg.pop("extends"))
        cfg = _merge(parent, cfg)
    return cfg


def load_models(cfg: dict) -> dict:
    with open(resolve(cfg.get("models_file", "configs/models.yaml")), encoding="utf-8") as f:
        reg = yaml.safe_load(f)
    models = dict(reg.get("models") or {})
    for k, v in (reg.get("reserves") or {}).items():
        models.setdefault(k, dict(v, reserve=True))
    for k, v in models.items():
        v["key"] = k
    return models


def run_dir(cfg: dict) -> Path:
    return resolve(cfg.get("results_dir", "results")) / cfg["run_name"]


def stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)
