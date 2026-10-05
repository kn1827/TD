"""Config loading. Paths in the yaml are relative to the TD-MAD root (the folder holding configs/)."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_yaml(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class Config:
    def __init__(self, path):
        self.path = resolve(path)
        self.raw = load_yaml(self.path)
        if "extends" in self.raw:  # e.g. g1_smoke.yaml = g1.yaml with a few values changed
            self.raw = deep_merge(Config(self.raw["extends"]).raw, self.raw)
            self.raw.pop("extends")
        self.models = load_yaml(resolve(self.raw.get("models_file", "configs/models.yaml")))

    def __getitem__(self, key):
        return self.raw[key]

    def get(self, key, default=None):
        return self.raw.get(key, default)

    def guard_fake(self, fake: bool) -> None:
        """Fake answers must never land in a real run folder."""
        if fake and "dry" not in self.raw["run_name"]:
            raise SystemExit(f"--fake writes fake data: use a dry-run config (run_name with 'dry'), "
                             f"e.g. --config configs/dry_run.yaml, not {self.path.name}")

    # ---- paths ------------------------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return resolve(self.raw.get("data_dir", "data"))

    @property
    def run_dir(self) -> Path:
        return resolve(self.raw.get("results_dir", "results")) / self.raw["run_name"]

    # ---- models -----------------------------------------------------------------------------
    def model(self, key: str) -> dict:
        if key not in self.models:
            raise KeyError(f"model '{key}' is not in {self.raw.get('models_file')}")
        return dict(self.models[key], key=key)

    def hf_id(self, key: str) -> str:
        return self.model(key)["hf_id"]

    def endpoints(self, key: str) -> list:
        urls = (self.raw.get("servers") or {}).get(key)
        if not urls:
            raise KeyError(f"no server for model '{key}' under `servers:` in {self.path.name}")
        return [urls] if isinstance(urls, str) else list(urls)

    def lane_of(self, model_key: str) -> dict | None:
        for name, lane in (self.raw.get("lanes") or {}).items():
            if lane.get("model") == model_key or model_key in (lane.get("generators") or []):
                return dict(lane, name=name)
        return None


def pool_agents(pool: dict) -> list:
    """`agents: {qwen2.5-7b: 6}` or `agents: [a, b, a, b]` -> list of model keys, one per agent."""
    a = pool["agents"]
    if isinstance(a, dict):
        out = []
        for k, n in a.items():
            out += [k] * int(n)
        return out
    return list(a)


def get_path(cfg: dict, dotted: str):
    """cfg_get helper: 'lanes.lane0.model' -> value."""
    cur = cfg
    for part in dotted.split("."):
        cur = cur[part]
    return cur
