"""Shared helpers for the analysis scripts (CPU only, numpy)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from tdmad.data import load_items


def items_index(cfg, tasks) -> dict:
    out = {}
    for t in tasks:
        try:
            for it in load_items(cfg.data_dir, t):
                out[it["id"]] = it
        except FileNotFoundError:
            pass
    return out


def cluster_bootstrap(groups: list, stat, B: int = 2000, seed: int = 0):
    """groups: list of per-cluster lists of events; stat(flat_list) -> float. 95% percentile CI."""
    groups = [g for g in groups if g]
    if not groups:
        return (None, None)
    rng = np.random.default_rng(seed)
    n = len(groups)
    vals = []
    for _ in range(B):
        flat = [e for i in rng.integers(0, n, n) for e in groups[i]]
        try:
            v = stat(flat)
        except ZeroDivisionError:
            continue
        if v == v:
            vals.append(v)
    if not vals:
        return (None, None)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def rate_ci(groups: list, B: int = 2000, seed: int = 0) -> dict:
    flat = [e for g in groups for e in g]
    if not flat:
        return {"n": 0, "p": None, "lo": None, "hi": None}
    lo, hi = cluster_bootstrap(groups, lambda x: sum(x) / len(x), B, seed)
    return {"n": len(flat), "p": sum(flat) / len(flat), "lo": lo, "hi": hi}


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_brown(r: float) -> float:
    return 2 * r / (1 + r) if r == r and r > -1 else float("nan")


def write_csv(path, rows: list, fields: list | None = None) -> None:
    if not rows:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:  # union of keys, first-seen order
        fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def fmt_ci(d: dict, pct: bool = True) -> str:
    if not d or d.get("p") is None:
        return "—"
    k = 100 if pct else 1
    if d.get("lo") is None:
        return f"{d['p'] * k:.1f} (n={d['n']})"
    return f"{d['p'] * k:.1f} [{d['lo'] * k:.1f}, {d['hi'] * k:.1f}] (n={d['n']})"
