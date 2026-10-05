"""G1 criterion 4: measured GPU-seconds per call vs the estimate fixed in the config.

    python -m analysis.cost_report --config configs/g1.yaml

Reads results/<run>/cost/*_stages.jsonl. For each (model, stage): GPU-s per call =
sum(wall_s * gpus) / sum(calls), where a call is one CoT sample (screen) or one agent turn
(mad). Ratio = measured / estimate; criterion 4 holds when every ratio is within
1 +- tolerance. Also fits effective prefill / decode rates per model from token totals
(GPU-s = a * prompt_tokens + b * completion_tokens) for the budget of NC1-NC3.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from tdmad.config import Config
from tdmad.utils import read_jsonl, write_json

UNIT_KEY = {"screen": "screen_sample", "mad": "mad_call"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/g1.yaml")
    args = ap.parse_args()
    cfg = Config(args.config)
    est = cfg.get("cost_estimate") or {}
    tol = est.get("tolerance", 0.30)
    stages = []
    for p in sorted((cfg.run_dir / "cost").glob("*_stages.jsonl")):
        stages += read_jsonl(p)
    if not stages:
        raise SystemExit("no cost logs under " + str(cfg.run_dir / "cost"))

    agg = defaultdict(lambda: dict(gpu_s=0.0, wall_s=0.0, calls=0, requests=0, prompt_tokens=0,
                                   completion_tokens=0, runs=0))
    for s in stages:
        a = agg[(s["model"], s["stage"])]
        a["gpu_s"] += s["wall_s"] * s.get("gpus", 1)
        a["wall_s"] += s["wall_s"]
        a["calls"] += s.get("samples", 0)
        a["requests"] += s.get("requests", 0)
        a["prompt_tokens"] += s.get("prompt_tokens", 0)
        a["completion_tokens"] += s.get("completion_tokens", 0)
        a["runs"] += 1

    rows, ok_all, any_checked = [], True, False
    for (model, stage), a in sorted(agg.items()):
        if a["calls"] == 0:
            continue
        per = a["gpu_s"] / a["calls"]
        e = (est.get(model) or {}).get(UNIT_KEY.get(stage, ""), None)
        ratio = per / e if e else None
        within = None if ratio is None else abs(ratio - 1) <= tol
        if within is not None:
            any_checked = True
            ok_all &= within
        rows.append(dict(model=model, stage=stage, calls=a["calls"], requests=a["requests"],
                         wall_h=a["wall_s"] / 3600, gpu_h=a["gpu_s"] / 3600,
                         gpu_s_per_call=per, estimate=e, ratio=ratio, within_tolerance=within,
                         prompt_tokens_per_call=a["prompt_tokens"] / a["calls"],
                         completion_tokens_per_call=a["completion_tokens"] / a["calls"]))

    fits = {}
    by_model = defaultdict(list)
    for (model, stage), a in agg.items():
        if a["calls"]:
            by_model[model].append(a)
    for model, lst in by_model.items():
        X = np.array([[a["prompt_tokens"], a["completion_tokens"]] for a in lst], float)
        y = np.array([a["gpu_s"] for a in lst], float)
        if len(lst) >= 2 and np.linalg.matrix_rank(X) == 2:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            if coef[0] > 0 and coef[1] > 0:
                fits[model] = {"prefill_tps": float(1 / coef[0]), "decode_tps": float(1 / coef[1])}

    out = dict(tolerance=tol, rows=rows, rate_fits=fits,
               criterion4=None if not any_checked else bool(ok_all))
    write_json(cfg.run_dir / "analysis" / "cost_summary.json", out)
    for r in rows:
        rs = f"{r['ratio']:.2f}x" if r["ratio"] is not None else "no estimate"
        print(f"{r['model']:<24} {r['stage']:<9} calls {r['calls']:>7,}  "
              f"{r['gpu_s_per_call']:.3f} GPU-s/call  ({rs})  {r['gpu_h']:.2f} GPU-h")
    print("rate fits:", fits)


if __name__ == "__main__":
    main()
