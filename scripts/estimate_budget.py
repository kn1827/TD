"""Cost estimate for choosing models and for fixing G1 criterion 4 before the pilot.

    python -m scripts.estimate_budget --config configs/g1.yaml
    python -m scripts.estimate_budget --config configs/g1.yaml --calibrate results/bench/*.json

Prints (1) how each candidate fits on one T4, (2) the G1 pilot workload of the lane models,
(3) the cost_estimate block for configs/g1.yaml, (4) a projection of the whole TD-MAD v2 plan on
Kaggle T4 x2. Writes results/budget.json. All numbers come from tdmad/costmodel.py priors
unless --calibrate gives measured bench files.
"""

from __future__ import annotations

import argparse
import glob

from tdmad import costmodel as cm
from tdmad.config import Config, resolve
from tdmad.utils import read_json, write_json

T4_POOL = ["qwen2.5-3b", "qwen2.5-7b", "llama3.2-3b", "llama3.1-8b", "phi-4-mini", "phi-4",
           "falcon3-3b", "falcon3-7b", "granite3.3-2b", "granite3.3-8b"]
KAGGLE_SESSION_H_PER_WEEK = 30   # GPU quota; one T4 x2 session hour = 2 T4-hours


def gpu_for(model_key: str, calib: dict) -> cm.GPU:
    c = calib.get(model_key)
    if not c:
        return cm.T4
    f = c.get("decode_speed_factor", 1.0)
    return cm.GPU(prefill_tflops=c["prefill_tflops"], bw_gbs=cm.T4.bw_gbs * f,
                  decode_tflops_awq=cm.T4.decode_tflops_awq * f,
                  decode_tflops_fp16=cm.T4.decode_tflops_fp16 * f,
                  step_overhead_ms=cm.T4.step_overhead_ms / f)


def plan_projection(cfg, pool: list, calib: dict, mem: str) -> dict:
    nc1 = {k: cm.nc1_per_model_gpu_h(cfg.model(k), len(pool), gpu_for(k, calib)) for k in pool}
    turn = {k: (cm.nc2_turn_gpu_s(cfg.model(k), 5, gpu_for(k, calib), mem) +
                cm.nc2_turn_gpu_s(cfg.model(k), 2, gpu_for(k, calib), mem)) / 2 for k in pool}
    avg_turn = sum(turn.values()) / len(turn)
    deg3 = [cm.nc2_turn_gpu_s(cfg.model(k), 3, gpu_for(k, calib), mem)
            for k in ("qwen2.5-3b", "qwen2.5-7b")]
    block_a = 10800 * 36 * avg_turn / 3600
    block_b = 4320 * 84 * sum(deg3) / 2 / 3600
    shapley = 240000 * sum(deg3) / 2 / 3600
    nc3 = 3600 * 36 * avg_turn / 3600
    total = sum(nc1.values()) + block_a + block_b + shapley + nc3
    return dict(memory=mem, pool=pool, nc1_gpu_h=nc1, nc2_block_a_gpu_h=block_a,
                nc2_block_b_gpu_h=block_b, nc2_shapley_gpu_h=shapley, nc3_gpu_h=nc3,
                total_gpu_h=total,
                kaggle_weeks_at_full_quota=total / (2 * KAGGLE_SESSION_H_PER_WEEK))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/g1.yaml")
    ap.add_argument("--calibrate", nargs="*", default=[])
    ap.add_argument("--pool", nargs="*", default=T4_POOL, help="10-model pool for the projection")
    args = ap.parse_args()
    cfg = Config(args.config)
    calib = {}
    for pat in args.calibrate:
        for f in glob.glob(pat):
            b = read_json(f)
            calib[b["model"]] = b["calibration"]
    out = {"calibrated": sorted(calib), "models": {}, "g1": {}, "plan": {}}

    # ---- 1. fit on one T4 ------------------------------------------------------------------
    print("\n## 1. Model fit on one Kaggle T4 (16 GB, fp16, vLLM, gpu-memory-utilization 0.90)\n")
    print("| model | quant | weights GB | KV KB/token | KV budget GB | seqs @2k | prefill tok/s "
          "| decode tok/s @2k | GPU-s/sample (screen) | GPU-s/turn (MAD complete6) | T4 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for key, m in cfg.models.items():
        m = dict(m, key=key)
        g = gpu_for(key, calib)
        row = dict(weights_gb=m["weights_gb"], kv_kb=cm.kv_bytes(m) / 1024,
                   kv_budget_gb=cm.kv_budget_gb(m, g), seqs_2k=cm.max_batch(m, 2000, g),
                   prefill_tps=cm.prefill_tps(m, g), decode_tps_2k=cm.decode_tps(m, 2000, g),
                   screen_s=cm.screen_gpu_s(m, "gsm8k_platinum", 30, g) / 30,
                   turn_s=cm.mad_debate_gpu_s(m, "gsm8k_platinum", 5, 6, 5, g)[0] / 36)
        out["models"][key] = row
        print(f"| {key} | {m['quant']} | {row['weights_gb']:.1f} | {row['kv_kb']:.0f} | "
              f"{row['kv_budget_gb']:.1f} | {row['seqs_2k']} | {row['prefill_tps']:,.0f} | "
              f"{row['decode_tps_2k']:,.0f} | {row['screen_s']:.2f} | {row['turn_s']:.2f} | "
              f"{m['t4']} |")

    # ---- 2. G1 pilot -----------------------------------------------------------------------
    print("\n## 2. G1 pilot workload per lane (one T4 each)\n")
    print("| lane | model | screen samples | screen GPU-h | MAD turns | MAD GPU-h | total GPU-h |")
    print("|---|---|---|---|---|---|---|")
    for name, lane in cfg["lanes"].items():
        key = lane["model"]
        w = cm.g1_workload(cfg.model(key), cfg.raw, gpu_for(key, calib))
        out["g1"][key] = w
        print(f"| {name} | {key} | {w['screen_samples']:,} | {w['screen_gpu_h']:.1f} | "
              f"{w['mad_turns']:,} | {w['mad_gpu_h']:.1f} | {w['total_gpu_h']:.1f} |")

    print("\n## 3. cost_estimate for configs/g1.yaml (GPU-s per call on one T4)\n")
    print("cost_estimate:\n  tolerance: 0.30")
    for key, w in out["g1"].items():
        print(f"  {key}: {{screen_sample: {w['screen_gpu_s_per_sample']:.2f}, "
              f"mad_call: {w['mad_gpu_s_per_turn']:.2f}}}")
    mem = cfg["mad"].get("memory", "last_round")
    other = "full" if mem == "last_round" else "last_round"
    print(f"\nSame pilot with mad.memory: {other}:")
    out["g1_" + other] = {}
    for lane in cfg["lanes"].values():
        key = lane["model"]
        w = cm.g1_workload(cfg.model(key), cfg.raw, gpu_for(key, calib), memory=other)
        out["g1_" + other][key] = w
        print(f"  {key}: MAD {w['mad_gpu_h']:.1f} GPU-h ({w['mad_gpu_s_per_turn']:.2f} GPU-s/turn)")

    # ---- 4. whole plan -----------------------------------------------------------------------
    pool = [k for k in args.pool if k in cfg.models]
    plan = out["plan"] = plan_projection(cfg, pool, calib, mem)
    out["plan_" + other] = plan_projection(cfg, pool, calib, other)
    print(f"\n## 4. Whole TD-MAD v2 plan on Kaggle T4 x2 (plan call counts, memory {mem})\n")
    print("| part | T4 GPU-hours |\n|---|---|")
    for k, v in plan["nc1_gpu_h"].items():
        print(f"| NC1 {k} | {v:.0f} |")
    print(f"| NC1 total | {sum(plan['nc1_gpu_h'].values()):.0f} |")
    print(f"| NC2 Block A (10,800 debates, pool-average turn) | {plan['nc2_block_a_gpu_h']:.0f} |")
    print(f"| NC2 Block B (4,320 debates, qwen2.5-3b + qwen2.5-7b) | {plan['nc2_block_b_gpu_h']:.0f} |")
    print(f"| NC2 Shapley replay (~0.24M calls) | {plan['nc2_shapley_gpu_h']:.0f} |")
    print(f"| NC3 intervention (3,600 debates) | {plan['nc3_gpu_h']:.0f} |")
    print(f"| **total** | **{plan['total_gpu_h']:.0f}** |\n")
    for m_, p_ in ((mem, plan), (other, out["plan_" + other])):
        wk = p_["kaggle_weeks_at_full_quota"]
        print(f"memory {m_}: {p_['total_gpu_h']:.0f} T4-h -> {wk:.1f} weeks of Kaggle quota at "
              f"100% use ({wk / 0.7:.1f} at 70%; {KAGGLE_SESSION_H_PER_WEEK} session-h/week x 2 T4)")
    write_json(resolve(cfg.get("results_dir", "results")) / "budget.json", out)


if __name__ == "__main__":
    main()
