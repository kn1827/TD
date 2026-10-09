"""Back-of-the-envelope numbers for a run: GPU memory, KV cache, disk, generations, time.

    python scripts/estimate.py --config configs/exp.yaml

Memory (fp16, per worker = tensor_parallel_size GPUs):
  weights      = params x 2 bytes
  KV per token = 2 (K and V) x layers x kv_heads x head_dim x 2 bytes
  KV budget    = tp x gpu_gb x gpu_memory_utilization - weights - overhead_gb x tp
Time (assumptions, all overridable; measure them in the smoke run and pass the measured values):
  phase        = start (process + vLLM init + weight load) + prompt tokens / prefill_tps
                 + generated tokens / (decode_tps x min(1, concurrency / ref_batch))
  concurrency  = sequences that fit in the KV cache at the mean sequence length of the phase
                 (decoding is memory-bound: a model whose cache holds few sequences is slower)
  step         = the models of one GPU group run at the same time: step = slowest phase of the group
  prompt size  = round 0: question + suffix; round t: question + suffix + overhead
                 + (peers read + own) x mean response length
Downloads: LRU cache of `disk_budget_gb`, simulated over the run's schedule (hmad.run.schedule).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hmad import choi  # noqa: E402
from hmad.config import load_config, load_models  # noqa: E402
from hmad.plan import build_plan  # noqa: E402
from hmad.run import gpu_groups, model_order, schedule  # noqa: E402

QUESTION_TOKENS = {"gsm8k": 120, "csqa": 70, "arithmetics": 25}
SUFFIX_TOKENS, OVERHEAD_TOKENS = 35, 60


def kv_bytes_per_token(m: dict) -> float:
    layers, kv, hd = m.get("layers") or 32, m.get("kv_heads") or 8, m.get("head_dim") or 128
    return 2 * layers * kv * hd * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp.yaml")
    ap.add_argument("--gpu-gb", type=float, default=15.0, help="usable memory per T4 (16 GB card)")
    ap.add_argument("--overhead-gb", type=float, default=1.5, help="per GPU: CUDA context, activations")
    ap.add_argument("--resp-tokens", type=float, default=300, help="mean response length (max 512)")
    ap.add_argument("--decode-tps", type=float, default=500, help="generated tokens/s per GPU at ref batch")
    ap.add_argument("--tp-efficiency", type=float, default=0.7,
                    help="speed of a model split over tp GPUs = tp x efficiency x one GPU")
    ap.add_argument("--ref-batch", type=float, default=32, help="batch size at which decode-tps holds")
    ap.add_argument("--prefill-tps", type=float, default=2500, help="prompt tokens/s per GPU")
    ap.add_argument("--start-s", type=float, default=120, help="process + vLLM init + weight load")
    ap.add_argument("--download-gbps", type=float, default=0.15, help="HF download, GB/s")
    args = ap.parse_args()

    cfg = load_config(args.config)
    reg = load_models(cfg)
    eng = cfg["engine"]
    util = float(eng.get("gpu_memory_utilization", 0.9))
    tp = int(eng.get("tensor_parallel_size", 1))
    speed = tp * (args.tp_efficiency if tp > 1 else 1.0)    # throughput of one worker vs one GPU
    groups = gpu_groups(eng)
    budget_gb = float(eng.get("disk_budget_gb", 60))
    R = int(cfg["rounds"])
    order = model_order(cfg, reg)

    print(f"## Bộ nhớ (fp16, mỗi mô hình trên {tp} GPU; {len(groups)} mô hình chạy cùng lúc)\n")
    print("| model | family | params (B) | weights GB | KV KB/token | KV budget GB | tokens in KV cache | "
          "4k-token sequences at once |")
    print("|---|---|---|---|---|---|---|---|")
    total_disk, kv_tokens = 0.0, {}
    for m in order:
        r = reg[m]
        w = 2 * r["params"] / 1e9
        total_disk += w
        kvt = kv_bytes_per_token(r)
        budget = tp * args.gpu_gb * util - w - args.overhead_gb * tp
        kv_tokens[m] = max(budget, 0) * 1e9 / kvt
        flag = "" if budget > 1 else " **(does not fit)**"
        print(f"| {m} | {r['family']} | {r['params'] / 1e9:.2f} | {w:.1f} | {kvt / 1024:.0f} | "
              f"{budget:.1f}{flag} | {kv_tokens[m] / 1e3:.0f}k | {kv_tokens[m] / 4096:.0f} |")
    print(f"\nDisk for all {len(order)} models: {total_disk:.0f} GB (cache budget {budget_gb:.0f} GB).\n")

    fake_items = {t: ([""] * s["n_items"], [0] * s["n_items"]) for t, s in cfg["tasks"].items()}
    debates = build_plan(cfg, reg, fake_items)
    calls = {m: [0] * (R + 1) for m in order}
    ptoks = {m: [0.0] * (R + 1) for m in order}
    for d in debates:
        q = QUESTION_TOKENS.get(d.task, 100) + SUFFIX_TOKENS
        for i, m in enumerate(d.models):
            deg = len(choi.peers_of(i, d.n, d.graph))
            for r in range(R + 1):
                calls[m][r] += 1
                ptoks[m][r] += q if r == 0 else q + OVERHEAD_TOKENS + (deg + 1) * args.resp_tokens
    by_setup = {}
    for d in debates:
        by_setup[(d.setup, d.n)] = by_setup.get((d.setup, d.n), 0) + 1
    print("## Số debate và lượt sinh\n")
    for (s, n), k in by_setup.items():
        print(f"- {s}: {k} debates x {n} agents x {R + 1} rounds = {k * n * (R + 1)} generations")
    steps = schedule(order, R, len(groups))
    print(f"- total: {len(debates)} debates, {sum(sum(c) for c in calls.values())} generations, "
          f"{(R + 1) * len(order)} model loads in {len(steps)} steps\n")

    def phase_s(m, r):
        g = calls[m][r]
        if not g:
            return 0.0
        seq = ptoks[m][r] / g + args.resp_tokens
        conc = min(kv_tokens[m] / seq, g)
        return (args.start_s + ptoks[m][r] / (args.prefill_tps * speed)
                + g * args.resp_tokens / (args.decode_tps * speed * min(1.0, conc / args.ref_batch)))

    print("## Thời gian ước tính\n")
    print("| model | generations | sequences at once (round 1) | GPU time h |")
    print("|---|---|---|---|")
    for m in order:
        g1 = calls[m][1] if R >= 1 else calls[m][0]
        seq = (ptoks[m][min(1, R)] / max(g1, 1)) + args.resp_tokens
        print(f"| {m} | {sum(calls[m])} | {min(kv_tokens[m] / seq, g1):.0f} | "
              f"{sum(phase_s(m, r) for r in range(R + 1)) / 3600:.2f} |")
    wall = sum(max(phase_s(m, r) for m in ms) for r, ms in steps)
    gpu_total = sum(phase_s(m, r) for m in order for r in range(R + 1))

    cache, used, downloads, dl_gb = [], {}, 0, 0.0
    for t, (r, ms) in enumerate(steps):
        for m in ms:
            if m not in cache:
                size = 2 * reg[m]["params"] / 1e9
                while cache and sum(2 * reg[c]["params"] / 1e9 for c in cache) + size > budget_gb:
                    cache.remove(min(cache, key=lambda c: used[c]))
                cache.append(m)
                downloads += 1
                dl_gb += size
            used[m] = t
    dl_h = dl_gb / args.download_gbps / 3600
    print(f"\nGPU time {gpu_total / 3600:.1f} h; wall time with {len(groups)} models at once: "
          f"{wall / 3600:.1f} h of session time.")
    print(f"Downloads with a {budget_gb:.0f} GB cache: {downloads} ({dl_gb:.0f} GB, {dl_h:.1f} h at "
          f"{args.download_gbps * 1000:.0f} MB/s), mostly overlapped with generation.")
    print(f"Upper bound (nothing overlaps): {wall / 3600 + dl_h:.1f} h.")


if __name__ == "__main__":
    main()
