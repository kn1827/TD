"""Back-of-envelope cost model for vLLM on one Kaggle T4, used to choose models and to fix the
per-call estimate of G1 criterion 4 BEFORE the pilot. Every constant is a prior; replace them
with measured numbers from scripts/bench_throughput.py (--calibrate) once a server has run.

Per-call GPU time = prefill + decode:
  prefill  = prompt_tokens * 2 * params / F_prefill                      (compute bound)
  decode   = completion_tokens / decode_tps(ctx, batch)
  decode step (one token for each of B sequences):
           = weights / BW  +  B * ctx * kv_bytes / BW  +  2 * params * B / F_decode  + overhead
The batch B is limited by max_num_seqs and by the KV cache left after the weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

GB = 1e9


@dataclass(frozen=True)
class GPU:
    name: str = "T4"
    mem_gb: float = 15.8          # 15,360 MiB visible to CUDA
    util: float = 0.90            # --gpu-memory-utilization
    reserve_gb: float = 1.3       # activations, CUDA graphs, allocator slack
    bw_gbs: float = 210.0         # effective HBM bandwidth (peak 320)
    prefill_tflops: float = 22.0  # effective fp16 tensor-core throughput on long prompts (peak 65)
    decode_tflops_awq: float = 12.0   # AWQ GEMM at batch 16-64 on Turing
    decode_tflops_fp16: float = 18.0
    step_overhead_ms: float = 4.0     # scheduler + sampling per step (CUDA graphs on)
    max_num_seqs: int = 64


T4 = GPU()


def kv_bytes(m: dict) -> int:
    return 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * 2


def kv_budget_gb(m: dict, gpu: GPU = T4, tp: int = 1) -> float:
    return gpu.mem_gb * gpu.util * tp - m["weights_gb"] - gpu.reserve_gb * tp


def max_batch(m: dict, ctx: float, gpu: GPU = T4, tp: int = 1) -> int:
    budget = kv_budget_gb(m, gpu, tp)
    if budget <= 0:
        return 0
    return int(min(gpu.max_num_seqs, budget * GB // (ctx * kv_bytes(m))))


def decode_tps(m: dict, ctx: float, gpu: GPU = T4, tp: int = 1, batch: int | None = None) -> float:
    b = batch or max_batch(m, ctx, gpu, tp)
    if b <= 0:
        return 0.0
    f = gpu.decode_tflops_awq if m["quant"] == "awq" else gpu.decode_tflops_fp16
    bw = gpu.bw_gbs * GB * tp
    t = (m["weights_gb"] * GB / bw + b * ctx * kv_bytes(m) / bw
         + 2 * m["params_b"] * 1e9 * b / (f * 1e12 * tp) + gpu.step_overhead_ms / 1000
         + (0.002 * m["layers"] / 40 if tp > 1 else 0.0))   # all-reduce over PCIe
    return b / t


def prefill_tps(m: dict, gpu: GPU = T4, tp: int = 1) -> float:
    return gpu.prefill_tflops * 1e12 * tp / (2 * m["params_b"] * 1e9)


def call_gpu_s(m: dict, prompt: float, completion: float, n: int = 1, gpu: GPU = T4,
               tp: int = 1) -> float:
    """GPU-seconds of ONE request (n samples share the prompt), counted on tp GPUs."""
    dt = decode_tps(m, prompt + completion / 2, gpu, tp)
    if dt <= 0:
        return math.inf
    return (prompt / prefill_tps(m, gpu, tp) + n * completion / dt) * tp


# ---- workload shapes (tokens) ----------------------------------------------------------------
# question tokens (options included), CoT completion tokens: assumptions, checked against the
# pilot's cost logs (analysis.cost_report fits the real rates)
TASK_SHAPE = {
    "gsm8k_platinum": dict(q=60, c=230, fmt="number"),
    "commonsenseqa": dict(q=60, c=150, fmt="letter"),
    "trap_numeric": dict(q=70, c=200, fmt="number"),
    "trap_mc": dict(q=85, c=160, fmt="letter"),
}
TEMPLATE = 30          # chat template incl. default system header (Qwen / Llama add one)
R0_INSTR = 40          # Du question prompt around the question
PEER = 8               # "One agent solution: ```...```"
DEBATE_HEAD, DEBATE_FOOT = 12, 40   # construct_message() text (+ the question again for math)


def solve_prompt_tokens(task):
    return TASK_SHAPE[task]["q"] + R0_INSTR + TEMPLATE


def debate_msg_tokens(task, degree):
    s = TASK_SHAPE[task]
    restate = s["q"] if s["fmt"] == "number" else 0
    return DEBATE_HEAD + degree * (s["c"] + PEER) + DEBATE_FOOT + restate


def debate_prompt_tokens(task, degree, t=1, memory="last_round"):
    """Prompt of a round-t turn (t >= 1)."""
    c, m = TASK_SHAPE[task]["c"], debate_msg_tokens(task, degree)
    if memory == "full":
        return solve_prompt_tokens(task) + t * c + t * m
    return solve_prompt_tokens(task) + c + m


def mad_debate_gpu_s(m, task, degree, n_agents, rounds, gpu=T4, memory="last_round"):
    c = TASK_SHAPE[task]["c"]
    g = call_gpu_s(m, solve_prompt_tokens(task), c, 1, gpu)
    for t in range(1, rounds + 1):
        g += call_gpu_s(m, debate_prompt_tokens(task, degree, t, memory), c, 1, gpu)
    return n_agents * g, n_agents * (rounds + 1)


def screen_gpu_s(m, task, n_samples, gpu=T4):
    return call_gpu_s(m, solve_prompt_tokens(task), TASK_SHAPE[task]["c"], n_samples, gpu)


# ---- G1 pilot workload for one model (configs/g1.yaml) ---------------------------------------

def g1_workload(m: dict, cfg: dict, gpu: GPU = T4, memory: str | None = None) -> dict:
    sc, md = cfg["screen"], cfg["mad"]
    n_num_trap = cfg["data"]["traps"]["n_numeric"]
    n_mc_trap = cfg["data"]["traps"]["n_misconception"]
    k_num, k_mc = sc["n_samples"]["number"], sc["n_samples"]["letter"]
    tasks_s = sc["tasks"]
    scr = [("gsm8k_platinum", tasks_s.get("gsm8k_platinum", 0), k_num),
           ("commonsenseqa", tasks_s.get("commonsenseqa", 0), k_mc),
           ("trap_numeric", round(tasks_s.get("traps_pilot", 0) * n_num_trap / (n_num_trap + n_mc_trap)), k_num),
           ("trap_mc", round(tasks_s.get("traps_pilot", 0) * n_mc_trap / (n_num_trap + n_mc_trap)), k_mc)]
    s_gpu = sum(n * screen_gpu_s(m, t, k, gpu) for t, n, k in scr)
    s_samples = sum(n * k for _, n, k in scr)
    n_lp = (tasks_s.get("commonsenseqa", 0) + scr[3][1]) * sc.get("logprob_perms", 0)
    s_gpu += n_lp * solve_prompt_tokens("commonsenseqa") / prefill_tps(m, gpu)

    deg = {"complete6": 5, "ring6": 2, "heawood14": 3, "rr3_14": 3, "cluster3_14": 3}
    n_ag = {"complete6": 6, "ring6": 6, "heawood14": 14, "rr3_14": 14, "cluster3_14": 14}
    memory = memory or md.get("memory", "last_round")
    tm = md["tasks"]
    mad_tasks = [("gsm8k_platinum", tm.get("gsm8k_platinum", 0)),
                 ("commonsenseqa", tm.get("commonsenseqa", 0)),
                 ("trap_numeric", round(tm.get("traps_pilot", 0) * n_num_trap / (n_num_trap + n_mc_trap))),
                 ("trap_mc", round(tm.get("traps_pilot", 0) * n_mc_trap / (n_num_trap + n_mc_trap)))]
    m_gpu = m_turns = 0.0
    for gname in md["graphs"]:
        for t, n in mad_tasks:
            g, turns = mad_debate_gpu_s(m, t, deg[gname], n_ag[gname], md["rounds"], gpu, memory)
            m_gpu += n * g * len(md["seeds"])
            m_turns += n * turns * len(md["seeds"])
    return dict(screen_samples=s_samples, screen_gpu_h=s_gpu / 3600,
                screen_gpu_s_per_sample=s_gpu / max(1, s_samples),
                mad_turns=int(m_turns), mad_gpu_h=m_gpu / 3600,
                mad_gpu_s_per_turn=m_gpu / max(1, m_turns),
                total_gpu_h=(s_gpu + m_gpu) / 3600)


# ---- whole-plan projection (TD-MAD v2 call counts) -------------------------------------------

def nc1_per_model_gpu_h(m: dict, n_models: int = 10, gpu: GPU = T4) -> float:
    """NC1: 7,200 (model, item) units x 34 exposure calls + 0.09M side-branch calls, split over
    the pool; exposure context = a debate turn with a 4-slot panel (question prompt, own answer,
    debate message with 4 messages). Screening + pi: 0.35M short samples."""
    units = 7200 / n_models
    expo_calls = units * 34 + 90000 / n_models
    c = TASK_SHAPE["trap_numeric"]["c"]
    g = expo_calls * call_gpu_s(m, debate_prompt_tokens("trap_numeric", 4), c, 1, gpu)
    g += (350000 / n_models) * call_gpu_s(m, solve_prompt_tokens("trap_numeric"), c, 20, gpu) / 20
    return g / 3600


def nc2_turn_gpu_s(m: dict, degree: int, gpu: GPU = T4, memory: str = "last_round",
                   rounds: int = 5) -> float:
    """Mean GPU-s of one agent turn of a T-round debate (round 0 included)."""
    g, turns = mad_debate_gpu_s(m, "trap_numeric", degree, 1, rounds, gpu, memory)
    return g / turns


def with_calibration(gpu: GPU, bench: dict | None) -> GPU:
    """bench: {"prefill_tflops": .., "decode_tflops_awq": .., "bw_gbs": ..} from bench_throughput."""
    if not bench:
        return gpu
    keys = {k: v for k, v in bench.items() if k in GPU.__dataclass_fields__}
    return replace(gpu, **keys)
