"""Measure a running vLLM server on the three call shapes of the pilot (~5 min per model).

    python -m scripts.bench_throughput --config configs/g1.yaml --model qwen2.5-7b

  prefill  64 requests, ~1,500-token prompt, 1 output token, 16 in flight -> prefill tokens/s
  screen   24 requests, ~120-token prompt, n=10 x 230 tokens            -> GPU-s per sample
  debate   128 requests, ~1,300-token prompt, 230 tokens, 64 in flight  -> GPU-s per turn
Outputs are forced to full length (ignore_eos) and prompts are random words (no prefix-cache
hits), so the numbers are conservative. Writes results/bench/<model>.json, which
scripts.estimate_budget --calibrate reads; compare with the priors before the big run.
"""

from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor

from tdmad import costmodel as cm
from tdmad.config import Config, resolve
from tdmad.llm import make_client
from tdmad.utils import write_json

WORDS = ("time year people way day man thing woman life child world school state family student "
         "group country problem hand part place case week company system program question work "
         "government number night point home water room mother area money story fact month lot "
         "right study book eye job word business issue side kind head house service friend father "
         "power hour game line end member law car city community name president team minute idea "
         "kid body information back parent face others level office door health person art war "
         "history party result change morning reason research girl guy moment air teacher force").split()


def words(n_tokens: int, rng: random.Random) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(int(n_tokens / 1.15)))


def shape(client, n_req, prompt_tokens, completion, n, inflight, seed):
    rng = random.Random(seed)
    prompts = [words(prompt_tokens, rng) for _ in range(n_req)]

    def one(p):
        return client.chat(p + "\nContinue the list.", n=n, temperature=0.8, max_tokens=completion,
                           extra={"ignore_eos": True} if completion > 1 else None, stage="bench")

    t0 = time.time()
    with ThreadPoolExecutor(inflight) as ex:
        res = list(ex.map(one, prompts))
    wall = time.time() - t0
    pt = sum(r.prompt_tokens for r in res)
    ct = sum(r.completion_tokens for r in res)
    return dict(requests=n_req, samples=n_req * n, wall_s=round(wall, 2), prompt_tokens=pt,
                completion_tokens=ct, prompt_tokens_per_req=pt / n_req,
                completion_tokens_per_sample=ct / (n_req * n), gpu_s_per_sample=wall / (n_req * n),
                prompt_tps=pt / wall, completion_tps=ct / wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/g1.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--endpoint")
    args = ap.parse_args()
    cfg = Config(args.config)
    m = cfg.model(args.model)
    client = make_client(cfg, args.model, None, False, args.endpoint)
    client.check()

    out = {"model": args.model, "hf_id": m["hf_id"], "date": time.strftime("%Y-%m-%d %H:%M")}
    print("warm-up ...")
    shape(client, 8, 200, 32, 1, 8, 0)
    out["prefill"] = shape(client, 64, 1500, 1, 1, 16, 1)
    print("prefill", out["prefill"])
    out["screen"] = shape(client, 24, 120, 230, 10, 8, 2)
    print("screen", out["screen"])
    out["debate"] = shape(client, 128, 1300, 230, 1, 64, 3)
    print("debate", out["debate"])

    # predicted by the prior cost model for the same shapes
    pred_screen = cm.call_gpu_s(m, out["screen"]["prompt_tokens_per_req"], 230, 10) / 10
    pred_debate = cm.call_gpu_s(m, out["debate"]["prompt_tokens_per_req"], 230, 1)
    out["prior"] = {"gpu_s_per_sample_screen": pred_screen, "gpu_s_per_turn_debate": pred_debate,
                    "prefill_tps": cm.prefill_tps(m)}
    # calibration for costmodel: effective prefill TFLOPs and a decode speed factor
    prefill_tps = out["prefill"]["prompt_tps"]
    out["calibration"] = {
        "prefill_tflops": prefill_tps * 2 * m["params_b"] * 1e9 / 1e12,
        "decode_speed_factor": pred_screen / out["screen"]["gpu_s_per_sample"],
    }
    p = resolve(cfg.get("results_dir", "results")) / "bench" / f"{args.model}.json"
    write_json(p, out)
    print(f"measured / prior: screen {out['screen']['gpu_s_per_sample'] / pred_screen:.2f}x, "
          f"debate {out['debate']['gpu_s_per_sample'] / pred_debate:.2f}x  -> {p}")


if __name__ == "__main__":
    main()
