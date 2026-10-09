"""One model, one round: generate every prompt of a jobs file, then exit (frees the GPUs for sure).

    python -m hmad.worker --spec <phase>.spec.json        (started by hmad.run)

spec.json: model_key, model_path (local folder), engine {...}, sampling {...}, jobs, out, meta,
fake (bool). jobs.jsonl: {"id", "prompt", "seed", "task", "gold"} (gold is read only by --fake).
Writes out.jsonl ({"id", "text", "finish_reason", "prompt_tokens", "completion_tokens"}) and, last,
meta.json; hmad.run treats a phase as done only when meta.json exists with n == number of jobs.

Sampling follows Choi et al. (HF `generate`, do_sample): temperature and top_p from the config;
with `hf_generate_defaults`, top_k and repetition_penalty come from the model's
generation_config.json as HF generate would take them (top_k 50 and repetition_penalty 1.0 when
absent). Prompts are raw text unless `chat_template` is true (Choi et al. use raw text).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path


def read_jsonl(path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_atomic(path, text: str) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def effective_sampling(model_path: str, sampling: dict) -> dict:
    eff = {"temperature": float(sampling["temperature"]), "top_p": float(sampling["top_p"]),
           "max_tokens": int(sampling["max_tokens"]), "top_k": -1, "repetition_penalty": 1.0}
    if sampling.get("hf_generate_defaults", True):
        gc = {}
        p = Path(model_path) / "generation_config.json"
        if p.exists():
            gc = json.loads(p.read_text(encoding="utf-8"))
        eff["top_k"] = int(gc.get("top_k", 50) or -1)
        eff["repetition_penalty"] = float(gc.get("repetition_penalty", 1.0) or 1.0)
    return eff


# ---------------------------------------------------------------------------------------------
# fake generation (no GPU): a plausible debater, for tests and dry runs
# ---------------------------------------------------------------------------------------------

_FINAL = re.compile(r"\{final answer:\s*([^}]*)\}")


def _fake_wrong(task: str, gold, rng):
    if task == "csqa":
        return f"({rng.choice([c for c in 'ABCDE' if f'({c})' != gold])})"
    return float(gold) + rng.choice([1, 2, 10, -3])


def _fmt(task: str, a) -> str:
    if task == "csqa":
        return f"{{final answer: {a}}}"
    a = float(a)
    return "{final answer: %s}" % (int(a) if a == int(a) else a)


def fake_generate(jobs: list, skill: float) -> list:
    out = []
    for j in jobs:
        rng = random.Random(j["seed"])
        prompt, task, gold = j["prompt"], j["task"], j["gold"]
        if "One of the agents' response" not in prompt and "recent opinion from another agent" not in prompt:
            ans = gold if rng.random() < skill else _fake_wrong(task, gold, rng)
        else:
            own_part = prompt.split("This was your most recent opinion:")[-1]
            peer_part = prompt.split("This was your most recent opinion:")[0]
            own = (_FINAL.findall(own_part) or [None])[0]
            peers = _FINAL.findall(peer_part)
            ans = own
            if peers:
                top = max(set(peers), key=peers.count)
                if rng.random() < 0.3 + 0.5 * peers.count(top) / len(peers):
                    ans = top
            if ans is None:
                ans = gold if rng.random() < skill else _fake_wrong(task, gold, rng)
            if isinstance(ans, str) and task != "csqa":
                try:
                    ans = float(ans)
                except ValueError:
                    ans = 0.0
        text = f"Let me think step by step about this question. {_fmt(task, ans)}"
        out.append({"id": j["id"], "text": text, "finish_reason": "stop",
                    "prompt_tokens": len(prompt) // 4, "completion_tokens": len(text) // 4})
    return out


# ---------------------------------------------------------------------------------------------
# vLLM generation
# ---------------------------------------------------------------------------------------------

def vllm_generate(spec: dict, jobs: list, eff: dict) -> tuple:
    import vllm
    from vllm import LLM, SamplingParams
    eng = spec["engine"]
    t0 = time.time()
    kw = dict(model=spec["model_path"], tokenizer=spec["model_path"], dtype=eng.get("dtype", "half"),
              tensor_parallel_size=int(eng.get("tensor_parallel_size", 1)),
              max_model_len=int(eng.get("max_model_len", 8192)),
              gpu_memory_utilization=float(eng.get("gpu_memory_utilization", 0.9)),
              enforce_eager=bool(eng.get("enforce_eager", True)),
              trust_remote_code=bool(spec.get("trust_remote_code", False)), seed=0)
    llm = LLM(**kw)
    load_s = time.time() - t0
    tok = llm.get_tokenizer()
    limit = int(eng.get("max_model_len", 8192)) - eff["max_tokens"]
    use_template = bool(spec["sampling"].get("chat_template", False))
    inputs, keep, too_long = [], [], 0
    for j in jobs:
        if use_template:
            ids = tok.apply_chat_template([{"role": "user", "content": j["prompt"]}], tokenize=True,
                                          add_generation_prompt=True)
        else:
            ids = tok(j["prompt"]).input_ids        # BOS added as in HF tokenizer(...) (Choi et al.)
        if len(ids) > limit:
            too_long += 1
            continue
        inputs.append({"prompt_token_ids": list(ids)})
        keep.append(j)
    params = [SamplingParams(temperature=eff["temperature"], top_p=eff["top_p"], top_k=eff["top_k"],
                             repetition_penalty=eff["repetition_penalty"], max_tokens=eff["max_tokens"],
                             seed=int(j["seed"])) for j in keep]
    t1 = time.time()
    outs = llm.generate(inputs, params, use_tqdm=True)
    gen_s = time.time() - t1
    res = {}
    for j, o in zip(keep, outs):
        c = o.outputs[0]
        res[j["id"]] = {"id": j["id"], "text": c.text, "finish_reason": c.finish_reason,
                        "prompt_tokens": len(o.prompt_token_ids), "completion_tokens": len(c.token_ids)}
    rows = [res.get(j["id"]) or {"id": j["id"], "text": "", "finish_reason": "prompt_too_long",
                                 "prompt_tokens": None, "completion_tokens": 0} for j in jobs]
    info = {"load_seconds": round(load_s, 1), "generate_seconds": round(gen_s, 1),
            "too_long": too_long, "vllm_version": vllm.__version__, "engine_args": kw}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        pass
    return rows, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    jobs = read_jsonl(spec["jobs"])
    t0 = time.time()
    if spec.get("fake"):
        eff = effective_sampling(spec.get("model_path") or "", spec["sampling"])
        if spec["model_key"] in os.environ.get("HMAD_FAKE_FAIL", "").split(","):   # tests
            raise RuntimeError(f"fake failure of {spec['model_key']} (HMAD_FAKE_FAIL)")
        rows, info = fake_generate(jobs, float(spec.get("fake_skill", 0.7))), {"fake": True}
    else:
        eff = effective_sampling(spec["model_path"], spec["sampling"])
        rows, info = vllm_generate(spec, jobs, eff)
    write_atomic(spec["out"], "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    meta = {"model_key": spec["model_key"], "model_path": spec.get("model_path"),
            "model_source": spec.get("model_source"), "n": len(rows), "seconds": round(time.time() - t0, 1),
            "sampling_effective": eff, **info}
    write_atomic(spec["meta"], json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"[worker {spec['model_key']}] {len(rows)} generations in {meta['seconds']}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
