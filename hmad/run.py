"""Run heterogeneous debates: all debates advance round by round, one model at a time.

    python -m hmad.run --config configs/exp.yaml
    python -m hmad.run --config configs/exp.yaml --fake          (no GPU, fake answers)
    python -m hmad.run --config configs/exp.yaml --plan-only     (write plan.json and stop)

Why this order: every agent of a debate is a different model and a GPU holds one model at a time.
Round r of every debate needs round r-1 of all its agents, so the run goes: for each round, for each
model, generate that model's turns in ALL debates of the round (one vLLM subprocess,
results/<run>/phases/r<r>/<model>.*). With `engine.gpus` = [0, 1] and tensor_parallel_size 1, two
models run at the same time, one per GPU (a group). That is (rounds + 1) x (number of models) model
loads for the whole run, whatever the number of debates. Groups alternate direction from one round
to the next so that the models used last are still on disk at the start of the next round, and the
next group is downloaded while the current one generates.

Resume: a phase whose meta.json is complete is never redone; rerun the same command after a
cut-off. plan.json and protocol.json are written once; a run with a different protocol (models,
sampling, rounds, revisions...) in the same folder is refused.
Deadline: if TD_DEADLINE_TS (unix time) is set, no phase is started that is expected to end after it
(expected length = longest phase so far, x 1.2); exit code 3 means "stopped early, rerun to resume".
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import choi
from .config import ROOT, load_config, load_models, resolve, run_dir, stable_seed
from .plan import build_plan, load_plan, save_plan
from .results import assemble, job_id, summary, write


def read_jsonl(path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def protocol(cfg: dict, registry: dict) -> dict:
    used = sorted({m for s in cfg["setups"].values() for m in s["models"]})
    return {"rounds": cfg["rounds"], "seed": cfg.get("seed", 0), "tasks": cfg["tasks"],
            "setups": cfg["setups"], "sampling": cfg["sampling"],
            "engine": {k: v for k, v in cfg["engine"].items()
                       if k in ("tensor_parallel_size", "dtype", "max_model_len")},
            "models": {m: {"hf_id": registry[m]["hf_id"], "revision": registry[m].get("revision")}
                       for m in used},
            "implementation": "Choi, Zhu & Li 2025 (debate-or-vote @82c929e), ported in hmad/choi.py"}


def check_protocol(rd: Path, proto: dict) -> None:
    p = rd / "protocol.json"
    proto = json.loads(json.dumps(proto))
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
        if old != proto:
            diff = sorted(k for k in set(old) | set(proto) if old.get(k) != proto.get(k))
            raise SystemExit(f"{p} differs in {diff}: use a new run_name instead of mixing protocols")
        return
    rd.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(proto, ensure_ascii=False, indent=1), encoding="utf-8")


def model_order(cfg: dict, registry: dict) -> list:
    used = {m for s in cfg["setups"].values() for m in s["models"]}
    return sorted(used, key=lambda m: (registry[m].get("rank", 99), m))


def gpu_groups(eng: dict) -> list:
    """GPU sets of the workers that run at the same time: gpus split in chunks of tp."""
    gpus = list(eng.get("gpus") or [0])
    tp = int(eng.get("tensor_parallel_size", 1))
    groups = [gpus[i:i + tp] for i in range(0, len(gpus) - tp + 1, tp)]
    if not groups:
        raise ValueError(f"tensor_parallel_size {tp} > number of gpus {gpus}")
    return groups


def schedule(order: list, rounds: int, width: int) -> list:
    """[(round, [models run together])], groups reversed every other round."""
    groups = [order[i:i + width] for i in range(0, len(order), width)]
    return [(r, g) for r in range(rounds + 1) for g in (groups if r % 2 == 0 else groups[::-1])]


class Phase:
    def __init__(self, rd: Path, r: int, model: str):
        base = rd / "phases" / f"r{r}"
        base.mkdir(parents=True, exist_ok=True)
        self.r, self.model = r, model
        self.jobs, self.out = base / f"{model}.jobs.jsonl", base / f"{model}.out.jsonl"
        self.meta, self.spec = base / f"{model}.meta.json", base / f"{model}.spec.json"
        self.log = rd / "logs" / f"r{r}_{model}.log"

    def done(self) -> bool:
        if not (self.meta.exists() and self.jobs.exists() and self.out.exists()):
            return False
        n_jobs = sum(1 for line in open(self.jobs, encoding="utf-8") if line.strip())
        return json.loads(self.meta.read_text(encoding="utf-8")).get("n") == n_jobs

    def seconds(self) -> float:
        return json.loads(self.meta.read_text(encoding="utf-8")).get("seconds", 0.0)


def build_jobs(debates: list, r: int, model: str, prev: dict, cfg: dict) -> list:
    """Turns of `model` at round r in every debate. prev: job id -> text of round r-1."""
    jobs = []
    sc = cfg["sampling"]
    for d in debates:
        positions = [i for i, m in enumerate(d.models) if m == model]
        if not positions:
            continue
        suffix = choi.instruction_suffix(d.task, sc.get("bae", False), sc.get("cot", False))
        if r == 0:
            prompts = {i: choi.first_prompt(d.question, suffix) for i in positions}
        else:
            last = [prev[job_id(d.id, r - 1, i)] for i in range(d.n)]
            msgs = choi.round_messages(d.question, last, d.graph, suffix)
            prompts = {i: msgs[i] for i in positions}
        for i, p in prompts.items():
            jid = job_id(d.id, r, i)
            jobs.append({"id": jid, "prompt": p, "seed": stable_seed(cfg.get("seed", 0), jid),
                         "task": d.task, "gold": d.gold})
    return jobs


def start_worker(phase: Phase, spec: dict, gpus: list | None):
    phase.spec.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    phase.log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "hmad.worker", "--spec", str(phase.spec)]
    env = dict(os.environ)
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    log = open(phase.log, "a", encoding="utf-8")
    log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} GPUs {gpus}: {' '.join(cmd)}\n")
    log.flush()
    return subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env), log


def error_lines(log_path: Path, k: int = 6) -> list:
    """The last k distinct lines of a log that look like errors (the real cause of a vLLM failure
    is often far above the final traceback)."""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    hits = [ln.strip() for ln in lines if re.search(r"(Error|error:|Exception|not supported|failed)", ln)]
    out = []
    for ln in reversed(hits):
        if ln not in out:
            out.append(ln)
        if len(out) == k:
            break
    return out[::-1]


def finish_worker(phase: Phase, proc, log, keep_going: bool = False) -> str | None:
    """Wait for a worker. On failure: raise, or with keep_going (smoke tests) write empty outputs
    marked finish_reason='worker_error' so the other models' debates go on, and return the error."""
    rc = proc.wait()
    log.close()
    if rc == 0 and phase.done():
        return None
    errs = error_lines(phase.log)
    if not keep_going:
        tail = phase.log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError(f"worker for {phase.model} round {phase.r} failed (exit {rc}); "
                           f"log {phase.log}:\n" + "\n".join(tail)
                           + "\n--- error lines:\n" + "\n".join(errs))
    jobs = read_jsonl(phase.jobs)
    phase.out.write_text("".join(json.dumps({"id": j["id"], "text": "", "finish_reason": "worker_error",
                                             "prompt_tokens": None, "completion_tokens": 0}) + "\n"
                                 for j in jobs), encoding="utf-8")
    phase.meta.write_text(json.dumps({"model_key": phase.model, "n": len(jobs), "failed": True,
                                      "exit_code": rc, "errors": errs, "seconds": 0.0}, indent=1),
                          encoding="utf-8")
    print(f"[run] round {phase.r} / {phase.model}: FAILED (exit {rc}), continuing:\n  "
          + "\n  ".join(errs), flush=True)
    return "; ".join(errs[-2:])


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp.yaml")
    ap.add_argument("--fake", action="store_true", help="no GPU: fake answers (testing)")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    registry = load_models(cfg)
    rd = run_dir(cfg)
    if args.fake and "dry" not in cfg["run_name"] and "test" not in cfg["run_name"]:
        cfg["run_name"] += "_dry"
        rd = run_dir(cfg)
    check_protocol(rd, protocol(cfg, registry))
    plan_path = rd / "plan.json"
    if plan_path.exists():
        debates = load_plan(plan_path)
    else:
        debates = build_plan(cfg, registry)
        save_plan(plan_path, debates)
    R = int(cfg["rounds"])
    order = model_order(cfg, registry)
    print(f"[run] {len(debates)} debates, {R + 1} rounds, {len(order)} models -> "
          f"{(R + 1) * len(order)} phases; results in {rd}", flush=True)
    if args.plan_only:
        return

    eng = cfg["engine"]
    store = None
    if not args.fake:
        from .weights import WeightStore
        store = WeightStore(registry, resolve(eng.get("model_cache_dir", ".cache/models")),
                            float(eng.get("disk_budget_gb", 60)), eng.get("model_search_paths") or [])
    deadline = float(os.environ.get("TD_DEADLINE_TS", "0") or 0)
    groups = gpu_groups(eng)
    steps = schedule(order, R, len(groups))
    longest = 0.0
    outputs = {}
    for k, (r, models) in enumerate(steps):
        todo = [m for m in models if not Phase(rd, r, m).done()]
        for m in models:
            if m not in todo:
                longest = max(longest, Phase(rd, r, m).seconds())
        if not todo:
            continue
        if r > 0 and not outputs.get(r - 1):
            outputs[r - 1] = {}
            for mm in order:
                p = Phase(rd, r - 1, mm)
                if p.out.exists():
                    for row in read_jsonl(p.out):
                        outputs[r - 1][row["id"]] = row["text"]
        if deadline and longest and time.time() + 1.2 * longest > deadline:
            print(f"[run] deadline: stopping before round {r} / {todo} (rerun to resume)", flush=True)
            sys.exit(3)
        t0 = time.time()
        running = []
        for m, gpus in zip(todo, groups):
            phase = Phase(rd, r, m)
            jobs = build_jobs(debates, r, m, outputs.get(r - 1, {}), cfg)
            phase.jobs.write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in jobs),
                                  encoding="utf-8")
            spec = {"model_key": m, "jobs": str(phase.jobs), "out": str(phase.out),
                    "meta": str(phase.meta), "engine": eng, "sampling": cfg["sampling"],
                    "trust_remote_code": bool(registry[m].get("trust_remote_code", False))}
            if args.fake:
                spec.update(fake=True, fake_skill=0.9 - 0.03 * registry[m].get("rank", 10),
                            model_path=None, model_source="fake")
            else:
                path, source = store.get(m, protect=set(todo))
                spec.update(model_path=path, model_source=source)
            print(f"[run] round {r} / {m} on GPU {gpus}: {len(jobs)} generations ...", flush=True)
            running.append((phase, *start_worker(phase, spec, None if args.fake else gpus)))
        if store is not None:      # download the next group while this one generates
            for rr, nxt in steps[k + 1:k + 2]:
                for mm in nxt:
                    if not Phase(rd, rr, mm).done():
                        store.prefetch(mm, protect=set(todo))
        for phase, proc, log in running:
            finish_worker(phase, proc, log, keep_going=bool(eng.get("continue_on_error", False)))
        longest = max(longest, time.time() - t0)
        print(f"[run] round {r} / {todo}: done in {time.time() - t0:.0f}s", flush=True)

    # assemble
    outputs_all = {}
    metas = {}
    for r in range(R + 1):
        for m in order:
            p = Phase(rd, r, m)
            for row in read_jsonl(p.out):
                outputs_all[row["id"]] = row
            metas[f"r{r}/{m}"] = json.loads(p.meta.read_text(encoding="utf-8"))
    prov = {"git_commit": git_commit(), "config": args.config, "fake": args.fake,
            "models": {m: {"hf_id": registry[m]["hf_id"], "revision": registry[m].get("revision"),
                           "family": registry[m]["family"],
                           "source": metas[f"r0/{m}"].get("model_source"),
                           "failed_phases": [k for k, v in metas.items() if k.endswith("/" + m) and v.get("failed")],
                           "sampling_effective": metas[f"r0/{m}"].get("sampling_effective"),
                           "vllm_version": metas[f"r0/{m}"].get("vllm_version")} for m in order}}
    recs = assemble(debates, outputs_all, R, registry, cfg.get("seed", 0), prov)
    write(rd / "debates.jsonl", recs)
    failed = {k: v for k, v in metas.items() if v.get("failed")}
    text = summary(recs, R)
    if failed:
        text += "\n## Failed phases (model did not run; its turns are empty)\n\n"
        text += "\n".join(f"- {k}: " + " | ".join(v.get("errors") or ["?"])
                          for k, v in sorted(failed.items()))
        text += "\n"
    (rd / "summary.md").write_text(text, encoding="utf-8")
    print(f"[run] {len(recs)} debates -> {rd / 'debates.jsonl'}; summary in {rd / 'summary.md'}")


if __name__ == "__main__":
    main()
