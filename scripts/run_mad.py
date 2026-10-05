"""S2 — collect Du et al. (2023) debates for one pool (plan NC2, Block A format).

    python -m scripts.run_mad --config configs/g1.yaml --pool qwen7-homo
    python -m scripts.run_mad --config configs/g1.yaml --pool qwen7-homo --init assigned
    python -m scripts.run_mad --config configs/dry_run.yaml --pool qwen7-homo --fake

Output: results/<run>/mad/<pool>.jsonl, one line per debate (rerun = resume). A debate the
server refuses (HTTP 4xx, e.g. a prompt over --max-model-len) is written to
mad/<pool>_rejected.jsonl and not retried; every other failure is retried on the next run.
Jobs run graph by graph (complete6 first), so a session cut short leaves whole graphs done.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from tdmad.answers import answer_class
from tdmad.config import Config, pool_agents
from tdmad.data import load_items, select
from tdmad.graphs import build
from tdmad.llm import CostLog, RequestRejected, make_client
from tdmad.mad import MEMORY_MODES, DebateRunner, assigned_init
from tdmad.prompts import PROMPT_VERSION
from tdmad.provenance import check_data, ensure_protocol, model_fingerprint, record_env
from tdmad.utils import JsonlWriter, done_keys, read_jsonl, run_bounded, stable_seed


def screening_pools(cfg, model: str, items_by_id: dict) -> dict:
    """item_id -> {"correct": [texts], "wrong": [texts], "wrong_top": [texts]} from S1."""
    out = {}
    for rec in read_jsonl(cfg.run_dir / "screen" / f"{model}.jsonl"):
        it = items_by_id.get(rec["item_id"])
        if it is None:
            continue
        by = {"correct": [], "wrong": []}
        wrong_answers = Counter()
        for s in rec["samples"]:
            if s.get("finish") == "length":   # truncated: not a complete solution to pass on
                continue
            c = answer_class(s["answer"], it)
            if c == "correct":
                by["correct"].append(s["text"])
            elif c is not None:
                by["wrong"].append(s["text"])
                wrong_answers[s["answer"]] += 1
        if wrong_answers:
            top = wrong_answers.most_common(1)[0][0]
            by["wrong_top"] = [s["text"] for s in rec["samples"]
                               if s["answer"] == top and s.get("finish") != "length"]
        out[rec["item_id"]] = by
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/g1.yaml")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--graphs", nargs="*")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--limit", type=int, help="items per task (smoke runs)")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--init", choices=["natural", "assigned"])
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    cfg = Config(args.config)
    cfg.guard_fake(args.fake)
    check_data(cfg.data_dir)
    md = cfg["mad"]
    pool = md["pools"][args.pool]
    agents = pool_agents(pool)
    graphs = args.graphs or md["graphs"]
    seeds = args.seeds or md["seeds"]
    init_mode = args.init or md.get("init", "natural")
    conc = args.concurrency or md["concurrency"]
    memory = md.get("memory", "last_round")
    if memory not in MEMORY_MODES:
        raise SystemExit(f"mad.memory must be one of {MEMORY_MODES}")
    sampling = dict(cfg["sampling"])

    items_by_task = {}
    for task, n in md["tasks"].items():
        if args.tasks and task not in args.tasks:
            continue
        n = min(n, args.limit) if args.limit else n
        items_by_task[task] = select(load_items(cfg.data_dir, task), md["split"], n)
    all_items = [it for its in items_by_task.values() for it in its]

    init_src = None
    if init_mode == "assigned":
        if len(set(agents)) != 1:
            raise SystemExit("--init assigned is implemented for single-model pools only")
        init_src = screening_pools(cfg, agents[0], {it["id"]: it for it in all_items})

    built = {g: build(g, md.get("graph_seed", 0)) for g in graphs}
    for g in built.values():
        if g.n != len(agents):
            raise SystemExit(f"pool {args.pool} has {len(agents)} agents, graph {g.name} has {g.n}")
    jobs = []
    for seed in seeds:
        for gname, g in built.items():
            for task, its in items_by_task.items():
                for it in its:
                    did = f"{args.pool}|{gname}|{task}|{it['id']}|s{seed}"
                    if init_mode == "natural":
                        jobs.append((did, g, it, seed, None))
                        continue
                    for q0 in md["q0_levels"]:
                        rng = random.Random(stable_seed("assign", did, q0))
                        init = assigned_init(init_src.get(it["id"]) or {}, g.n, q0, rng)
                        if init is not None:
                            jobs.append((f"{did}|q{q0}", g, it, seed, init))

    out = cfg.run_dir / "mad" / f"{args.pool}.jsonl"
    rejected = cfg.run_dir / "mad" / f"{args.pool}_rejected.jsonl"
    done = done_keys(out, "debate_id") | done_keys(rejected, "debate_id")
    todo = [j for j in jobs if j[0] not in done]
    print(f"[mad {args.pool}] {len(jobs)} debates, {len(done)} done, {len(todo)} to run "
          f"({len(agents)} agents, graphs {graphs}, memory {memory}, init {init_mode})")
    if not todo:
        return

    tag = f"mad_{args.pool}"
    cost = CostLog(cfg.run_dir, tag)
    clients = {m: make_client(cfg, m, cost, args.fake) for m in sorted(set(agents))}
    for c in clients.values():
        if args.fake:
            c.register(all_items)
        c.check()
    ensure_protocol(cfg.run_dir, tag, {
        "prompt_version": PROMPT_VERSION, "protocol": "du2023", "memory": memory,
        "rounds": md["rounds"], "sampling": sampling, "graph_seed": md.get("graph_seed", 0),
        "agents": agents, "models": {m: model_fingerprint(cfg, m) for m in sorted(set(agents))}})
    record_env(cfg.run_dir, tag, list(clients.values()), {"n_todo": len(todo)})
    model_label = agents[0] if len(set(agents)) == 1 else "+".join(sorted(set(agents)))
    call_pool = ThreadPoolExecutor(max_workers=conc)
    runners = {g.name: DebateRunner(clients, g, agents, md["rounds"], sampling, call_pool,
                                    cfg["run_name"], args.pool, memory) for g in built.values()}

    def one(job):
        did, g, it, seed, init = job
        try:
            return runners[g.name].run(it, seed, init, did)
        except RequestRejected as e:   # permanent: record it, do not retry forever
            return {"debate_id": did, "item_id": it["id"], "graph": g.name, "rejected": str(e)}

    debates_in_flight = max(2, math.ceil(1.5 * conc / len(agents)))
    n_ok = 0
    try:
        with JsonlWriter(out) as w, JsonlWriter(rejected) as wr, \
                cost.stage("mad", model_label, gpus=pool.get("gpus", 1), unit="turn",
                           count_prefix="mad", pool=args.pool, n_debates=len(todo)):
            n_ok = run_bounded(one, todo, debates_in_flight,
                               lambda job, rec: (wr if "rejected" in rec else w).write(rec),
                               f"mad {args.pool}", every=5)
    finally:
        call_pool.shutdown(wait=False, cancel_futures=True)

    if n_ok < len(todo):  # deadline or failed jobs: stop the lane here; rerun resumes
        sys.exit(3)


if __name__ == "__main__":
    main()
