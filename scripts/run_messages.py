"""S3 — D0-D3 messages written by one generator model (needs S1 of every pool model first).

    python -m scripts.run_messages --config configs/g1.yaml --generator phi-4
    python -m scripts.run_messages --config configs/g1.yaml --generator phi-4 --fake

Targets: correct branch = gold answer; wrong branch = the pre-registered lure for numeric traps,
otherwise the most common wrong answer pooled over the pool models' screening samples (the
"sai phổ biến nhất" class). Items with no observed wrong answer are skipped. Items are dealt to
the generators in turn, so each generator writes about half of them.
Output: results/<run>/messages/messages.jsonl and targets.json.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from tdmad.answers import answer_class
from tdmad.config import Config
from tdmad.data import load_items, select
from tdmad.llm import CostLog, make_client
from tdmad.messages import make_messages
from tdmad.prompts import PROMPT_VERSION
from tdmad.provenance import check_data, ensure_protocol, model_fingerprint, record_env
from tdmad.utils import JsonlWriter, read_json, read_jsonl, run_bounded, write_json


def pool_models(cfg) -> list:
    return [lane["model"] for lane in cfg["lanes"].values() if lane.get("model")]


def generators(cfg) -> list:
    out = []
    for lane in cfg["lanes"].values():
        out += lane.get("generators") or []
    return out


def wrong_targets(cfg, items_by_id: dict) -> dict:
    """item_id -> most common wrong answer over all pool models' screening samples."""
    cnt = {}
    for m in pool_models(cfg):
        for rec in read_jsonl(cfg.run_dir / "screen" / f"{m}.jsonl"):
            it = items_by_id.get(rec["item_id"])
            if it is None:
                continue
            c = cnt.setdefault(rec["item_id"], Counter())
            for s in rec["samples"]:
                if answer_class(s["answer"], it) in ("lure", "other"):
                    c[s["answer"]] += 1
    return {k: v.most_common(1)[0][0] for k, v in cnt.items() if v}


def choose_targets(cfg) -> list:
    msg = cfg["messages"]
    gens = generators(cfg)
    rows = []
    for task, quota in msg["items"].items():
        items = select(load_items(cfg.data_dir, task), msg["split"], None)
        wt = wrong_targets(cfg, {it["id"]: it for it in items})
        k = 0
        for it in items:
            if k >= quota:
                break
            wrong = it["lure"] if it.get("lure") is not None else wt.get(it["id"])
            if wrong is None:
                continue
            rows.append({"item_id": it["id"], "task": task, "correct": it["answer"],
                         "wrong": wrong, "wrong_source": "lure" if it.get("lure") else "screening",
                         "generator": gens[len(rows) % len(gens)]})
            k += 1
        if k < quota:
            print(f"[messages] {task}: only {k}/{quota} items have a wrong target")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/g1.yaml")
    ap.add_argument("--generator", required=True)
    ap.add_argument("--endpoint")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--retarget", action="store_true",
                    help="recompute targets.json (default: reuse it, so both generators share it)")
    args = ap.parse_args()

    cfg = Config(args.config)
    cfg.guard_fake(args.fake)
    check_data(cfg.data_dir)
    msg = cfg["messages"]
    tpath = cfg.run_dir / "messages" / "targets.json"
    if tpath.exists() and not args.retarget:
        targets = read_json(tpath)
    else:
        targets = choose_targets(cfg)
        write_json(tpath, targets)
    items = {it["id"]: it for task in msg["items"] for it in load_items(cfg.data_dir, task)}

    out = cfg.run_dir / "messages" / "messages.jsonl"
    done = {r["msg_id"] for r in read_jsonl(out) if "error" not in r}  # failed ones are retried
    jobs = []
    for row in targets:
        if row["generator"] != args.generator:
            continue
        for branch in ("correct", "wrong"):
            if f"{row['item_id']}|{branch}|D2" not in done:
                jobs.append((items[row["item_id"]], branch, row[branch]))
    print(f"[messages {args.generator}] {len(jobs)} (item, branch) jobs to run")
    if not jobs:
        return
    cost = CostLog(cfg.run_dir, f"gen_{args.generator}")
    client = make_client(cfg, args.generator, cost, args.fake, args.endpoint)
    client.check()
    ensure_protocol(cfg.run_dir, f"gen_{args.generator}", {
        "prompt_version": PROMPT_VERSION, "temperature": msg["temperature"],
        "max_tokens": msg["max_tokens"], "attempts": msg["attempts"],
        "model": model_fingerprint(cfg, args.generator)})
    record_env(cfg.run_dir, f"gen_{args.generator}", [client], {"n_todo": len(jobs)})
    with JsonlWriter(out) as w, cost.stage("messages", args.generator, gpus=1, unit="message",
                                           count_prefix="gen"):
        def write(job, recs):
            for r in recs:
                w.write(r)
        n_ok = run_bounded(lambda j: make_messages(client, j[0], j[1], j[2], msg, cfg["run_name"]),
                           jobs, msg["concurrency"], write, f"messages {args.generator}")

    if n_ok < len(jobs):  # deadline or failed jobs: stop the lane here; rerun resumes
        sys.exit(3)


if __name__ == "__main__":
    main()
