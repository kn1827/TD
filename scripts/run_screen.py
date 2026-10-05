"""S1 — screening + pi samples for one model (one vLLM server, one lane).

    python -m scripts.run_screen --config configs/g1.yaml --model qwen2.5-7b
    python -m scripts.run_screen --config configs/dry_run.yaml --model qwen2.5-7b --fake

Output: results/<run>/screen/<model>.jsonl (one line per item; rerun = resume).
"""

from __future__ import annotations

import argparse
import sys

from tdmad.config import Config
from tdmad.data import load_items, select
from tdmad.llm import CostLog, make_client
from tdmad.prompts import PROMPT_VERSION
from tdmad.provenance import check_data, ensure_protocol, model_fingerprint, record_env
from tdmad.screen import screen_item
from tdmad.utils import JsonlWriter, done_keys, run_bounded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/g1.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", nargs="*", help="subset of screen.tasks")
    ap.add_argument("--limit", type=int, help="at most this many items per task (smoke runs)")
    ap.add_argument("--endpoint", help="override the server URL from the config")
    ap.add_argument("--fake", action="store_true", help="no server: FakeClient (dry run)")
    args = ap.parse_args()

    cfg = Config(args.config)
    cfg.guard_fake(args.fake)
    check_data(cfg.data_dir)
    # same sampling as debate round 0 (shared `sampling` block), so pi = round-0 distribution
    sc = dict(cfg["screen"], **cfg["sampling"])
    items = []
    for task, n in sc["tasks"].items():
        if args.tasks and task not in args.tasks:
            continue
        n = min(n, args.limit) if args.limit else n
        items += select(load_items(cfg.data_dir, task), sc["split"], n)

    out = cfg.run_dir / "screen" / f"{args.model}.jsonl"
    done = done_keys(out, "item_id")
    todo = [it for it in items if it["id"] not in done]
    print(f"[screen {args.model}] {len(items)} items, {len(done)} done, {len(todo)} to run")
    if not todo:
        return
    cost = CostLog(cfg.run_dir, args.model)
    client = make_client(cfg, args.model, cost, args.fake, args.endpoint)
    if args.fake:
        client.register(items)
    client.check()
    ensure_protocol(cfg.run_dir, f"screen_{args.model}", {
        "prompt_version": PROMPT_VERSION, "sampling": cfg["sampling"],
        "n_samples": sc["n_samples"], "logprob_perms": sc.get("logprob_perms", 0),
        "model": model_fingerprint(cfg, args.model)})
    record_env(cfg.run_dir, f"screen_{args.model}", [client], {"n_todo": len(todo)})

    with JsonlWriter(out) as w, cost.stage("screen", args.model, gpus=1, unit="sample",
                                           count_prefix="screen", n_items=len(todo)):
        n_ok = run_bounded(lambda it: screen_item(client, it, sc, cfg["run_name"]), todo,
                           sc["concurrency"], lambda it, rec: w.write(rec), f"screen {args.model}")

    if n_ok < len(todo):  # deadline or failed jobs: stop the lane here; rerun resumes
        sys.exit(3)


if __name__ == "__main__":
    main()
