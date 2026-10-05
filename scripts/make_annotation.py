"""S4 — blinded annotation sheets for G1 criterion 3 (D level recognised >= 85%, kappa >= 0.7).

    python -m scripts.make_annotation --config configs/g1.yaml

Samples annotation.n_per_level valid messages per level (D0..D3), balanced over branch and task
as far as the pool allows, and writes, in results/<run>/annotation/:
  annotator_A.csv, annotator_B.csv   same messages, each in its own random order; no level,
                                     branch, generator or target shown
  answer_key.csv                     ann_id -> msg_id, true level, branch, generator (keep
                                     away from the annotators)
The guide for annotators is docs/annotation_guide.md. Fill the column `label_D` with D0/D1/D2/D3
(plus `n_errors` and `style_leak`), then run analysis.annotation_agreement.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict

from tdmad.config import Config
from tdmad.data import load_items, render_question
from tdmad.utils import read_jsonl

FIELDS_ANN = ["row", "ann_id", "question", "message", "label_D", "n_errors", "style_leak", "comment"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/g1.yaml")
    args = ap.parse_args()
    cfg = Config(args.config)
    ann = cfg["annotation"]
    rng = random.Random(ann["seed"])

    latest = {}
    for r in read_jsonl(cfg.run_dir / "messages" / "messages.jsonl"):
        latest[r["msg_id"]] = r
    msgs = [r for r in latest.values() if "error" not in r and r.get("text")]
    if not msgs:
        raise SystemExit("no messages yet: run scripts.run_messages first")
    items = {it["id"]: it for task in cfg["messages"]["items"]
             for it in load_items(cfg.data_dir, task)}

    picked = []
    for level in ("D0", "D1", "D2", "D3"):
        cells = defaultdict(list)   # (branch, task) -> messages, drawn round-robin for balance
        for m in msgs:
            if m["level"] == level:
                cells[(m["branch"], m["task"])].append(m)
        for v in cells.values():
            rng.shuffle(v)
        keys = sorted(cells)
        got = []
        while len(got) < ann["n_per_level"] and any(cells[k] for k in keys):
            for k in keys:
                if cells[k] and len(got) < ann["n_per_level"]:
                    got.append(cells[k].pop())
        if len(got) < ann["n_per_level"]:
            print(f"[annotation] {level}: only {len(got)} valid messages")
        picked += got

    # msg_id spells out level and branch, so annotators only see an opaque id
    ids = rng.sample(range(1000, 10000), len(picked))
    for m, k in zip(picked, ids):
        m["ann_id"] = f"m{k}"

    out = cfg.run_dir / "annotation"
    out.mkdir(parents=True, exist_ok=True)
    for who in ("A", "B"):
        order = picked[:]
        random.Random(f"{ann['seed']}-{who}").shuffle(order)
        with open(out / f"annotator_{who}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS_ANN)
            w.writeheader()
            for i, m in enumerate(order, 1):
                w.writerow({"row": i, "ann_id": m["ann_id"],
                            "question": render_question(items[m["item_id"]])[0],
                            "message": m["text"], "label_D": "", "n_errors": "",
                            "style_leak": "", "comment": ""})
    with open(out / "answer_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ann_id", "msg_id", "level", "branch", "task",
                                          "generator", "target", "gold"])
        w.writeheader()
        for m in picked:
            w.writerow({"ann_id": m["ann_id"], "msg_id": m["msg_id"], "level": m["level"],
                        "branch": m["branch"], "task": m["task"], "generator": m["generator"], "target": m["target"],
                        "gold": items[m["item_id"]]["answer"]})
    print(f"[annotation] {len(picked)} messages -> {out}")


if __name__ == "__main__":
    main()
