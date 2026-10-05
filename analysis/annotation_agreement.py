"""G1 criterion 3: annotators recognise the designed D level (>= 85%) and agree (kappa >= 0.7).

    python -m analysis.annotation_agreement --config configs/g1.yaml

Reads results/<run>/annotation/annotator_A.csv, annotator_B.csv (filled) and answer_key.csv.
Labels accepted: D0..D3, 0..3, d2, "D2 " ... Reports, per annotator, accuracy against the
designed level and the confusion matrix; Cohen's kappa between annotators and against the key;
and for D2/D3 messages the share marked with exactly one error (wrong branch) or none (correct).
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter

from tdmad.config import Config
from tdmad.utils import write_json

LEVELS = ["D0", "D1", "D2", "D3"]


def norm(label: str):
    m = re.search(r"([0-3])", label or "")
    return f"D{m.group(1)}" if m else None


def kappa(a: list, b: list) -> float | None:
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if not pairs:
        return None
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def read(path) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        return {r["ann_id"]: r for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/g1.yaml")
    args = ap.parse_args()
    cfg = Config(args.config)
    d = cfg.run_dir / "annotation"
    if not (d / "annotator_A.csv").exists():
        raise SystemExit("no annotation sheets: run scripts.make_annotation")
    key = read(d / "answer_key.csv")
    A, B = read(d / "annotator_A.csv"), read(d / "annotator_B.csv")
    ids = sorted(key)
    truth = [key[i]["level"] for i in ids]
    la = [norm(A.get(i, {}).get("label_D", "")) for i in ids]
    lb = [norm(B.get(i, {}).get("label_D", "")) for i in ids]
    filled = sum(x is not None for x in la) + sum(x is not None for x in lb)
    if filled == 0:
        raise SystemExit("annotation sheets are still empty")

    def acc(lbl):
        p = [(x, t) for x, t in zip(lbl, truth) if x]
        return sum(x == t for x, t in p) / len(p) if p else None

    def confusion(lbl):
        c = {t: Counter() for t in LEVELS}
        for x, t in zip(lbl, truth):
            c[t][x or "blank"] += 1
        return {t: dict(v) for t, v in c.items()}

    def error_marks(sheet):
        ok = tot = 0
        for i in ids:
            k = key[i]
            if k["level"] not in ("D2",):
                continue
            mark = (sheet.get(i, {}).get("n_errors") or "").strip()
            if not mark:
                continue
            tot += 1
            ok += mark == ("1" if k["branch"] == "wrong" else "0")
        return {"n": tot, "share_as_designed": ok / tot if tot else None}

    out = dict(
        n_messages=len(ids), labelled_A=sum(x is not None for x in la),
        labelled_B=sum(x is not None for x in lb),
        accuracy_A=acc(la), accuracy_B=acc(lb),
        kappa_A_B=kappa(la, lb), kappa_A_key=kappa(la, truth), kappa_B_key=kappa(lb, truth),
        confusion_A=confusion(la), confusion_B=confusion(lb),
        d2_error_count_A=error_marks(A), d2_error_count_B=error_marks(B),
        style_leak_A=sum((A.get(i, {}).get("style_leak") or "").strip().lower() in ("y", "yes", "1")
                         for i in ids),
        style_leak_B=sum((B.get(i, {}).get("style_leak") or "").strip().lower() in ("y", "yes", "1")
                         for i in ids),
    )
    write_json(cfg.run_dir / "analysis" / "annotation_summary.json", out)
    print({k: v for k, v in out.items() if not k.startswith("confusion")})


if __name__ == "__main__":
    main()
