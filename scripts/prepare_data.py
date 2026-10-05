"""Download the pilot tasks, assign fixed splits and build the pilot trap set -> data/*.json.

    python -m scripts.prepare_data --config configs/g1.yaml

Run once (needs internet) and ship data/ with the code, so every Kaggle session reads the
same items. Re-running rebuilds the files identically (seeded); data/MANIFEST.json records counts
and sha256 so a changed file is noticed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from collections import Counter

from tdmad.config import Config
from tdmad.data import (assign_splits, build_commonsenseqa, build_gsm8k_platinum, hf_rows,
                        save_items)
from tdmad.traps import misconception_traps, numeric_traps
from tdmad.utils import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/g1.yaml")
    args = ap.parse_args()
    cfg = Config(args.config)
    d = cfg["data"]
    out = cfg.data_dir
    manifest = {"built": dt.datetime.now().isoformat(timespec="seconds"), "files": {}}

    def save(task, items):
        p = save_items(out, task, items)
        manifest["files"][p.name] = {
            "n": len(items), "splits": dict(Counter(it.get("split") for it in items)),
            "families": dict(Counter(it.get("family") for it in items)),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        print(f"{p}: {len(items)} items {manifest['files'][p.name]['splits']}")

    gsm = assign_splits(build_gsm8k_platinum(), d["split_seed"], d["n_pilot"], d["fractions"])
    save("gsm8k_platinum", gsm)
    csqa = assign_splits(build_commonsenseqa(), d["split_seed"], d["n_pilot"], d["fractions"])
    save("commonsenseqa", csqa)

    t = d["traps"]
    num = numeric_traps(t["n_numeric"], t["seed"])
    gen = hf_rows("truthfulqa/truthful_qa", "generation", "validation")
    mc = hf_rows("truthfulqa/truthful_qa", "multiple_choice", "validation")
    mis = misconception_traps(t["n_misconception"], t["seed"], gen, mc)
    save("traps_pilot", num + mis)
    manifest["templates"] = dict(Counter(it["meta"]["template"] for it in num))
    write_json(out / "MANIFEST.json", manifest)


if __name__ == "__main__":
    main()
