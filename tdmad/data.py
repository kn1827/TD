"""Items, fixed splits and question rendering.

Item schema (data/<task>.json is a list of these):
  id        "gsm8kp-0042"            stable across runs
  task      gsm8k_platinum | commonsenseqa | traps_pilot | ...
  format    number | letter | yesno
  question  stem only (options are kept apart so their order can be permuted)
  options   ["bank", "library", ...] or null
  answer    canonical answer ("18", or a letter in the stored option order)
  lure      the pre-registered wrong answer (trap templates) or null
  family    natural | crt | modified_classic | misconception
  split     pilot | nc1 | nc23 | reserve   (plan: "chia cố định thành ba phần rời nhau" + pilot)
"""

from __future__ import annotations

import io
from pathlib import Path

from .answers import LETTERS, canon_number
from .utils import read_json, stable_unit, write_json

SPLITS = ("pilot", "nc1", "nc23", "reserve")


# ---------------------------------------------------------------------------------------------
# loading / selecting
# ---------------------------------------------------------------------------------------------

def load_items(data_dir, task: str) -> list:
    p = Path(data_dir) / f"{task}.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `python -m scripts.prepare_data` first")
    return read_json(p)


def select(items: list, split: str | None, n: int | None, seed=0) -> list:
    """Items of one split in a fixed pseudo-random order; the first n. Two stages that ask for
    100 and 150 items of the same split get nested sets (the 100 are inside the 150)."""
    pool = [it for it in items if split in (None, "all") or it.get("split") == split]
    pool.sort(key=lambda it: stable_unit("order", seed, it["id"]))
    return pool if n is None else pool[:n]


def render_question(item: dict, perm: list | None = None) -> tuple:
    """-> (text shown to the model, display_to_orig) where display_to_orig[k] is the stored
    option index shown at position k. perm=None keeps the stored order."""
    opts = item.get("options")
    if not opts:
        return item["question"], None
    order = list(range(len(opts))) if perm is None else list(perm)
    lines = [f"{LETTERS[k]}) {opts[i]}" for k, i in enumerate(order)]
    return item["question"].rstrip() + "\n" + "\n".join(lines), order


def cyclic_perms(n_options: int, k: int) -> list:
    return [[(i + s) % n_options for i in range(n_options)] for s in range(min(k, n_options))]


def option_letters(item: dict) -> str:
    return LETTERS[: len(item["options"])] if item.get("options") else ""


# ---------------------------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------------------------

def assign_splits(items: list, seed: int, n_pilot: int, fractions: dict) -> list:
    """pilot = the first n_pilot items in a seeded order; the rest is cut by `fractions`."""
    order = sorted(items, key=lambda it: stable_unit("split", seed, it["id"]))
    rest = order[n_pilot:]
    cut1 = round(len(rest) * fractions["nc1"])
    cut2 = cut1 + round(len(rest) * fractions["nc23"])
    for i, it in enumerate(order[:n_pilot]):
        it["split"] = "pilot"
    for i, it in enumerate(rest):
        it["split"] = "nc1" if i < cut1 else "nc23" if i < cut2 else "reserve"
    return items


# ---------------------------------------------------------------------------------------------
# HuggingFace download (datasets if installed, else the parquet files the Hub publishes)
# ---------------------------------------------------------------------------------------------

_PARQUET_API = "https://huggingface.co/api/datasets/{repo}/parquet/{config}/{split}"


def _to_python(x):
    if hasattr(x, "tolist"):
        return _to_python(x.tolist())
    if isinstance(x, dict):
        return {k: _to_python(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_python(v) for v in x]
    return x


def hf_rows(repo: str, config: str = "default", split: str = "test") -> list:
    try:
        from datasets import load_dataset
        name = None if config == "default" else config
        return [dict(r) for r in load_dataset(repo, name, split=split)]
    except ImportError:
        pass
    import pandas as pd
    import requests
    urls = requests.get(_PARQUET_API.format(repo=repo, config=config, split=split), timeout=60).json()
    if not isinstance(urls, list) or not urls:
        raise RuntimeError(f"no parquet files for {repo}/{config}/{split}: {urls}")
    rows = []
    for url in urls:
        resp = requests.get(url, timeout=300)
        resp.raise_for_status()
        df = pd.read_parquet(io.BytesIO(resp.content))
        rows.extend(_to_python(r) for r in df.to_dict(orient="records"))
    return rows


# ---------------------------------------------------------------------------------------------
# natural tasks
# ---------------------------------------------------------------------------------------------

def build_gsm8k_platinum() -> list:
    """madrylab/gsm8k-platinum: GSM8K test with label errors fixed and ambiguous items removed
    (1,209 items; cleaning_status verified / consensus / revised are all kept)."""
    items = []
    for i, r in enumerate(hf_rows("madrylab/gsm8k-platinum", "main", "test")):
        ans = canon_number(r["answer"].split("####")[-1])
        if ans is None:
            continue
        items.append(dict(id=f"gsm8kp-{i:04d}", task="gsm8k_platinum", format="number",
                          question=r["question"].strip(), options=None, answer=ans, lure=None,
                          family="natural", source="madrylab/gsm8k-platinum",
                          meta={"cleaning_status": r.get("cleaning_status")}))
    return items


def build_commonsenseqa() -> list:
    """tau/commonsense_qa validation (the test split has no labels), 5 options."""
    items = []
    for i, r in enumerate(hf_rows("tau/commonsense_qa", "default", "validation")):
        key = str(r.get("answerKey", "")).strip().upper()
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        if key not in labels:
            continue
        items.append(dict(id=f"csqa-{i:04d}", task="commonsenseqa", format="letter",
                          question=r["question"].strip(), options=[t.strip() for t in texts],
                          answer=LETTERS[labels.index(key)], lure=None, family="natural",
                          source="tau/commonsense_qa:validation",
                          meta={"concept": r.get("question_concept", "")}))
    return items


def save_items(data_dir, task: str, items: list) -> Path:
    p = Path(data_dir) / f"{task}.json"
    write_json(p, items)
    return p
