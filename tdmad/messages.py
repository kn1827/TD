"""S3 — messages at four demonstrability levels (plan: "Thao tác demonstrability trên từng tin nhắn").

  D0  answer only                          "The answer is \\boxed{X}." / "The answer is (X)."
  D1  non-checkable argument               written by a generator: intuition / analogy, no steps
  D2  checkable step-by-step argument      written by a generator; for a WRONG target exactly one
                                           checkable error, marked as key_step
  D3  D2 with key_step replaced by its bare claim (key_step_bare), built mechanically

Generators are models outside the pool from two families (plan: "Hai mô hình tham chiếu từ hai
họ khác nhau, nằm ngoài pool"); the generator is stored so it can enter the analysis as a random
factor. Length checks (plan: "Độ dài khớp trong ±10%") are recorded as flags, not enforced.
Every message ends the way a debating agent ends (Du et al. answer forms), so a message can be
dropped into a debate panel ("One agent solution: ```...```") unchanged in NC1.
"""

from __future__ import annotations

import json
import re

from .answers import LETTERS, canon, same
from .prompts import (PROMPT_VERSION, gen_d1_prompt, gen_d2_prompt, render_d0, render_d1,
                      render_steps)
from .utils import stable_seed

_CALC = re.compile(r"\d\s*[-+*/×x÷=]\s*\d|=")


def parse_json_obj(text: str):
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def target_display(item: dict, target: str) -> str:
    if item["format"] == "letter":
        return f"{target}) {item['options'][LETTERS.index(target)]}"
    return target


def _valid_d2(obj, item, target):
    if not obj:
        return "no JSON"
    steps, k, bare = obj.get("steps"), obj.get("key_step"), obj.get("key_step_bare")
    if not isinstance(steps, list) or not 3 <= len(steps) <= 8 or \
            not all(isinstance(s, str) and s.strip() for s in steps):
        return "steps"
    if not isinstance(k, int) or not 0 <= k < len(steps):
        return "key_step"
    if not isinstance(bare, str) or not bare.strip():
        return "key_step_bare"
    a = canon(str(obj.get("final_answer", "")), item["format"], item.get("options"))
    if not same(a, target, item["format"]):
        return f"final_answer {a!r} != {target!r}"
    return None


def _valid_d1(obj, item, target):
    if not obj or not isinstance(obj.get("argument"), str) or not obj["argument"].strip():
        return "argument"
    a = canon(str(obj.get("final_answer", "")), item["format"], item.get("options"))
    if not same(a, target, item["format"]):
        return f"final_answer {a!r} != {target!r}"
    return None


def make_messages(client, item: dict, branch: str, target: str, cfg: dict, run_id: str) -> list:
    """branch: 'correct' (target = gold) or 'wrong' (target = a wrong answer). Returns 4 records
    (D0..D3) or one record with error set."""
    fmt, disp = item["format"], target_display(item, target)
    base = dict(item_id=item["id"], task=item["task"], format=fmt, family=item.get("family"),
                branch=branch, target=target, generator=client.model_key,
                prompt_version=PROMPT_VERSION)
    d2, err, tries = None, None, 0
    for attempt in range(cfg["attempts"]):
        tries += 1
        r = client.chat(gen_d2_prompt(item, disp, branch == "correct"), n=1,
                        temperature=cfg["temperature"], max_tokens=cfg["max_tokens"],
                        seed=stable_seed(run_id, "d2", item["id"], branch, attempt), stage="gen_d2")
        obj = parse_json_obj(r.texts[0])
        err = _valid_d2(obj, item, target)
        if err is None:
            d2 = obj
            break
    if d2 is None:
        return [dict(base, msg_id=f"{item['id']}|{branch}|D2", level="D2", error=err, attempts=tries)]
    steps = [s.strip() for s in d2["steps"]]
    k = d2["key_step"]
    d3_steps = steps[:k] + [d2["key_step_bare"].strip()] + steps[k + 1:]
    t2, t3 = render_steps(steps, target, fmt), render_steps(d3_steps, target, fmt)
    n_words = len(" ".join(steps).split())

    d1, err1, tries1 = None, None, 0
    for attempt in range(cfg["attempts"]):
        tries1 += 1
        r = client.chat(gen_d1_prompt(item, disp, n_words), n=1,
                        temperature=cfg["temperature"], max_tokens=cfg["max_tokens"],
                        seed=stable_seed(run_id, "d1", item["id"], branch, attempt), stage="gen_d1")
        obj = parse_json_obj(r.texts[0])
        err1 = _valid_d1(obj, item, target)
        if err1 is None:
            d1 = obj
            break

    out = [dict(base, msg_id=f"{item['id']}|{branch}|D0", level="D0", text=render_d0(target, fmt))]
    if d1 is not None:
        t1 = render_d1(d1["argument"], target, fmt)
        out.append(dict(base, msg_id=f"{item['id']}|{branch}|D1", level="D1", text=t1,
                        attempts=tries1, flag_d1_has_calculation=bool(_CALC.search(d1["argument"])),
                        len_ratio_vs_d2=round(len(t1) / max(1, len(t2)), 3)))
    else:
        out.append(dict(base, msg_id=f"{item['id']}|{branch}|D1", level="D1", error=err1,
                        attempts=tries1))
    out.append(dict(base, msg_id=f"{item['id']}|{branch}|D2", level="D2", text=t2, steps=steps,
                    key_step=k, attempts=tries))
    ratio = len(t3) / max(1, len(t2))
    out.append(dict(base, msg_id=f"{item['id']}|{branch}|D3", level="D3", text=t3, steps=d3_steps,
                    key_step=k, replaced_step=steps[k], len_ratio_vs_d2=round(ratio, 3),
                    flag_length=abs(ratio - 1) > cfg.get("length_tolerance", 0.10)))
    return out
