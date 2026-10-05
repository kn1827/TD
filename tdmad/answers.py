"""Answer extraction and equivalence classes (plan, step 1: "chốt quy tắc trích đáp án, lớp tương đương").

Prompts ask for Du et al.'s answer forms, and parsing follows the authors' evaluation scripts:
  number  the LAST \\boxed{...} (braces matched, so \\boxed{\\frac{1}{2}} works); without one, the
          last number in the response (Du: eval_gsm.py)
  letter  the LAST "(X)" with X one of the item's option letters (Du: eval_mmlu.py); without one,
          a \\boxed{X}, "answer is X", or the text of exactly one option on the last lines
  yesno   the last \\boxed{yes|no}; without one, the last yes/no word near the end
extract_final() returns (canonical answer, how) with how = "format" (the requested form was
used), "fallback" or "none", so the share of answers outside the requested form can be tracked.
Canonical forms: numbers "18", "0.05", "-36" (commas, $, %, units dropped; a/b and \\frac
evaluated); letters "A".."R" in the stored option order; "yes" / "no".
"""

from __future__ import annotations

import re

LETTERS = "ABCDEFGHIJKLMNOPQR"

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?|-?\.\d+")
_FRAC = re.compile(r"\\[dt]?frac\{\s*(-?[\d.,]+)\s*\}\{\s*(-?[\d.,]+)\s*\}")
_LETTER_PATTERNS = [
    re.compile(r"^\W*\(?([A-R])\)?\W*$"),                     # "B", "(B)", "**B**"
    re.compile(r"^\W*\(?([A-R])[\)\.:]"),                     # "B) text", "B. text"
    re.compile(r"\b(?:OPTION|CHOICE|ANSWER)\s*(?:IS\s*)?:?\s*\(?([A-R])\)?(?![A-Z])"),
    re.compile(r"\(([A-R])\)"),
    re.compile(r"\b([A-R])\)"),
]


def fmt_number(val: float) -> str:
    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    return ("%.6f" % val).rstrip("0").rstrip(".")


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} / \\fbox{...}, nested braces included."""
    if not text:
        return None
    starts = [m.end() for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text)]
    if not starts:
        return None
    i = starts[-1]
    depth, j = 1, i
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i:j - 1] if depth == 0 else text[i:]


def canon_number(s) -> str | None:
    if s is None:
        return None
    s = str(s).replace("\u2212", "-").replace("\\!", "").replace("\\,", "").replace("{,}", "")
    b = last_boxed(s)
    if b is not None:
        s = b
    s = _FRAC.sub(r"\1/\2", s)
    s = s.replace("\\$", "").replace("$", "")
    if "=" in s:  # "3 + 4 = 7" -> "7"
        s = s.rsplit("=", 1)[1]
    m = _NUM.search(s)
    if not m:
        return None
    tok = m.group(0).replace(",", "").replace(" ", "")
    try:
        if "/" in tok:
            a, b = tok.split("/")
            val = float(a) / float(b)
        else:
            val = float(tok)
    except (ValueError, ZeroDivisionError):
        return None
    return fmt_number(val)


def canon_letter(s, options: list | None = None) -> str | None:
    if s is None:
        return None
    valid = LETTERS[: len(options)] if options else LETTERS
    raw = str(s).strip()
    up = raw.upper()
    for pat in _LETTER_PATTERNS:
        m = pat.search(up)
        if m and m.group(1) in valid:
            return m.group(1)
    if options:  # "bank" -> the option whose text matches
        low = re.sub(r"[\*\.\s]+$", "", raw.lower()).strip(" *")
        hits = [i for i, o in enumerate(options) if o.strip().lower() == low]
        if not hits:
            hits = [i for i, o in enumerate(options) if o.strip().lower() and o.strip().lower() in low]
        if len(hits) == 1:
            return valid[hits[0]]
    found = [x for x in re.findall(r"\b([A-R])\b", up) if x in valid]
    return found[-1] if found else None


def canon_yesno(s) -> str | None:
    if s is None:
        return None
    m = re.findall(r"\b(yes|no)\b", str(s).lower())
    return m[-1] if m else None


def canon(s, fmt: str, options: list | None = None) -> str | None:
    if fmt == "number":
        return canon_number(s)
    if fmt == "letter":
        return canon_letter(s, options)
    if fmt == "yesno":
        return canon_yesno(s)
    return None if s is None else str(s).strip().lower()


def extract_final(text: str, fmt: str, options: list | None = None) -> tuple:
    """-> (canonical answer or None, how) with how in {"format", "fallback", "none"}."""
    text = text or ""
    if fmt == "number":
        b = last_boxed(text)
        if b is not None and canon_number(b) is not None:
            return canon_number(b), "format"
        nums = _NUM.findall(_FRAC.sub(r"\1/\2", text.replace("\u2212", "-")))
        a = canon_number(nums[-1]) if nums else None
        return (a, "fallback") if a is not None else (None, "none")

    if fmt == "letter":
        valid = LETTERS[: len(options)] if options else LETTERS
        paren = [x for x in re.findall(r"\(([A-R])\)", text) if x in valid]
        if paren:
            return paren[-1], "format"
        b = last_boxed(text)
        if b is not None:
            a = canon_letter(b, options)
            if a:
                return a, "fallback"
        lines = [ln for ln in text.splitlines() if ln.strip()][-3:]
        for line in reversed(lines):
            m = re.search(r"answer\s*(?:is)?\s*[:\-]?\s*\**\(?([A-R])\)?(?![A-Za-z])", line, re.I)
            a = m.group(1).upper() if m and m.group(1).upper() in valid else canon_letter(line, options)
            if a:
                return a, "fallback"
        return None, "none"

    if fmt == "yesno":
        b = last_boxed(text)
        if b is not None and canon_yesno(b):
            return canon_yesno(b), "format"
        a = canon_yesno(text[-300:])
        return (a, "fallback") if a else (None, "none")
    raise ValueError(f"unknown format {fmt}")


def same(a, b, fmt: str) -> bool:
    if a is None or b is None:
        return False
    if fmt == "number":
        try:
            fa, fb = float(a), float(b)
        except ValueError:
            return a == b
        return abs(fa - fb) <= 1e-6 * max(1.0, abs(fb))
    return a == b


def answer_class(ans, item: dict, lure=None) -> str | None:
    """correct / lure / other, or None for an unreadable answer."""
    if ans is None:
        return None
    fmt = item["format"]
    if same(ans, item["answer"], fmt):
        return "correct"
    lure = item.get("lure") if lure is None else lure
    if lure is not None and same(ans, lure, fmt):
        return "lure"
    return "other"
