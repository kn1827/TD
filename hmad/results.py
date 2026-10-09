"""Turn the phase outputs into one record per debate (debates.jsonl) and a short summary.md.

Per round of a debate: each position's model, response text, finish reason, token counts, parsed
final answer and correctness (Choi et al.'s parser), the majority answer of all agents (random tie
break, seeded) and, for `centralized`, also the hub's answer, which is what Choi et al. score in that
graph.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict

from . import choi
from .config import stable_seed


def assemble(debates: list, outputs: dict, rounds: int, registry: dict, seed: int,
             provenance: dict) -> list:
    """outputs: job id -> worker row. Returns debate records (only fully generated debates)."""
    recs = []
    for d in debates:
        rows = []
        complete = True
        for r in range(rounds + 1):
            outs = [outputs.get(job_id(d.id, r, i)) for i in range(d.n)]
            if any(o is None for o in outs):
                complete = False
                break
            texts = [o["text"] for o in outs]
            rng = random.Random(stable_seed(seed, d.id, r))
            finals, maj, maj_ok, ok = choi.evaluate(d.task, texts, d.gold, rng)
            row = dict(round=r, responses=texts, finish=[o["finish_reason"] for o in outs],
                       prompt_tokens=[o.get("prompt_tokens") for o in outs],
                       completion_tokens=[o.get("completion_tokens") for o in outs],
                       final_answers=[_plain(f) for f in finals], agent_correct=ok,
                       majority=_plain(maj), majority_correct=maj_ok)
            if d.graph == "centralized":
                row["central"] = _plain(finals[0])
                row["central_correct"] = ok[0]
            rows.append(row)
        if not complete:
            continue
        recs.append(dict(id=d.id, setup=d.setup, task=d.task, item=d.item, graph=d.graph, n=d.n,
                         models=d.models, families=[registry[m]["family"] for m in d.models],
                         question=d.question, gold=d.gold, rounds=rows, provenance=provenance))
    return recs


def job_id(debate_id: str, r: int, pos: int) -> str:
    return f"{debate_id}#r{r}#p{pos}"


def _plain(x):
    try:
        return x.item()
    except AttributeError:
        return x


def summary(recs: list, rounds: int) -> str:
    lines = ["# Summary", "", f"{len(recs)} complete debates.", "",
             "## Majority accuracy by round", "",
             "| setup | graph | task | n debates | " + " | ".join(f"r{r}" for r in range(rounds + 1)) + " |",
             "|---|---|---|---|" + "---|" * (rounds + 1)]
    groups = defaultdict(list)
    for d in recs:
        groups[(d["setup"], d["graph"], d["task"])].append(d)
    for (s, g, t), ds in sorted(groups.items()):
        accs = [sum(d["rounds"][r]["majority_correct"] for d in ds) / len(ds) for r in range(rounds + 1)]
        lines.append(f"| {s} | {g} | {t} | {len(ds)} | " + " | ".join(f"{a:.2f}" for a in accs) + " |")
    per = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))   # model -> task -> [n, readable, correct r0, truncated]
    for d in recs:
        for r_ in d["rounds"]:
            for i, m in enumerate(d["models"]):
                c = per[m][d["task"]]
                c[0] += 1
                c[1] += r_["final_answers"][i] != ""
                c[2] += bool(r_["agent_correct"][i]) if r_["round"] == 0 else 0
                c[3] += r_["finish"][i] == "length"
    lines += ["", "## Per model (all rounds; accuracy = round 0, i.e. alone)", "",
              "| model | task | generations | readable answer | hit 512-token limit | round-0 accuracy |",
              "|---|---|---|---|---|---|"]
    for m in sorted(per):
        for t, (n, ok, cor, trunc) in sorted(per[m].items()):
            n0 = n / (rounds + 1)
            lines.append(f"| {m} | {t} | {n} | {ok / n:.0%} | {trunc / n:.0%} | {cor / max(n0, 1):.0%} |")
    return "\n".join(lines) + "\n"


def write(path, recs: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
