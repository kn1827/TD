"""Exposure events and raw kernels from the S2 debates (exploratory pre-estimate for G1 — the
plan's "Ước tính G1 sơ bộ từ log MAD", now on the basic MAD instead of the DUS solver/critic).

    python -m analysis.mad_kernel --config configs/g1.yaml

For each agent j and round t -> t+1, with b = its answer at t and the answers its in-neighbours
showed at t:
  * for every other answer a held by k_a >= 1 neighbours (k_b neighbours hold b): one exposure
    event; adopt = (answer at t+1 == a). kind = plus (a correct, b wrong), minus (a wrong,
    b correct), ww (both wrong). Sampled by exposure, not by outcome (plan, R_eff bullet 3).
  * if no neighbour holds another answer: a baseline event (epsilon: leaving b anyway).
Reported per pool x task: p_adopt by kind and k_a with item-cluster bootstrap CIs, the
asymmetry Delta(1) = p+(1) - p-(1), epsilon, the signed ledger of 2609.35279 (keep / collapse /
fix / no-fix of the round-0 majority), the win curve P(final correct | q0), invalid and
truncation rates, and natural trap items (>= 2 agents share one wrong answer at round 0).
This is NOT the identified R_eff (no replay); it is the naive kernel used to check direction
and data volume.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np

from analysis.common import cluster_bootstrap, rate_ci, write_csv
from tdmad.config import Config
from tdmad.utils import read_jsonl, write_json


def events_of(rec: dict) -> tuple:
    ans, cls = rec["answers"], rec["classes"]
    nbrs = rec["graph"]["nbrs"]
    ev, base = [], []
    for t in range(len(ans) - 1):
        for j in range(len(nbrs)):
            b = ans[t][j]
            if b is None:
                continue
            held = Counter(ans[t][i] for i in nbrs[j] if ans[t][i] is not None)
            nxt = ans[t + 1][j]
            b_ok = cls[t][j] == "correct"
            alts = [(a, k) for a, k in held.items() if a != b]
            if not alts:
                base.append(dict(round=t + 1, agent=j, from_correct=b_ok, left=nxt != b,
                                 to_correct=(not b_ok) and cls[t + 1][j] == "correct"))
                continue
            a_cls = {ans[t][i]: cls[t][i] for i in nbrs[j]}
            for a, k in alts:
                a_ok = a_cls[a] == "correct"
                kind = "plus" if a_ok and not b_ok else "minus" if b_ok and not a_ok else "ww"
                ev.append(dict(round=t + 1, agent=j, kind=kind, k_a=k, k_b=held.get(b, 0),
                               degree=len(nbrs[j]), a_class=a_cls[a], b_class=cls[t][j],
                               adopt=int(nxt == a), keep=int(nxt == b)))
    return ev, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/g1.yaml")
    args = ap.parse_args()
    cfg = Config(args.config)
    recs = []
    for p in sorted((cfg.run_dir / "mad").glob("*.jsonl")):
        recs += [r for r in read_jsonl(p) if "answers" in r]   # skip *_rejected.jsonl
    if not recs:
        raise SystemExit("no debates under " + str(cfg.run_dir / "mad"))

    rows = []
    summary = {}
    cells = defaultdict(list)
    for r in recs:
        cells[(r["pool"], r["task"])].append(r)
        cells[(r["pool"], "ALL")].append(r)

    for (pool, task), rs in sorted(cells.items()):
        g = defaultdict(lambda: defaultdict(list))
        ledger, collapse_round, win = Counter(), Counter(), defaultdict(list)
        inval = trunc = turns = 0
        nat_traps = 0
        for r in rs:
            ev, base = events_of(r)
            iid = r["item_id"]  # cluster = item (shared across graphs and seeds): conservative CIs
            for e in ev:
                g[f"{e['kind']}"][iid].append(e["adopt"])
                g[f"{e['kind']}_k{min(e['k_a'], 3)}"][iid].append(e["adopt"])
                g[f"{e['kind']}_k{min(e['k_a'], 3)}_kb{min(e['k_b'], 2)}"][iid].append(e["adopt"])
                if task != "ALL":
                    rows.append(dict(pool=pool, task=task, graph=r["graph"]["name"],
                                     item_id=r["item_id"], seed=r["seed"], **e))
            for e in base:
                g["eps_from_correct" if e["from_correct"] else "eps_from_wrong"][iid].append(int(e["left"]))
            m0, mT = r["majority_class"][0], r["majority_class"][-1]
            if m0 == "correct":
                ledger["keep" if mT == "correct" else "collapse"] += 1
                if mT != "correct":
                    first = next(t for t, c in enumerate(r["majority_class"]) if c != "correct")
                    collapse_round[first] += 1
            else:
                ledger["fix" if mT == "correct" else "no_fix"] += 1
            if m0 == "undecided":
                ledger["undecided_r0"] += 1
            win[round(r["q0"], 2)].append(int(r["final_correct"]))
            wrong0 = Counter(a for a, c in zip(r["answers"][0], r["classes"][0])
                             if c in ("lure", "other"))
            nat_traps += bool(wrong0) and wrong0.most_common(1)[0][1] >= 2
            for tr in r["turns"]:
                turns += 1
                inval += tr["answer"] is None
                trunc += tr.get("finish") == "length"

        kern = {k: rate_ci(list(v.values())) for k, v in sorted(g.items())}
        plus1, minus1 = list(g.get("plus_k1", {}).values()), list(g.get("minus_k1", {}).values())

        def asym(flat_pairs):
            p = [e for kind, e in flat_pairs if kind == "p"]
            m = [e for kind, e in flat_pairs if kind == "m"]
            return sum(p) / len(p) - sum(m) / len(m)
        delta = None
        if plus1 and minus1:
            items_all = set(g["plus_k1"]) | set(g["minus_k1"])
            paired = [[("p", e) for e in g["plus_k1"].get(i, [])] + [("m", e) for e in g["minus_k1"].get(i, [])]
                      for i in items_all]
            est = kern["plus_k1"]["p"] - kern["minus_k1"]["p"]
            lo, hi = cluster_bootstrap(paired, asym)
            delta = {"est": est, "lo": lo, "hi": hi}
        acc_r0 = float(np.mean([c == "correct" for r in rs for c in r["classes"][0]]))
        acc_rT = float(np.mean([c == "correct" for r in rs for c in r["classes"][-1]]))
        summary[f"{pool}|{task}"] = dict(
            pool=pool, task=task, n_debates=len(rs),
            individual_acc_r0=acc_r0, individual_acc_final=acc_rT,
            majority_acc_r0=float(np.mean([r["round0_majority_correct"] for r in rs])),
            majority_acc_final=float(np.mean([r["final_correct"] for r in rs])),
            du_vote_acc_r0=float(np.mean([r.get("du_vote_round0_correct", False) for r in rs])),
            du_vote_acc_final=float(np.mean([r.get("du_vote_final_correct", False) for r in rs])),
            kernel=kern, delta_p1=delta, ledger=dict(ledger), collapse_round=dict(collapse_round),
            win_curve={q: {"n": len(v), "p_final_correct": float(np.mean(v))} for q, v in sorted(win.items())},
            invalid_rate=inval / max(1, turns), truncated_rate=trunc / max(1, turns),
            natural_trap_items_r0=int(nat_traps),
            tokens_per_debate=float(np.mean([r["tokens"]["prompt"] + r["tokens"]["completion"] for r in rs])),
        )
    out = cfg.run_dir / "analysis"
    write_csv(out / "exposure_events.csv", rows)
    write_json(out / "mad_summary.json", summary)
    for key, s in summary.items():
        d = s["delta_p1"]
        ds = f"{d['est']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]" if d and d["lo"] is not None else "—"
        print(f"{key}: n={s['n_debates']} acc r0 {s['individual_acc_r0']:.2f} -> "
              f"{s['individual_acc_final']:.2f}, majority {s['majority_acc_r0']:.2f} -> "
              f"{s['majority_acc_final']:.2f}, Delta p1 {ds}, ledger {s['ledger']}")
    print(f"-> {out / 'mad_summary.json'}")


if __name__ == "__main__":
    main()
