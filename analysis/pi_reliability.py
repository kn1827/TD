"""Plausibility pi from S1, its split-half reliability (G1 criterion 2), valid traps
(criterion 1) and the screening summary (accuracy, items kept, chance-level check C9).

    python -m analysis.pi_reliability --config configs/g1.yaml

pi_CoT(a)  share of CoT samples ending in a, shrunk by a beta prior fitted per
           (model, task, role) with the method of moments (binomial noise removed):
           pi = (k + alpha) / (N + alpha + beta)
pi_LP(a)   multiple choice only: direct-answer letter probability, mean over option orders
Split-half: even vs odd samples (CoT), orders {0,2} vs {1,3} (LP), on RAW shares; pooled
within-task Pearson r across items (task means removed), Spearman-Brown corrected,
item-bootstrap 95% CI; computed separately for the gold target (every item) and the lure target
(trap items: fixed numeric lure, or for TruthfulQA the wrong option with the highest pooled
pi_CoT over the pilot models). Criterion 2 uses the smaller of the two.
Valid trap (criterion 1): more of that model's CoT samples end in the lure than in the gold
answer (raw shares over the same N samples; ties are not valid). Strict 2x2 cell: lure share
>= 0.4 and gold share <= 0.15 (plan thresholds).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np

from analysis.common import items_index, pearson, spearman_brown, write_csv
from tdmad.answers import answer_class, same
from tdmad.config import Config
from tdmad.utils import read_jsonl, write_json


def beta_prior(ks: list, ns: list) -> tuple:
    ps = np.array([k / n for k, n in zip(ks, ns) if n > 0], float)
    nn = np.array([n for n in ns if n > 0], float)
    if len(ps) < 3:
        return 0.5, 0.5
    m = float(np.clip(ps.mean(), 0.005, 0.995))
    noise = float(np.mean(ps * (1 - ps) / nn))
    v = float(ps.var()) - noise
    conc = 1000.0 if v <= 1e-6 else float(np.clip(m * (1 - m) / v - 1, 1.0, 1000.0))
    return m * conc, (1 - m) * conc


def share(answers, target, fmt) -> int:
    return sum(1 for a in answers if same(a, target, fmt))


def _centered_r(a, b, g) -> float:
    """Pearson r after removing each group's (task's) mean from both halves: the pooled
    within-task correlation, so differences between task means cannot inflate it."""
    a, b = a.copy(), b.copy()
    for k in set(g):
        m = g == k
        a[m] -= a[m].mean()
        b[m] -= b[m].mean()
    return pearson(a, b)


def reliability(pairs: list, B: int = 2000, seed: int = 0) -> dict:
    """pairs: [(item_id, task, half_A, half_B)] on raw shares -> within-task r, Spearman-Brown
    and an item-bootstrap CI. Raw shares, not shrunk pi: shrinkage uses a prior fitted on all
    samples and pulls both halves toward the same mean, which inflates r."""
    pairs = [p for p in pairs if p[2] is not None and p[3] is not None]
    if len(pairs) < 5:
        return {"n_pairs": len(pairs), "r": None, "sb": None, "lo": None, "hi": None}
    a = np.array([p[2] for p in pairs], float)
    b = np.array([p[3] for p in pairs], float)
    g = np.array([p[1] for p in pairs])
    r = _centered_r(a, b, g)
    by_item = defaultdict(list)
    for k, p in enumerate(pairs):
        by_item[p[0]].append(k)
    keys = list(by_item)
    rng = np.random.default_rng(seed)
    sbs = []
    for _ in range(B):
        idx = [k for i in rng.integers(0, len(keys), len(keys)) for k in by_item[keys[i]]]
        rr = _centered_r(a[idx], b[idx], g[idx])
        if rr == rr:
            sbs.append(spearman_brown(rr))
    lo, hi = (float(np.percentile(sbs, 2.5)), float(np.percentile(sbs, 97.5))) if sbs else (None, None)
    sb = spearman_brown(r) if r == r else None
    return {"n_pairs": len(pairs), "n_items": len(keys), "r": r if r == r else None, "sb": sb,
            "lo": lo, "hi": hi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/g1.yaml")
    args = ap.parse_args()
    cfg = Config(args.config)
    tasks = list(cfg["screen"]["tasks"])
    items = items_index(cfg, tasks)
    recs = {}
    for p in sorted((cfg.run_dir / "screen").glob("*.jsonl")):
        recs[p.stem] = {r["item_id"]: r for r in read_jsonl(p) if r["item_id"] in items}
    if not recs:
        raise SystemExit("no screening output under " + str(cfg.run_dir / "screen"))
    models = sorted(recs)

    # ---- pooled lure for multiple-choice traps (rule fixed before the pilot)
    pooled_lure = {}
    for iid, it in items.items():
        if it["task"] != "traps_pilot" or it.get("lure") is not None:
            continue
        acc = Counter()
        for m in models:
            r = recs[m].get(iid)
            if r:
                n = len(r["samples"])
                for s in r["samples"]:
                    if answer_class(s["answer"], it) in ("other", "lure"):
                        acc[s["answer"]] += 1 / n
        if acc:
            pooled_lure[iid] = acc.most_common(1)[0][0]

    # ---- per (model, item, role) counts
    rows = []
    for m in models:
        for iid, r in recs[m].items():
            it = items[iid]
            fmt = it["format"]
            ans = [s["answer"] for s in r["samples"]]
            targets = {"gold": it["answer"]}
            if it["task"] == "traps_pilot":
                lure = it.get("lure") if it.get("lure") is not None else pooled_lure.get(iid)
                if lure is not None:
                    targets["lure"] = lure
            else:
                wrong = Counter(a for a in ans if answer_class(a, it) in ("other", "lure"))
                if wrong:
                    targets["top_wrong"] = wrong.most_common(1)[0][0]
            lp = r.get("logprob") or []
            for role, tgt in targets.items():
                row = dict(model=m, item_id=iid, task=it["task"], family=it.get("family"),
                           format=fmt, role=role, target=tgt, N=len(ans),
                           k=share(ans, tgt, fmt), kA=share(ans[0::2], tgt, fmt),
                           kB=share(ans[1::2], tgt, fmt), NA=len(ans[0::2]), NB=len(ans[1::2]))
                if lp:
                    vals = [x["probs"].get(tgt, 0.0) for x in lp]
                    row.update(pi_lp=float(np.mean(vals)), pi_lpA=float(np.mean(vals[0::2])),
                               pi_lpB=float(np.mean(vals[1::2])) if len(vals) > 1 else None,
                               lp_mass=float(np.mean([x["mass"] for x in lp])))
                rows.append(row)

    # ---- shrinkage per (model, task, role)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["task"], row["role"])].append(row)
    for g in groups.values():
        a, b = beta_prior([x["k"] for x in g], [x["N"] for x in g])
        for x in g:
            x["prior_a"], x["prior_b"] = round(a, 3), round(b, 3)
            x["pi"] = (x["k"] + a) / (x["N"] + a + b)
            x["piA"] = (x["kA"] + a) / (x["NA"] + a + b)
            x["piB"] = (x["kB"] + a) / (x["NB"] + a + b)

    summary = {"models": models, "pooled_lure_mc_traps": pooled_lure, "per_model": {}}
    by_mi = {(x["model"], x["item_id"], x["role"]): x for x in rows}
    trap_ids = sorted(i for i, it in items.items() if it["task"] == "traps_pilot")
    valid_by_item = Counter()
    for m in models:
        mine = [x for x in rows if x["model"] == m]

        # gold and lure are scored separately: pooling them would let the gap between the two
        # roles (gold ~ high, lure ~ low) inflate r even with no item-level signal
        def halves(role, lp=False):
            out = []
            for x in mine:
                if x["role"] != role:
                    continue
                if lp:
                    out.append((x["item_id"], x["task"], x.get("pi_lpA"), x.get("pi_lpB")))
                else:
                    out.append((x["item_id"], x["task"], x["kA"] / max(1, x["NA"]),
                                x["kB"] / max(1, x["NB"])))
            return out

        rel = {"cot_gold": reliability(halves("gold")), "cot_lure": reliability(halves("lure")),
               "lp_gold": reliability(halves("gold", lp=True)),
               "lp_lure": reliability(halves("lure", lp=True))}
        sbs = [rel[k]["sb"] for k in ("cot_gold", "cot_lure")]
        rel["primary_min"] = min(sbs) if all(v is not None for v in sbs) else None
        rel["cot_gold_by_task"] = {t: reliability([h for h in halves("gold") if h[1] == t])
                                   for t in sorted({x["task"] for x in mine})}

        valid, strict, valid_lp, checked = [], [], [], 0
        for iid in trap_ids:
            g, l = by_mi.get((m, iid, "gold")), by_mi.get((m, iid, "lure"))
            if not g or not l:
                continue
            checked += 1
            # raw shares (same N samples for both): shrinkage pulls gold and lure toward
            # different prior means and would flip close calls toward gold
            if l["k"] > g["k"]:
                valid.append(iid)
                valid_by_item[iid] += 1
                if l["k"] / l["N"] >= 0.4 and g["k"] / g["N"] <= 0.15:
                    strict.append(iid)
            if l.get("pi_lp") is not None and l["pi_lp"] > g["pi_lp"]:
                valid_lp.append(iid)
        fam = Counter(items[i]["family"] for i in valid)

        scr = {}
        for t in tasks:
            rs = [r for iid, r in recs[m].items() if items[iid]["task"] == t]
            if not rs:
                continue
            acc10, keep, inval, trunc, offfmt = [], 0, [], [], []
            for r in rs:
                it = items[r["item_id"]]
                first = r["samples"][:10]
                cls = [answer_class(s["answer"], it) for s in first]
                acc10.append(np.mean([c == "correct" for c in cls]))
                keep += any(c == "correct" for c in cls) and any(c in ("lure", "other") for c in cls)
                inval += [s["answer"] is None for s in r["samples"]]
                trunc += [s.get("finish") == "length" for s in r["samples"]]
                offfmt += [s.get("parse") != "format" for s in r["samples"]]
            nopt = [len(items[r["item_id"]]["options"]) for r in rs if items[r["item_id"]].get("options")]
            chance = 1 / np.mean(nopt) if nopt else 0.0
            acc = float(np.mean(acc10))
            scr[t] = dict(n_items=len(rs), acc_first10=acc, n_keep_mixed=int(keep),
                          invalid_rate=float(np.mean(inval)), truncated_rate=float(np.mean(trunc)),
                          off_format_rate=float(np.mean(offfmt)), chance=chance,
                          near_chance_C9=bool(abs(acc - chance) < 0.10))
        summary["per_model"][m] = dict(
            reliability=rel, traps_checked=checked, valid_traps=len(valid),
            valid_traps_by_family=dict(fam), valid_traps_strict_cell=len(strict),
            valid_traps_logprob=len(valid_lp), valid_trap_ids=valid, screening=scr)

    n_models = len(models)
    summary["traps_valid_for_half_pool"] = sum(1 for v in valid_by_item.values()
                                               if v >= max(1, n_models / 2))
    out = cfg.run_dir / "analysis"
    write_csv(out / "pi_items.csv", rows)
    write_json(out / "pi_summary.json", summary)
    for m in models:
        s = summary["per_model"][m]
        rel = s["reliability"]
        sb = "  ".join(f"{k} {rel[k]['sb']:.3f}" if rel[k]["sb"] is not None else f"{k} -"
                       for k in ("cot_gold", "cot_lure", "lp_gold"))
        print(f"{m}: valid traps {s['valid_traps']}/{s['traps_checked']} "
              f"(strict cell {s['valid_traps_strict_cell']}); split-half {sb}")
    print(f"-> {out / 'pi_summary.json'}")


if __name__ == "__main__":
    main()
