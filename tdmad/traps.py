"""Pilot trap set (plan: "Bộ câu bẫy", ô nghịch của thiết kế 2 x 2).

Two sources:
1. Numeric templates (CRT-style and modified classics). Each template draws fresh numbers and
   computes BOTH the correct answer and the pre-registered lure (the intuitive or memorised
   wrong answer) in code, so every key is code-verified. Fresh numbers also limit contamination:
   the memorised answer of the famous version is wrong for most variants.
2. TruthfulQA MC1 items from misconception categories (correct option + 3 human-written false
   options). The lure is not fixed here: it is the wrong option with the highest pooled pi in
   the pilot (analysis/pi_reliability.py), a rule written down before the pilot runs.

Validity (G1 criterion 1) is decided per model from measured pi: pi(lure) > pi(correct).
Human double-checking of keys (plan: "hai người gán nhãn xác nhận độc lập") still applies to the
final ~600-item set; `needs_human_check` marks what was not code-verified.
"""

from __future__ import annotations

import math
import random

from .answers import LETTERS, fmt_number
from .utils import stable_seed

ORD = {2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh"}


def _usd(cents: int) -> str:
    return f"${cents // 100}.{cents % 100:02d}"


# ---- CRT-style -------------------------------------------------------------------------------

def t_bat_ball(rng):
    a, b = rng.choice([("bat", "ball"), ("notebook", "pencil"), ("book", "bookmark"),
                       ("racket", "shuttlecock"), ("lamp", "light bulb"), ("teapot", "cup")])
    small = rng.randrange(5, 100, 5)                     # cents
    diff = rng.choice([80, 100, 120, 150, 200, 300])
    if (small, diff) == (5, 100):                         # the famous version is not a trap
        return None
    total = 2 * small + diff
    q = (f"A {a} and a {b} cost {_usd(total)} in total. The {a} costs {_usd(diff)} more than "
         f"the {b}. How much does the {b} cost, in dollars?")
    return dict(question=q, answer=small / 100, lure=(total - diff) / 100, template="bat_ball",
                family="crt")


def t_widgets(rng):
    n = rng.choice([3, 4, 5, 6, 7, 8, 10, 12])
    m = rng.choice([40, 50, 60, 80, 100, 120, 150, 200])
    thing = rng.choice(["widgets", "bottles", "toys", "bricks"])
    q = (f"If {n} machines take {n} minutes to make {n} {thing}, how many minutes would it take "
         f"{m} machines to make {m} {thing}?")
    return dict(question=q, answer=n, lure=m, template="widgets", family="crt")


def t_lily(rng):
    d = rng.choice([20, 24, 30, 36, 40, 48, 52, 60])
    part, back, div = rng.choice([("half", 1, 2), ("a quarter", 2, 4)])
    q = (f"A patch of lily pads on a lake doubles in size every day. It takes {d} days for the "
         f"patch to cover the entire lake. How many days does it take for the patch to cover "
         f"{part} of the lake?")
    return dict(question=q, answer=d - back, lure=d / div, template="lily_pads", family="crt")


def t_sheep(rng):
    n = rng.randint(12, 30)
    k = rng.randint(3, n - 3)
    if n - k == k:
        return None
    q = f"A farmer has {n} sheep. All but {k} run away. How many sheep does the farmer have left?"
    return dict(question=q, answer=k, lure=n - k, template="sheep", family="crt")


def t_race(rng):
    p = rng.choice([2, 3, 4, 5, 6, 7])
    q = (f"In a race, you overtake the runner who is in {ORD[p]} place. What place are you in "
         f"now? Answer with the place number.")
    return dict(question=q, answer=p, lure=p - 1, template="race_overtake", family="crt")


def t_round_trip(rng):
    a, b = rng.choice([(30, 60), (40, 60), (20, 30), (60, 90), (12, 24), (10, 15), (45, 90),
                       (36, 45), (20, 60), (24, 40), (15, 30), (30, 70), (40, 120), (28, 84)])
    v = rng.choice(["A cyclist rides", "A driver travels", "A runner goes"])
    q = (f"{v} from town P to town Q at {a} km/h and returns along the same road at {b} km/h. "
         f"What is the average speed for the whole round trip, in km/h?")
    return dict(question=q, answer=2 * a * b / (a + b), lure=(a + b) / 2, template="round_trip",
                family="crt")


def t_up_down(rng):
    p = rng.choice([10, 20, 25, 30, 40, 50])
    thing = rng.choice(["jacket", "share", "bicycle", "laptop"])
    first, second = rng.choice([("increased", "decreased"), ("decreased", "increased")])
    q = (f"The price of a {thing} is {first} by {p}%, and then the new price is {second} by "
         f"{p}%. By what percentage is the final price lower than the original price?")
    return dict(question=q, answer=p * p / 100, lure=0, template="percent_up_down", family="crt")


def t_discounts(rng):
    a, b = rng.choice([(20, 30), (10, 20), (25, 20), (50, 50), (40, 25), (30, 30), (10, 10),
                       (20, 20), (50, 20)])
    q = (f"A shop takes {a}% off the price of a coat, and then takes a further {b}% off the "
         f"reduced price. What is the total discount, as a percentage of the original price?")
    return dict(question=q, answer=100 - (100 - a) * (100 - b) / 100, lure=a + b,
                template="stacked_discounts", family="crt")


def t_log_cut(rng):
    t = rng.choice([6, 8, 10, 12, 15, 20])
    k = rng.choice([3, 4, 5, 6])
    q = (f"It takes {t} minutes to saw a log into 2 pieces. Sawing at the same rate, how many "
         f"minutes does it take to saw an identical log into {k} pieces?")
    return dict(question=q, answer=(k - 1) * t, lure=k * t / 2, template="log_cutting",
                family="crt")


def t_fence(rng):
    s = rng.choice([2, 3, 4, 5, 10])
    c = rng.randint(6, 20)
    q = (f"A straight fence is {s * c} meters long. There is a post at each end and a post every "
         f"{s} meters along it. How many posts are there?")
    return dict(question=q, answer=c + 1, lure=c, template="fence_posts", family="crt")


def t_clock(rng):
    n, s, m = rng.choice([(6, 5, 12), (3, 4, 6), (4, 6, 10), (6, 10, 11), (3, 6, 9), (5, 8, 9),
                          (4, 9, 7), (6, 15, 12)])
    q = (f"A clock chimes once for each hour. At {n} o'clock it chimes {n} times, and it takes "
         f"{s} seconds from the first chime to the last. At the same pace, how many seconds does "
         f"it take from the first chime to the last at {m} o'clock?")
    return dict(question=q, answer=s * (m - 1) / (n - 1), lure=s * m / n, template="clock_chimes",
                family="crt")


def t_stairs(rng):
    a, t, b = rng.choice([(3, 30, 6), (3, 40, 9), (5, 60, 9), (2, 15, 5), (4, 45, 7), (3, 20, 11)])
    q = (f"Walking up the stairs from floor 1 to floor {a} takes {t} seconds. At the same pace, "
         f"how many seconds does it take to walk from floor 1 to floor {b}?")
    return dict(question=q, answer=t * (b - 1) / (a - 1), lure=t * b / a, template="stairs",
                family="crt")


def t_pipes(rng):
    a, b = rng.choice([(3, 6), (4, 12), (6, 12), (10, 15), (12, 24), (20, 30), (2, 6), (5, 20),
                       (9, 18), (6, 30), (8, 24), (15, 30)])
    q = (f"One pipe fills a tank in {a} hours. A second pipe fills the same tank in {b} hours. "
         f"If both pipes are opened together, how many hours does it take to fill the tank?")
    return dict(question=q, answer=a * b / (a + b), lure=(a + b) / 2, template="two_pipes",
                family="crt")


def t_snail(rng):
    d, u, s = rng.choice([(10, 3, 2), (20, 5, 3), (12, 4, 2), (30, 3, 2), (25, 6, 4), (16, 4, 1),
                          (40, 5, 3), (14, 4, 2)])
    days = 1 if u >= d else math.ceil((d - u) / (u - s)) + 1
    q = (f"A snail is at the bottom of a well that is {d} meters deep. Each day it climbs up {u} "
         f"meters, and each night it slides back down {s} meters. On which day does the snail "
         f"first reach the top of the well?")
    return dict(question=q, answer=days, lure=d / (u - s), template="snail_well", family="crt")


def t_mean_of_means(rng):
    na, ma, nb, mb = rng.choice([(10, 80, 30, 60), (20, 90, 30, 70), (10, 70, 40, 50),
                                 (15, 84, 5, 64), (30, 60, 10, 100), (25, 72, 75, 88)])
    q = (f"Class A has {na} students with an average test score of {ma}. Class B has {nb} "
         f"students with an average test score of {mb}. What is the average score of all the "
         f"students in both classes together?")
    return dict(question=q, answer=(na * ma + nb * mb) / (na + nb), lure=(ma + mb) / 2,
                template="mean_of_means", family="crt")


# ---- modified classics: the memorised answer of the famous puzzle is now wrong ---------------

def t_widgets_mod(rng):
    n = rng.choice([4, 5, 6])
    m = rng.choice([50, 100, 120])
    k = rng.choice([2, 3])
    q = (f"If {n} machines take {n} minutes to make {n} widgets, how many minutes would it take "
         f"{m} machines to make {k * m} widgets?")
    return dict(question=q, answer=k * n, lure=n, template="widgets_modified",
                family="modified_classic")


def t_lily_mod(rng):
    d = rng.choice([30, 40, 48, 60])
    k = rng.choice([2, 3])
    q = (f"A patch of lily pads on a lake doubles in size every {k} days. The patch covers the "
         f"entire lake on day {d}. On which day did it cover exactly half of the lake?")
    return dict(question=q, answer=d - k, lure=d - 1, template="lily_pads_modified",
                family="modified_classic")


def t_monty_ignorant(rng):
    doors = rng.choice([3, 4])
    if doors == 3:
        q = ("You are on a game show with 3 closed doors: behind one is a car, behind the other "
             "two are goats. You pick door 1. The host does NOT know where the car is: he opens "
             "one of the other two doors at random, and it happens to reveal a goat. If you "
             "switch to the remaining closed door, what is your probability of winning the car? "
             "Give a percentage rounded to the nearest whole number.")
        return dict(question=q, answer=50, lure=67, template="monty_hall_ignorant_host",
                    family="modified_classic")
    q = ("You are on a game show with 4 closed doors: behind one is a car, behind the other three "
         "are goats. You pick door 1. The host does NOT know where the car is: he opens two of the "
         "other three doors at random, and both happen to reveal goats. If you switch to the "
         "remaining closed door, what is your probability of winning the car? Give a percentage "
         "rounded to the nearest whole number.")
    return dict(question=q, answer=50, lure=75, template="monty_hall_ignorant_host",
                family="modified_classic")


def t_two_children(rng):
    sex, which = rng.choice([("girl", "older"), ("boy", "older"), ("girl", "younger"),
                             ("boy", "younger")])
    q = (f"A family has two children. The {which} child is a {sex}. What is the probability "
         f"that both children are {sex}s? Give a percentage rounded to the nearest whole number.")
    return dict(question=q, answer=50, lure=33, template="two_children_ordered",
                family="modified_classic")


NUMERIC_TEMPLATES = [t_bat_ball, t_widgets, t_lily, t_sheep, t_race, t_round_trip, t_up_down,
                     t_discounts, t_log_cut, t_fence, t_clock, t_stairs, t_pipes, t_snail,
                     t_mean_of_means, t_widgets_mod, t_lily_mod, t_monty_ignorant,
                     t_two_children]


def numeric_traps(n: int, seed: int) -> list:
    """Round-robin over templates, skipping duplicates, until n items."""
    rng = random.Random(seed)
    seen, out, tries = set(), [], 0
    while len(out) < n and tries < 200 * n:
        tpl = NUMERIC_TEMPLATES[tries % len(NUMERIC_TEMPLATES)]  # exhausted templates just skip
        tries += 1
        it = tpl(rng)
        if it is None or it["question"] in seen:
            continue
        ans, lure = fmt_number(round(it["answer"], 4)), fmt_number(round(it["lure"], 4))
        if ans == lure:
            continue
        seen.add(it["question"])
        out.append(dict(id=f"trap-num-{len(out):03d}", task="traps_pilot", format="number",
                        question=it["question"], options=None, answer=ans, lure=lure,
                        family=it["family"], source=f"template:{it['template']}",
                        needs_human_check=False, split="pilot",
                        meta={"template": it["template"], "key": "computed in tdmad/traps.py"}))
    if len(out) < n:
        raise RuntimeError(f"only {len(out)} distinct numeric traps")
    return out


# ---- TruthfulQA misconceptions ---------------------------------------------------------------

MISCONCEPTION_CATEGORIES = ("Misconceptions", "Superstitions", "Myths and Fairytales",
                            "Misquotations", "Proverbs", "Mandela Effect")


def misconception_traps(n: int, seed: int, rows_gen: list, rows_mc: list) -> list:
    """rows_gen: truthful_qa/generation (has category); rows_mc: truthful_qa/multiple_choice."""
    cat = {r["question"].strip(): r.get("category", "") for r in rows_gen}
    cands = []
    for i, r in enumerate(rows_mc):
        q = r["question"].strip()
        c = cat.get(q, "")
        if c not in MISCONCEPTION_CATEGORIES:
            continue
        choices, labels = r["mc1_targets"]["choices"], r["mc1_targets"]["labels"]
        if 1 not in labels or len(choices) < 4:
            continue
        correct = choices[labels.index(1)]
        wrong = [ch for ch, lb in zip(choices, labels) if lb == 0][:3]
        cands.append((i, q, c, correct, wrong))
    # Misconceptions first (the core category), then the others, each in a seeded order
    rank = {c: k for k, c in enumerate(MISCONCEPTION_CATEGORIES)}
    cands.sort(key=lambda x: (rank[x[2]], stable_seed("tqa", seed, x[0])))
    out = []
    for i, q, c, correct, wrong in cands[:n]:
        opts = [correct] + wrong
        rng = random.Random(stable_seed("tqa-shuffle", seed, i))
        rng.shuffle(opts)
        out.append(dict(id=f"trap-tqa-{i:03d}", task="traps_pilot", format="letter", question=q,
                        options=opts, answer=LETTERS[opts.index(correct)], lure=None,
                        family="misconception", source=f"truthfulqa/truthful_qa:mc1:{i}",
                        needs_human_check=False, split="pilot", meta={"category": c}))
    if len(out) < n:
        raise RuntimeError(f"only {len(out)} TruthfulQA misconception items")
    return out
