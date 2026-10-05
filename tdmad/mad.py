"""Multi-agent debate of Du et al. (2023) on a read graph (plan NC2 / Block A).

Protocol (wording and chat structure in tdmad/prompts.py, from the authors' code):
  round 0   every agent gets the question prompt and answers alone (or gets an assigned answer,
            see `init`); n agents = n independent samples of the same policy.
  round t   agent j gets "These are the solutions to the problem from other agents: ... One agent
            solution: ```<answer of neighbour i at round t-1>``` ..." for every in-neighbour i
            (fresh random order), and writes an updated answer. Rounds are synchronous: every
            agent of round t sees round t-1 answers only.
  result    the answers of the last round; the plan's group state is the unique plurality
            ("undecided" on a tie); Du's majority vote (ties -> first agent's answer among the
            tied, as in the authors' most_frequent()) is stored as well for comparison.

Memory (what an agent's request contains at round t >= 1):
  last_round (default)  [question prompt, own answer of round t-1, debate message of round t]
  full                  the whole chat [question prompt, a0, m1, a1, ..., m_t] as in Du's code
At round 1 both are identical. last_round is the default because (1) the plan measures the
contagion kernel in NC1 by placing an agent "in exactly the context format of a debate" —
question, own solution, a panel of neighbour messages — which matches last_round at every round
but full only at round 1 (conditions A4/A5); (2) the plan's state is (own answer, k_a, k_b) and
it keeps "the context of each turn at 3 messages" at degree 3; (3) with full memory a round-5
prompt on complete6 is ~10k tokens, about 3x the T4 cost of the whole debate stage. Running a
subset with memory: full is the natural robustness check that the choice does not drive results.

Not taken from the DUS debate, on purpose: critic role, flip guard (it forced answers back to
round 0, i.e. it suppressed the contagion being measured), round-0 self-consistency, confidence
fields, early stop.

Each record keeps what NC2 replay needs (Shapley over neighbour subsets): the graph, who was
shown to whom in which order, every text, the per-call seed, sampling, memory and
PROMPT_VERSION, so the exact request of any turn can be rebuilt (build_request()).
"""

from __future__ import annotations

import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .answers import answer_class, extract_final
from .graphs import Graph
from .prompts import PROMPT_VERSION, debate_message, round0_prompt
from .utils import stable_seed

MEMORY_MODES = ("last_round", "full")


def majority(answers: list):
    """Unique plurality among readable answers; 'undecided' on a tie or if none is readable
    (plan: "Trạng thái không có đa số duy nhất được gán 'chưa quyết'")."""
    c = Counter(a for a in answers if a is not None)
    if not c:
        return "undecided"
    top = c.most_common(2)
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "undecided"
    return top[0][0]


def du_vote(answers: list):
    """Du et al.'s most_frequent(): highest count, ties broken by the earliest agent."""
    vals = [a for a in answers if a is not None]
    if not vals:
        return None
    c = Counter(vals)   # insertion order = agent order, kept by most_common on ties
    return c.most_common(1)[0][0]


def build_request(item: dict, memory: str, history: list, own_prev: str, peer_texts: list) -> list:
    """Chat messages for one debate turn. history: the agent's full chat so far
    [question prompt, a0, m1, a1, ...]; own_prev: its answer of the previous round."""
    msg = {"role": "user", "content": debate_message(item, peer_texts)}
    if memory == "full":
        return history + [msg]
    return [history[0], {"role": "assistant", "content": own_prev}, msg]


class DebateRunner:
    def __init__(self, clients: dict, graph: Graph, agent_models: list, rounds: int,
                 sampling: dict, call_pool: ThreadPoolExecutor, run_id: str, pool_name: str,
                 memory: str = "last_round"):
        if len(agent_models) != graph.n:
            raise ValueError(f"pool has {len(agent_models)} agents, graph {graph.name} has {graph.n}")
        if memory not in MEMORY_MODES:
            raise ValueError(f"memory must be one of {MEMORY_MODES}")
        self.clients, self.graph, self.models = clients, graph, agent_models
        self.rounds, self.sampling, self.pool = rounds, sampling, call_pool
        self.run_id, self.pool_name, self.memory = run_id, pool_name, memory

    def _call(self, j: int, messages: list, seed: int, round_id: int):
        c = self.clients[self.models[j]]
        return c.chat(messages=messages, n=1, temperature=self.sampling["temperature"],
                      top_p=self.sampling.get("top_p", 1.0),
                      max_tokens=self.sampling["max_tokens"], seed=seed,
                      stage=f"mad_r{round_id}")

    def run(self, item: dict, seed: int, init: dict | None = None, debate_id: str = "") -> dict:
        fmt, opts = item["format"], item.get("options")
        n, g = self.graph.n, self.graph
        q_prompt = {"role": "user", "content": round0_prompt(item)}
        history = [[q_prompt] for _ in range(n)]   # full chat of each agent
        texts = [None] * n                         # answer of each agent in the last round
        answers, classes, turns = [], [], []
        tok_p = tok_c = 0

        def collect(t, futs, shown_by_agent, seeds, user_msgs, assigned=()):
            nonlocal tok_p, tok_c
            results = {j: f.result() for j, f in futs.items()}   # raises RequestRejected etc.
            ans_t, cls_t = [None] * n, [None] * n
            for j in range(n):
                if j in assigned:
                    text, meta = init["texts"][j], {"assigned": True}
                else:
                    r = results[j]
                    text = r.texts[0]
                    meta = {"prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                            "latency": round(r.latency, 3), "finish": r.finish[0], "seed": seeds[j]}
                    tok_p += r.prompt_tokens
                    tok_c += r.completion_tokens
                a, how = extract_final(text, fmt, opts)
                ans_t[j], cls_t[j] = a, answer_class(a, item)
                turns.append(dict(round=t, agent=j, model=self.models[j], shown=shown_by_agent[j],
                                  text=text, answer=a, parse=how, cls=cls_t[j], **meta))
                if user_msgs[j] is not None:
                    history[j].append(user_msgs[j])
                history[j].append({"role": "assistant", "content": text})
                texts[j] = text
            answers.append(ans_t)
            classes.append(cls_t)

        # ---- round 0
        assigned = set(range(n)) if init and init.get("texts") else set()
        seeds0 = [stable_seed(self.run_id, debate_id, seed, 0, j) for j in range(n)]
        futs = {j: self.pool.submit(self._call, j, [q_prompt], seeds0[j], 0)
                for j in range(n) if j not in assigned}
        collect(0, futs, [[] for _ in range(n)], seeds0, [None] * n, assigned)

        # ---- rounds 1..T
        for t in range(1, self.rounds + 1):
            seeds = [stable_seed(self.run_id, debate_id, seed, t, j) for j in range(n)]
            prev = list(texts)
            shown_by_agent, user_msgs, futs = [], [], {}
            for j in range(n):
                order = list(g.nbrs[j])
                random.Random(stable_seed("order", debate_id, seed, t, j)).shuffle(order)
                shown_by_agent.append(order)
                req = build_request(item, self.memory, history[j], prev[j], [prev[i] for i in order])
                user_msgs.append(req[-1])
                futs[j] = self.pool.submit(self._call, j, req, seeds[j], t)
            collect(t, futs, shown_by_agent, seeds, user_msgs)

        maj = [majority(a) for a in answers]
        maj_cls = [answer_class(m, item) if m != "undecided" else "undecided" for m in maj]
        du = du_vote(answers[-1])
        return {
            "debate_id": debate_id, "run_id": self.run_id, "pool": self.pool_name,
            "item_id": item["id"], "task": item["task"], "format": fmt, "family": item.get("family"),
            "gold": item["answer"], "lure": item.get("lure"),
            "graph": g.to_dict(), "models": self.models, "seed": seed, "rounds": self.rounds,
            "memory": self.memory, "sampling": self.sampling, "prompt_version": PROMPT_VERSION,
            "init": {k: v for k, v in (init or {"mode": "natural"}).items() if k != "texts"},
            "answers": answers, "classes": classes, "majority": maj, "majority_class": maj_cls,
            "q0": sum(c == "correct" for c in classes[0]) / n,
            "final_correct": maj_cls[-1] == "correct",
            "round0_majority_correct": maj_cls[0] == "correct",
            "du_vote_final": du, "du_vote_final_correct": answer_class(du, item) == "correct",
            "du_vote_round0_correct": answer_class(du_vote(answers[0]), item) == "correct",
            "tokens": {"prompt": tok_p, "completion": tok_c},
            "turns": turns,
        }


def assigned_init(samples_by_class: dict, n: int, q0: float, rng: random.Random):
    """Round 0 drawn from a model's own screening samples (same prompt and sampling as round 0)
    so that round(q0*n) agents start correct (plan: "Vòng 0 gán trước", q0 from 0.2 to 0.8).
    Wrong starts use the model's most common wrong answer when possible. None if not possible."""
    k = round(q0 * n)
    good, bad = samples_by_class.get("correct", []), samples_by_class.get("wrong_top", []) or \
        samples_by_class.get("wrong", [])
    if len(good) < 1 or len(bad) < 1:
        return None
    slots = list(range(n))
    rng.shuffle(slots)
    texts = [None] * n
    for idx, j in enumerate(slots):
        src = good if idx < k else bad
        texts[j] = src[rng.randrange(len(src))]
    return {"mode": "assigned", "q0_target": q0, "correct_agents": sorted(slots[:k]),
            "texts": texts}
