"""Society of Mind debate as implemented by Choi, Zhu & Li (NeurIPS 2025), "Debate or Vote".

Ported from the official repository github.com/deeplearning-wisc/debate-or-vote (commit 82c929e,
MIT License, Copyright (c) the authors): src/main.py (get_new_message), src/evaluator.py
(get_instruction_suffix, evaluate_arithmetics, evaluate_mcq), src/data/{gsm8k,csqa,arithmetics}.py.
Every prompt string is copied verbatim; tests/test_choi_fidelity.py runs the original functions on
the same inputs and checks that the outputs are identical.

Deliberate differences, all outside the debate itself:
  * functions take plain arguments instead of the `args` namespace;
  * the random tie-break of the majority vote takes an explicit `rng` (the original uses the global
    `random` module) so that a resumed run gives the same result;
  * only the multi-agent branches are kept (no persona / single-agent branches), since every agent
    here is a different model family and personas are not used.

What an agent receives (original behaviour, unchanged):
  round 0      the question followed by the instruction suffix, as raw text: the original does NOT
               apply the chat template ("we find that NOT using chat template is better in MAD",
               src/model/model_utils.py)
  round t >= 1 the previous-round responses of its peers, its own previous response, the question
               again and the suffix, in one fresh message (no memory of earlier rounds)
  topologies   decentralized: every other agent; sparse: agents i-1 and i+1 (a ring);
               centralized: agent 0 reads everyone, every other agent reads only agent 0
"""

from __future__ import annotations

import collections
import re

import numpy as np

GRAPHS = ("decentralized", "sparse", "centralized")


# ---------------------------------------------------------------------------------------------
# instruction suffix (src/evaluator.py: get_instruction_suffix)
# ---------------------------------------------------------------------------------------------

def instruction_suffix(data: str, bae: bool = False, cot: bool = False) -> str:
    if data in ['arithmetics']:
        if bae:
            return ' Make sure to state your answer at the end of the response.'
        elif cot:
            return " Make sure to state your final answer in curly brackets at the very end of your response, just like: '{final answer: 12.34}'. Let's think step by step."
        else:
            return ' Make sure to state your final answer in curly brackets at the very end of your response, just like: "{final answer: 12.34}".'
    elif data in ['gsm8k']:
        if bae:
            return ' Make sure to state your answer at the end of the response.'
        elif cot:
            return " Make sure to state your final answer in curly brackets at the very end of your response, just like: '{final answer: 123}'. Let's think step by step."
        else:
            return ' Make sure to state your final answer in curly brackets at the very end of your response, just like: "{final answer: 123}".'
    elif data in ['hellaswag', 'pro_medicine', 'formal_logic', 'csqa', 'hh_rlhf']:
        if bae:
            return ' Put your final answer in the form (X) at the end of your response.'
        elif cot:
            return " Make sure to state your final answer choice in curly brackets at the very end of your response, just like: '{final answer: (A)}'. Let's think step by step."
        else:
            return ' Make sure to state your final answer choice in curly brackets at the very end of your response, just like: "{final answer: (A)}".'
    raise ValueError(f"no instruction suffix for dataset {data!r}")


def first_prompt(question: str, suffix: str) -> str:
    """Round 0 (src/main.py: messages = [{"role": "user", "content": x + SUFFIX}] * num_agents)."""
    return question + suffix


# ---------------------------------------------------------------------------------------------
# debate messages (src/main.py: get_new_message, multi-agent branches)
# ---------------------------------------------------------------------------------------------

def peers_of(i: int, n: int, graph: str) -> list:
    """Positions whose previous response agent i reads."""
    if graph == "decentralized":
        return list(range(i)) + list(range(i + 1, n))
    if graph == "sparse":
        return [(i - 1) % n, (i + 1) % n]
    if graph == "centralized":
        return list(range(1, n)) if i == 0 else [0]
    raise ValueError(f"unknown graph {graph!r} (one of {GRAPHS})")


def round_messages(question: str, responses: list, graph: str, suffix: str | None) -> list:
    """Messages of round t >= 1 for every position. `responses` = previous-round responses in
    position order. Returns one string per position, identical to the 'content' that the original
    get_new_message builds."""
    n = len(responses)
    out = []
    for i in range(n):
        if graph == "centralized" and i != 0:
            msg = f"This is the recent opinion from another agent: \n{responses[0]}\n"
            msg += f"\n\nThis was your most recent opinion:\n{responses[i]}\n"
            msg += f'\n\nUse these opinions carefully as additional advice to revise your recent opinion to give your final answer to the question:\n{question}'
        else:
            msg = "These are the recent opinions from other agents: "
            for j in peers_of(i, n, graph):
                msg += f"\n\nOne of the agents' response: \n{responses[j]}\n"
            msg += f"\n\nThis was your most recent opinion:\n{responses[i]}\n"
            msg += f'\n\nUse these opinions carefully as additional advice to revise your recent opinion to give your final answer to the question:\n{question}'
        if suffix is not None:
            msg += suffix
        out.append(msg)
    return out


# ---------------------------------------------------------------------------------------------
# answer parsing and vote (src/evaluator.py: evaluate_arithmetics, evaluate_mcq)
# ---------------------------------------------------------------------------------------------

def parse_number(response: str):
    """Last {...} in the response, 'final answer:' removed, as a float rounded to 1 decimal;
    "" if it cannot be read (as in the original)."""
    try:
        pred = re.findall(r"\{(.*?)\}", response)[-1]
        pred = float(pred.replace("final answer:", "").strip())
        return np.round(pred, 1)
    except Exception:  # noqa: BLE001 — the original catches everything
        return ""


def parse_choice(response: str):
    """Last {...}, 'final answer:' removed; "(X)" from its first (short) or second (long) char."""
    try:
        pred = re.findall(r"\{(.*?)\}", response)[-1]
        pred = pred.replace("final answer:", "").strip()
        if len(pred) == 0:
            return ""
        elif len(pred) < 3:
            return f"({pred[0]})"
        else:
            return f"({pred[1]})"
    except Exception:  # noqa: BLE001
        return ""


def vote(final_answers: list, rng):
    """Majority over readable answers; a tie is broken at random; "" if nothing is readable."""
    if len(set(final_answers)) == 1 and list(set(final_answers))[0] == "":
        return ""
    counter = collections.Counter([x for x in final_answers if x != ""])
    max_count = max(counter.values())
    most_common = [key for key, value in counter.items() if value == max_count]
    return rng.choice(most_common)


def evaluate(data: str, responses: list, answer, rng) -> tuple:
    """-> (final answer of each agent, debate answer, debate answer correct, each agent correct)."""
    if data in ['arithmetics', 'gsm8k']:
        finals = [parse_number(r) for r in responses]
        gold = np.round(answer, 1)
    elif data in ['hellaswag', 'pro_medicine', 'formal_logic', 'csqa', 'hh_rlhf']:
        finals = [parse_choice(r) for r in responses]
        gold = answer
    else:
        raise ValueError(f"no evaluator for dataset {data!r}")
    if len(set(finals)) == 1 and list(set(finals))[0] == "":
        finals = [""] * len(finals)
    debate = vote(finals, rng)
    return finals, debate, bool(debate == gold), [bool(f == gold) for f in finals]


# ---------------------------------------------------------------------------------------------
# datasets (src/data/gsm8k.py, csqa.py, arithmetics.py; the debate uses split='test')
# ---------------------------------------------------------------------------------------------

_GSM_ANS = re.compile(r"#### (\-?[0-9\.\,]+)")


def load_gsm8k(data_size: int, cache_dir=None) -> tuple:
    import pandas as pd
    from datasets import load_dataset
    dataset = pd.DataFrame(load_dataset('openai/gsm8k', 'main', cache_dir=cache_dir)['test'])
    dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True).head(data_size)
    questions, labels = [], []
    for question, answer in zip(dataset['question'], dataset['answer']):
        m = _GSM_ANS.search(answer)
        questions.append(question)
        labels.append(int(m.group(1).strip().replace(",", "").strip()) if m else None)
    return questions, labels


def load_csqa(data_size: int, cache_dir=None) -> tuple:
    """The original always takes the first 300 shuffled validation questions for split='test'
    (data_size is ignored there); here the first `data_size` of those same 300 are kept."""
    import pandas as pd
    from datasets import load_dataset
    dataset = pd.DataFrame(load_dataset('tau/commonsense_qa', cache_dir=cache_dir)['validation'])
    dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True).head(300)
    questions, labels = [], []
    template = '{}\n(A) {}\n(B) {}\n(C) {}\n(D) {}\n(E) {}\n\n'
    for ctx, choices, answer in zip(dataset['question'], dataset['choices'], dataset['answerKey']):
        options = choices['text']
        if len(options) != 5:
            continue
        questions.append(template.format(ctx, options[0], options[1], options[2], options[3], options[4]))
        labels.append(f"({answer})")
    return questions[:data_size], labels[:data_size]


def load_arithmetics(data_size: int, easy: bool = False) -> tuple:
    num_params = 4 if easy else 6
    x = np.random.default_rng(1).integers(0, 30, size=num_params * data_size)
    X, Y = [], []
    for i in range(0, num_params * data_size, num_params):
        if easy:
            a, b, c, d = x[i:i + 4]
            question = f'What is the result of {a}+{b}*{c}-{d}?'
            answer = a + b * c - d
        else:
            a, b, c, d, e, f = x[i:i + 6]
            if f == 0:
                f = 1
            question = f'What is the result of {a}+{b}*{c}+{d}-{e}÷{f}?'
            answer = a + b * c + d - e / f
        X.append(question)
        Y.append(float(answer))
    return X, Y


LOADERS = {"gsm8k": load_gsm8k, "csqa": load_csqa, "arithmetics": load_arithmetics}


def load_task(data: str, data_size: int, cache_dir=None) -> tuple:
    if data not in LOADERS:
        raise ValueError(f"dataset {data!r} not ported (available: {sorted(LOADERS)})")
    if data == "arithmetics":
        return load_arithmetics(data_size)
    return LOADERS[data](data_size, cache_dir)
