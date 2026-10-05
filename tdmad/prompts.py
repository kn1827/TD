"""Prompt templates.

The debate follows Du et al. (2023), "Improving Factuality and Reasoning in Language Models
through Multiagent Debate" (ICML 2024). Wording is copied from the authors' code,
github.com/composable-models/llm_multiagent_debate:
  numeric answers  gen_gsm.py   question prompt, construct_message(), answer as \\boxed{answer}
  multiple choice  gen_mmlu.py  question prompt, construct_message(), answer as (X)
Each agent's context is a chat: user = question prompt, assistant = its own answer, user =
"These are the solutions to the problem from other agents: ... One agent solution: ```...```",
assistant = updated answer, and so on. Other agents' solutions are anonymous (no names).

Deviations, all set by the TD-MAD v2 plan and stored in every record (mad.memory, graph):
  * an agent reads only its in-neighbours in the read graph (Du: all other agents), as in sparse
    MAD (Li et al. 2024);
  * neighbours appear in a fresh random order each turn (plan C5; Du: fixed agent order);
  * memory "last_round" (default) keeps only the latest exchange: [question prompt, own previous
    answer, newest debate message]; at round 1 this IS Du's context. memory "full" keeps the
    whole chat as in Du's code. See tdmad/mad.py for why last_round is the default.
  * yes/no questions (not in Du) reuse the MMLU wording with the answer as \\boxed{yes|no}.
Only user/assistant roles are sent (Mistral templates reject a system role; Du sends none).
Changing any template must bump PROMPT_VERSION (stored in every record).
"""

from __future__ import annotations

from .answers import LETTERS

PROMPT_VERSION = "du2023-v2"


def options_inline(item: dict, perm: list | None = None) -> str:
    """'A) bank, B) library, ...' as in gen_mmlu.py (stored order unless perm is given)."""
    opts = item["options"]
    order = list(range(len(opts))) if perm is None else perm
    return ", ".join(f"{LETTERS[k]}) {opts[i]}" for k, i in enumerate(order))


def round0_prompt(item: dict) -> str:
    """Question prompt of round 0 (also the CoT prompt of screening / pi, so pi is measured with
    exactly the policy that produces round-0 answers)."""
    q, fmt = item["question"].strip(), item["format"]
    if fmt == "number":   # gen_gsm.py
        return (f"Can you solve the following math problem? {q} Explain your reasoning. Your "
                f"final answer should be a single numerical number, in the form \\boxed{{answer}}, "
                f"at the end of your response.")
    if fmt == "letter":   # gen_mmlu.py
        return (f"Can you answer the following question as accurately as possible? {q}: "
                f"{options_inline(item)} Explain your answer, putting the answer in the form (X) "
                f"at the end of your response.")
    if fmt == "yesno":    # adaptation of gen_mmlu.py
        return (f"Can you answer the following question as accurately as possible? {q} Explain "
                f"your answer, putting the answer (yes or no) in the form \\boxed{{answer}} at the "
                f"end of your response.")
    raise ValueError(f"unknown format {fmt}")


def debate_message(item: dict, peer_texts: list) -> str:
    """construct_message() of Du et al.: the previous-round solutions of the agents read."""
    fmt = item["format"]
    if not peer_texts:    # an agent with no in-neighbour (not in the G1 graphs)
        if fmt == "letter":
            return ("Can you double check that your answer is correct. Put your final answer in "
                    "the form (X) at the end of your response.")
        return ("Can you double check that your answer is correct. Please reiterate your answer, "
                "with your final answer a single numerical number, in the form \\boxed{answer}.")
    s = "These are the solutions to the problem from other agents: "
    for t in peer_texts:
        s += "\n\n One agent solution: ```{}```".format(t)
    if fmt == "number":
        s += ("\n\n Using the reasoning from other agents as additional information, can you "
              "provide your answer to the math problem? \n The original math problem is {}. Your "
              "final answer should be a single numerical number, in the form \\boxed{{answer}}, "
              "at the end of your response.").format(item["question"].strip())
    elif fmt == "letter":
        s += ("\n\n Using the reasoning from other agents as additional advice, can you give an "
              "updated answer? Examine your solution and that other agents step by step. Put your "
              "answer in the form (X) at the end of your response.")
    else:
        s += ("\n\n Using the reasoning from other agents as additional advice, can you give an "
              "updated answer? Examine your solution and that other agents step by step. Put your "
              "answer (yes or no) in the form \\boxed{answer} at the end of your response.")
    return s


def direct_prompt(item: dict, perm: list | None) -> tuple:
    """MC plausibility by logprob: (user message, assistant prefix to continue). Same question
    wording as round 0, answer letter only."""
    q = item["question"].strip()
    return (f"Can you answer the following question as accurately as possible? {q}: "
            f"{options_inline(item, perm)} Answer with the letter of the correct option only, in "
            f"the form (X).", "The answer is (")


# ---- answer endings used by the D0-D3 messages (same form a debating agent writes) ----------

def answer_ending(target: str, fmt: str) -> str:
    return f"The answer is ({target})." if fmt == "letter" else f"The answer is \\boxed{{{target}}}."


def render_d0(target: str, fmt: str) -> str:
    return answer_ending(target, fmt)


def render_d1(argument: str, target: str, fmt: str) -> str:
    return f"{argument.strip()}\n{answer_ending(target, fmt)}"


def render_steps(steps: list, target: str, fmt: str) -> str:
    body = "\n".join(f"Step {i + 1}: {s.strip()}" for i, s in enumerate(steps))
    return f"{body}\n{answer_ending(target, fmt)}"


# ---- D1 / D2 message writers (plan: "Thao tác demonstrability trên từng tin nhắn") ----------

def question_text(item: dict) -> str:
    if item.get("options"):
        return f"{item['question'].strip()}\n" + "\n".join(
            f"{LETTERS[i]}) {o}" for i, o in enumerate(item["options"]))
    return item["question"].strip()


def gen_d2_prompt(item: dict, target_display: str, correct: bool) -> str:
    if correct:
        role = ("Write a step-by-step solution that reaches this final answer. Every step must "
                "state a concrete intermediate result that a reader can check (a calculation, a "
                "definition, a fact). Mark as key_step the single step a reader most needs in "
                "order to verify the answer.")
    else:
        role = ("This answer is WRONG; write a step-by-step solution that reaches it anyway. "
                "Exactly ONE step must contain an error that a careful reader can check and catch "
                "(e.g. a miscalculation, a misread quantity, a false fact); every other step must "
                "be correct given the steps before it. Do not hint that anything is wrong. Mark "
                "the erroneous step as key_step.")
    return (f"Problem:\n{question_text(item)}\n\nFinal answer to argue for: {target_display}\n\n"
            f"{role}\n\n"
            "Also write key_step_bare: the same step rewritten as a bare claim of its conclusion "
            "with no calculation, reason or justification, about as long as the original step "
            "(e.g. \"Clearly, the remaining amount is 42.\").\n\n"
            "Rules: 3 to 7 steps; plain sentences; no step numbers inside the strings; no "
            "comments about correctness or confidence.\n\n"
            "Output ONLY a JSON object:\n"
            '{"steps": ["...", "..."], "key_step": <0-based index>, "key_step_bare": "...", '
            f'"final_answer": "{target_display}"}}')


def gen_d1_prompt(item: dict, target_display: str, n_words: int) -> str:
    return (f"Problem:\n{question_text(item)}\n\nFinal answer to argue for: {target_display}\n\n"
            "Write a short argument for this answer that relies only on intuition, analogy, "
            "general impressions or appeal to common knowledge. It must contain NO checkable "
            "steps: no calculations, no equations, no intermediate numbers, no step-by-step "
            f"derivation. Length: about {n_words} words. Do not comment on your confidence.\n\n"
            "Output ONLY a JSON object:\n"
            f'{{"argument": "...", "final_answer": "{target_display}"}}')
