"""Unit tests (no server, no GPU).   python -m pytest tests   or   python -m tests.test_core"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor

from tdmad import costmodel as cm
from tdmad.answers import answer_class, canon_letter, canon_number, extract_final, last_boxed
from tdmad.data import cyclic_perms, render_question
from tdmad.graphs import build, clustered_regular, girth, heawood, random_regular, triangles
from tdmad.llm import FakeClient, letter_probs
from tdmad.mad import DebateRunner, assigned_init, build_request, du_vote, majority
from tdmad.messages import make_messages, parse_json_obj
from tdmad.prompts import debate_message, direct_prompt, options_inline, round0_prompt
from tdmad.traps import NUMERIC_TEMPLATES, numeric_traps

MC = {"id": "x1", "task": "commonsenseqa", "format": "letter", "question": "Where is money kept?",
      "options": ["bank", "library", "mall", "park", "zoo"], "answer": "A", "lure": None}
NUM = {"id": "x2", "task": "gsm8k_platinum", "format": "number",
       "question": "Janet has 16 eggs, eats 3 and bakes 4. She sells the rest at $2. Income?",
       "options": None, "answer": "18", "lure": None}


def test_numbers():
    assert canon_number("$1,234.50") == "1234.5"
    assert canon_number("18 dollars") == "18"
    assert canon_number("3/4") == "0.75"
    assert canon_number("\\boxed{\\frac{1}{2}}") == "0.5"
    assert canon_number("9 × 2 = 18") == "18"
    assert canon_number("−36") == "-36"
    assert canon_number("50%") == "50"
    assert canon_number("1{,}000") == "1000"
    assert canon_number("no number") is None
    assert last_boxed("a \\boxed{1} b \\boxed{18 \\text{ dollars}} c") == "18 \\text{ dollars}"


def test_extract_du_formats():
    # numbers: last \boxed{}, else last number (Du eval_gsm.py)
    assert extract_final("They said \\boxed{20}. I get \\boxed{18}.", "number") == ("18", "format")
    assert extract_final("So she earns \\boxed{\\$18}", "number") == ("18", "format")
    assert extract_final("so it is 7 then 9 and 12", "number") == ("12", "fallback")
    assert extract_final("", "number") == (None, "none")
    # multiple choice: last (X) among the item's letters (Du eval_mmlu.py)
    o = MC["options"]
    assert extract_final("Agent said (C) but the answer is (A).", "letter", o) == ("A", "format")
    assert extract_final("Answer: (G) is not an option; (B)", "letter", o) == ("B", "format")
    assert extract_final("The answer is \\boxed{D}", "letter", o) == ("D", "fallback")
    assert extract_final("My answer is E", "letter", o) == ("E", "fallback")
    assert extract_final("Final answer: bank", "letter", o) == ("A", "fallback")
    assert extract_final("no idea", "letter", o) == (None, "none")
    assert canon_letter("B) library", o) == "B"
    assert extract_final("so \\boxed{yes}", "yesno") == ("yes", "format")


def test_classes():
    trap = dict(NUM, answer="0.05", lure="0.1")
    assert answer_class("0.05", trap) == "correct"
    assert answer_class("0.10", trap) == "lure"
    assert answer_class("1", trap) == "other"
    assert answer_class(None, trap) is None


def test_du_prompts():
    p0 = round0_prompt(NUM)
    assert p0.startswith("Can you solve the following math problem? Janet")
    assert p0.endswith("in the form \\boxed{answer}, at the end of your response.")
    m0 = round0_prompt(MC)
    assert "as accurately as possible? Where is money kept?: A) bank, B) library, C) mall" in m0
    assert m0.endswith("putting the answer in the form (X) at the end of your response.")
    d = debate_message(NUM, ["sol one", "sol two"])
    assert d.startswith("These are the solutions to the problem from other agents: "
                        "\n\n One agent solution: ```sol one```\n\n One agent solution: ```sol two```")
    assert "The original math problem is Janet has 16 eggs" in d and "\\boxed{answer}" in d
    assert "Examine your solution and that other agents step by step." in debate_message(MC, ["x"])
    assert "double check" in debate_message(NUM, [])
    user, prefix = direct_prompt(MC, [2, 3, 4, 0, 1])
    assert "A) mall, B) park" in user and prefix == "The answer is ("
    assert options_inline(MC).startswith("A) bank, B) library")


def test_render_perm():
    text, order = render_question(MC, [2, 3, 4, 0, 1])
    assert text.splitlines()[1] == "A) mall" and order[3] == 0
    assert len(cyclic_perms(5, 4)) == 4


def test_graphs():
    h = heawood()
    assert all(len(x) == 3 for x in h.nbrs) and girth(h) == 6 and triangles(h) == 0
    r = random_regular(14, 3, 3)
    assert all(len(x) == 3 for x in r.nbrs)
    c = clustered_regular(14, 3, 3, steps=5000)
    assert all(len(x) == 3 for x in c.nbrs) and triangles(c) > triangles(r)
    assert build("complete6").nbrs[0] == [1, 2, 3, 4, 5]
    assert build("ring6").nbrs[0] == [1, 5]


def test_traps():
    items = numeric_traps(60, 7)
    assert len({it["question"] for it in items}) == 60
    assert all(it["answer"] != it["lure"] for it in items)
    rng = random.Random(0)
    for tpl in NUMERIC_TEMPLATES:          # every template produces an item with distinct keys
        it = None
        while it is None:
            it = tpl(rng)
        assert abs(float(it["answer"]) - float(it["lure"])) > 1e-9
    snail = [it for it in items if it["meta"]["template"] == "snail_well"]
    assert snail and all(float(it["answer"]) == int(float(it["answer"])) for it in snail)


def test_votes():
    assert majority(["1", "1", "2"]) == "1"
    assert majority(["1", "2", None]) == "undecided"
    assert majority([None, None]) == "undecided"
    assert du_vote(["2", "1", "1", "2", None]) == "2"     # tie -> earliest agent (Du)
    assert du_vote([None]) is None


def test_letter_probs():
    p, mass = letter_probs({" A": -0.1, "B": -2.5, "The": -3.0}, "ABCD")
    assert p["A"] > 0.9 and 0.9 < mass < 1.0


class Recorder(FakeClient):
    """FakeClient that keeps every request it receives."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.requests = []

    def chat(self, user=None, *, messages=None, **kw):
        self.requests.append(messages or [{"role": "user", "content": user}])
        return super().chat(user, messages=messages, **kw)


def test_debate_memory_modes():
    for memory in ("last_round", "full"):
        fake = Recorder("m1", seed=1)
        fake.register([MC, NUM])
        pool = ThreadPoolExecutor(8)
        run = DebateRunner({"m1": fake}, build("ring6"), ["m1"] * 6, 3,
                           {"temperature": 0.7, "max_tokens": 64}, pool, "t", "p", memory)
        rec = run.run(NUM, 1, None, "d1")
        assert len(rec["answers"]) == 4 and len(rec["turns"]) == 24 and rec["memory"] == memory
        assert set(rec["turns"][6]["shown"]) == {1, 5}
        roles = [[m["role"] for m in r] for r in fake.requests]
        assert roles[:6] == [["user"]] * 6
        last = roles[-1]
        if memory == "last_round":
            assert all(r == ["user", "assistant", "user"] for r in roles[6:])
        else:
            assert last == ["user", "assistant"] * 3 + ["user"]
        # the neighbour solutions in a turn are the previous-round texts of exactly its neighbours
        t1 = rec["turns"][6]                      # agent 0, round 1
        msg = fake.requests[6][-1]["content"]
        assert msg.count("One agent solution: ```") == 2
        for i in t1["shown"]:
            assert rec["turns"][i]["text"] in msg
        pool.shutdown()


def test_assigned_and_messages():
    fake = FakeClient("m1", seed=1)
    fake.register([MC, NUM])
    pool = ThreadPoolExecutor(8)
    run = DebateRunner({"m1": fake}, build("ring6"), ["m1"] * 6, 2,
                       {"temperature": 0.7, "max_tokens": 64}, pool, "t", "p")
    init = assigned_init({"correct": ["so \\boxed{18}"], "wrong": ["so \\boxed{17}"]}, 6, 0.5,
                         random.Random(0))
    rec = run.run(NUM, 1, init, "d2")
    assert sum(c == "correct" for c in rec["classes"][0]) == 3
    msgs = make_messages(fake, MC, "wrong", "C", {"attempts": 2, "temperature": 0.7,
                                                  "max_tokens": 200}, "t")
    assert [m["level"] for m in msgs] == ["D0", "D1", "D2", "D3"]
    assert msgs[0]["text"] == "The answer is (C)."
    assert msgs[3]["text"] != msgs[2]["text"]
    assert all(extract_final(m["text"], "letter", MC["options"]) == ("C", "format") for m in msgs)
    assert parse_json_obj('```json\n{"a": 1}\n```') == {"a": 1}
    req = build_request(NUM, "last_round", [{"role": "user", "content": "q"},
                                            {"role": "assistant", "content": "a0"}], "a0", ["p"])
    assert [m["role"] for m in req] == ["user", "assistant", "user"]
    pool.shutdown()


def test_costmodel():
    m = {"params_b": 7.62, "quant": "awq", "weights_gb": 5.57, "layers": 28, "kv_heads": 4,
         "head_dim": 128}
    assert cm.kv_bytes(m) == 57344
    assert cm.max_batch(m, 2000) == 64
    assert 0 < cm.call_gpu_s(m, 1500, 230) < 10
    assert cm.debate_prompt_tokens("gsm8k_platinum", 5, 5, "full") > \
        3 * cm.debate_prompt_tokens("gsm8k_platinum", 5, 5, "last_round")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
