"""hmad/choi.py must behave exactly like the original code of Choi et al.

The original functions are read from the cloned repository (debate-or-vote/, next to hmad/) with
`ast` and executed on the same inputs; the test is skipped when the clone is absent.
    python -m pytest tests/test_choi_fidelity.py    or    python -m tests.test_choi_fidelity
"""

from __future__ import annotations

import ast
import collections
import random
import re
import types
from pathlib import Path

import numpy as np

from hmad import choi

REPO = Path(__file__).resolve().parent.parent / "debate-or-vote" / "src"


def _original(file: str, names: list) -> dict:
    tree = ast.parse((REPO / file).read_text(encoding="utf-8"))
    mod = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],
                     type_ignores=[])
    ns = {"re": re, "collections": collections, "random": random, "np": np}
    exec(compile(mod, file, "exec"), ns)
    return {n: ns[n] for n in names}


def _skip() -> bool:
    if not REPO.exists():
        print("skip: debate-or-vote/ not cloned")
        return True
    return False


def test_suffix():
    if _skip():
        return
    orig = _original("evaluator.py", ["get_instruction_suffix"])["get_instruction_suffix"]
    for data in ["arithmetics", "gsm8k", "csqa", "hellaswag", "pro_medicine", "formal_logic"]:
        for bae in (False, True):
            for cot in (False, True):
                a = types.SimpleNamespace(data=data, bae=bae, cot=cot)
                assert choi.instruction_suffix(data, bae, cot) == orig(a), (data, bae, cot)


def test_round_messages():
    if _skip():
        return
    orig = _original("main.py", ["get_new_message"])["get_new_message"]
    question = "Janet has 16 eggs. How many?\n(A) 1\n(B) 2\n"
    suffix = choi.instruction_suffix("gsm8k")
    for n in (2, 3, 6, 12):
        texts = [f"answer of agent {i} ... {{final answer: {i}}}" for i in range(n)]
        responses = {f"name__None__Agent{i + 1}": t for i, t in enumerate(texts)}
        for graph in choi.GRAPHS:
            a = types.SimpleNamespace(sparse=graph == "sparse", centralized=graph == "centralized")
            want = [m["content"] for m in orig(a, question, responses, None, suffix=suffix).values()]
            got = choi.round_messages(question, texts, graph, suffix)
            assert got == want, (n, graph)


def test_evaluate():
    if _skip():
        return
    o = _original("evaluator.py", ["evaluate_arithmetics", "evaluate_mcq"])
    cases = [
        ("gsm8k", ["... {final answer: 18}", "{final answer: 18.0}", "so {final answer: 17}"], 18),
        ("gsm8k", ["no answer", "{final answer: twelve}", "{final answer: 3.14159}"], 3.1),
        ("gsm8k", ["x", "y", "z"], 5),
        ("csqa", ["{final answer: (B)}", "{final answer: B}", "{final answer: (C) bank}"], "(B)"),
        ("csqa", ["{final answer: }", "{final answer: (A)}", "nothing"], "(A)"),
    ]
    for task, resps, gold in cases:
        fn = o["evaluate_arithmetics"] if task == "gsm8k" else o["evaluate_mcq"]
        random.seed(0)
        f_o, d_o, ok_o = fn({f"a{i}": r for i, r in enumerate(resps)}, gold)
        f_n, d_n, ok_n, _ = choi.evaluate(task, resps, gold, random.Random(0))
        assert [str(x) for x in f_o] == [str(x) for x in f_n], (task, resps)
        if len({x for x in f_o if x != ""}) <= 1 or d_o == d_n:       # no tie, or same pick
            assert str(d_o) == str(d_n) and bool(ok_o) == ok_n, (task, resps)


def test_arithmetics_loader():
    if _skip():
        return
    tree = ast.parse((REPO / "data" / "arithmetics.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_data")
    ns = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "arithmetics.py", "exec"), ns)
    X, Y = ns["load_data"](types.SimpleNamespace(data_size=20), split="test")
    X2, Y2 = choi.load_arithmetics(20)
    assert X == X2 and [float(y) for y in Y] == Y2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
