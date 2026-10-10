"""End-to-end run with fake answers (no GPU), resume, plan checks, Kaggle independence.
    python -m pytest tests/test_pipeline.py    or    python -m tests.test_pipeline
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import yaml

from hmad import choi
from hmad.config import ROOT, load_config, load_models
from hmad.plan import build_plan
from hmad.run import gpu_groups, schedule

MODELS6 = ["qwen2.5-7b", "llama3.1-8b", "mistral-7b", "olmo3-7b", "apertus-8b", "exaone3.5-7.8b"]


def _cfg(tmp: Path, **over) -> Path:
    cfg = {"extends": str(ROOT / "configs" / "exp.yaml"), "run_name": "test_fake",
           "results_dir": str(tmp), "rounds": 2, "tasks": {"arithmetics": {"n_items": 3}},
           "setups": {"main": {"models": MODELS6, "graphs": list(choi.GRAPHS)},
                      "baseline": {"models": MODELS6[:3], "graphs": ["decentralized"]}}}
    cfg.update(over)
    p = tmp / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def _run(cfg: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "hmad.run", "--config", str(cfg), "--fake"],
                          cwd=str(ROOT), capture_output=True, text=True)


def test_plan_assignment():
    cfg = load_config(ROOT / "configs" / "exp.yaml")
    reg = load_models(cfg)
    items = {t: ([""] * s["n_items"], [0] * s["n_items"]) for t, s in cfg["tasks"].items()}
    debates = build_plan(cfg, reg, items)
    assert len(debates) == 2 * 100 * 3 + 2 * 100 + 2 * 100
    for d in debates:
        fams = [reg[m]["family"] for m in d.models]
        if d.setup in ("main", "baseline"):
            assert len(set(fams)) == len(fams) == d.n      # different families
        else:
            assert d.n == 12 and len(set(fams)) == 8       # 8 families, 4 doubled
    hubs = Counter(d.models[0] for d in debates if d.graph == "centralized")
    assert len(hubs) == 6 and min(hubs.values()) >= 15    # each model is the hub about 1/6 of the time


def test_same_family_refused():
    cfg = load_config(ROOT / "configs" / "exp.yaml")
    reg = load_models(cfg)
    reg["fake-qwen"] = dict(reg["qwen2.5-7b"], key="fake-qwen")
    cfg["setups"] = {"bad": {"models": ["qwen2.5-7b", "fake-qwen"], "graphs": ["decentralized"]}}
    cfg["unique_families"] = True
    try:
        build_plan(cfg, reg, {t: ([""], [0]) for t in cfg["tasks"]})
    except ValueError:
        return
    raise AssertionError("two agents of the same family were accepted")


def test_fake_run_and_resume():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        cfg = _cfg(tmp)
        p = _run(cfg)
        assert p.returncode == 0, p.stdout + p.stderr
        rd = tmp / "test_fake"
        recs = [json.loads(line) for line in open(rd / "debates.jsonl", encoding="utf-8")]
        assert len(recs) == 3 * 3 + 3                       # main: 3 graphs x 3 items; baseline: 3
        for r in recs:
            assert len(r["rounds"]) == 3 and len(r["rounds"][0]["responses"]) == r["n"]
            assert len(set(r["families"])) == r["n"]
            if r["graph"] == "centralized":
                assert "central_correct" in r["rounds"][-1]
        # round 1 prompts contain the round-0 responses, as in Choi et al.
        jobs = [json.loads(line) for line in open(rd / "phases" / "r1" / "qwen2.5-7b.jobs.jsonl", encoding="utf-8")]
        assert all("These are the recent opinions" in j["prompt"] or "recent opinion from another" in j["prompt"]
                   for j in jobs)
        # rerun = nothing to do; phases untouched
        before = {f: f.stat().st_mtime for f in (rd / "phases").rglob("*.meta.json")}
        p2 = _run(cfg)
        assert p2.returncode == 0, p2.stdout + p2.stderr
        assert before == {f: f.stat().st_mtime for f in (rd / "phases").rglob("*.meta.json")}
        # a different protocol in the same folder is refused
        p3 = _run(_cfg(tmp, rounds=3))
        assert p3.returncode != 0 and "differs" in (p3.stdout + p3.stderr)


def test_registry_not_gated_and_schedule():
    for name, n in (("exp.yaml", 8), ("exp_slm.yaml", 12), ("smoke.yaml", 8)):
        cfg = load_config(ROOT / "configs" / name)
        reg = load_models(cfg)
        used = {m for s in cfg["setups"].values() for m in s["models"]}
        assert len(used) == n and len({reg[m]["family"] for m in used}) == n, name
        assert all(not reg[m].get("gated") for m in used), name
    assert gpu_groups({"gpus": [0, 1], "tensor_parallel_size": 1}) == [[0], [1]]
    assert gpu_groups({"gpus": [0, 1], "tensor_parallel_size": 2}) == [[0, 1]]
    steps = schedule(["a", "b", "c"], 1, 2)
    assert steps == [(0, ["a", "b"]), (0, ["c"]), (1, ["c"]), (1, ["a", "b"])]


def test_continue_on_error():
    import os
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        cfg = _cfg(tmp, engine={"continue_on_error": True})
        env = dict(os.environ, HMAD_FAKE_FAIL="mistral-7b")
        p = subprocess.run([sys.executable, "-m", "hmad.run", "--config", str(cfg), "--fake"],
                           cwd=str(ROOT), capture_output=True, text=True, env=env)
        assert p.returncode == 0, p.stdout + p.stderr
        summ = (tmp / "test_fake" / "summary.md").read_text(encoding="utf-8")
        assert "Failed phases" in summ and "mistral-7b" in summ and "fake failure" in summ
    with tempfile.TemporaryDirectory() as t:
        env = dict(os.environ, HMAD_FAKE_FAIL="mistral-7b")
        p = subprocess.run([sys.executable, "-m", "hmad.run", "--config", str(_cfg(Path(t))), "--fake"],
                           cwd=str(ROOT), capture_output=True, text=True, env=env)
        assert p.returncode != 0 and "error lines" in p.stderr


def test_core_does_not_depend_on_kaggle():
    for folder in ("hmad", "scripts"):
        for f in (ROOT / folder).rglob("*.py"):
            text = f.read_text(encoding="utf-8")
            assert "kaggle" not in text.lower(), f"{f} mentions kaggle"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
