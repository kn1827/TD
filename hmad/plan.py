"""The list of debates of a run: setups x tasks x items x graphs, and who sits where.

A debate has n positions (agents); position i is played by model `models[i]`. The assignment is a
random permutation of the setup's models, seeded by (seed, setup, task, item, graph), so that every
model sits in every position about equally often over the items. Position matters in two graphs of
Choi et al.: `centralized` (position 0 is the hub) and `sparse` (position i reads i-1 and i+1);
in `decentralized` it only sets the order in which peers' responses are listed.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass

from . import choi
from .config import stable_seed


@dataclass
class Debate:
    id: str
    setup: str
    task: str
    item: int
    question: str
    gold: object
    graph: str
    models: list            # model key of each position

    @property
    def n(self) -> int:
        return len(self.models)


def build_plan(cfg: dict, registry: dict, items: dict | None = None) -> list:
    """items: {task: (questions, labels)}; loaded with the ported Choi loaders when None."""
    if items is None:
        items = {t: choi.load_task(t, spec["n_items"]) for t, spec in cfg["tasks"].items()}
    debates = []
    for setup, sc in cfg["setups"].items():
        missing = [m for m in sc["models"] if m not in registry]
        if missing:
            raise KeyError(f"setup {setup}: models not in the registry: {missing}")
        fams = [registry[m]["family"] for m in sc["models"]]
        if cfg.get("unique_families", True) and len(set(fams)) != len(fams):
            raise ValueError(f"setup {setup}: two agents of the same family: {fams}")
        for graph in sc["graphs"]:
            if graph not in choi.GRAPHS:
                raise ValueError(f"unknown graph {graph!r}")
            for task in cfg["tasks"]:
                qs, ys = items[task]
                for k, (q, y) in enumerate(zip(qs, ys)):
                    models = list(sc["models"])
                    random.Random(stable_seed(cfg.get("seed", 0), setup, task, k, graph)).shuffle(models)
                    debates.append(Debate(f"{setup}|{graph}|{task}|{k}", setup, task, k, q,
                                          _plain(y), graph, models))
    return debates


def _plain(y):
    try:
        return y.item()          # numpy scalar -> python
    except AttributeError:
        return y


def save_plan(path, debates: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(d) for d in debates], f, ensure_ascii=False, indent=0)


def load_plan(path) -> list:
    with open(path, encoding="utf-8") as f:
        return [Debate(**d) for d in json.load(f)]


def calls_per_model(debates: list, rounds: int) -> dict:
    """Generations each model has to make over the whole run."""
    out = {}
    for d in debates:
        for m in d.models:
            out[m] = out.get(m, 0) + rounds + 1
    return out
