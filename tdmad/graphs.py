"""Read graphs G (plan: "cạnh i -> j nghĩa là j đọc i"). nbrs[j] = agents whose messages j reads.

Block A (n = 6): complete6, ring6.
Block B (n = 14, all 3-regular): heawood14 (girth 6, no triangles), rr3_14 (random 3-regular),
cluster3_14 (3-regular rewired by degree-preserving swaps to maximise triangles).
No networkx needed; every construction is deterministic given its seed.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Graph:
    name: str
    n: int
    nbrs: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name, "n": self.n, "nbrs": self.nbrs, **metrics(self)}


def _from_edges(name: str, n: int, edges) -> Graph:
    nb = [set() for _ in range(n)]
    for a, b in edges:
        nb[a].add(b)
        nb[b].add(a)
    return Graph(name, n, [sorted(s) for s in nb])


def complete(n: int) -> Graph:
    return _from_edges(f"complete{n}", n, [(i, j) for i in range(n) for j in range(i + 1, n)])


def ring(n: int) -> Graph:
    return _from_edges(f"ring{n}", n, [(i, (i + 1) % n) for i in range(n)])


def heawood() -> Graph:
    """LCF notation [5, -5]^7: 14 vertices, 3-regular, girth 6, diameter 3."""
    edges = [(i, (i + 1) % 14) for i in range(14)]
    edges += [(i, (i + 5) % 14) for i in range(0, 14, 2)]
    return _from_edges("heawood14", 14, edges)


def _edges(g: Graph) -> set:
    return {(min(i, j), max(i, j)) for i in range(g.n) for j in g.nbrs[i]}


def random_regular(n: int, d: int, seed: int, name: str | None = None) -> Graph:
    """Pairing model with rejection: uniform over simple d-regular graphs; kept only if connected."""
    rng = random.Random(seed)
    for _ in range(100000):
        pts = [v for v in range(n) for _ in range(d)]
        rng.shuffle(pts)
        edges = set()
        ok = True
        for a, b in zip(pts[::2], pts[1::2]):
            e = (min(a, b), max(a, b))
            if a == b or e in edges:
                ok = False
                break
            edges.add(e)
        if ok:
            g = _from_edges(name or f"rr{d}_{n}", n, edges)
            if is_connected(g):
                return g
    raise RuntimeError("could not draw a simple connected regular graph")


def clustered_regular(n: int, d: int, seed: int, steps: int = 20000) -> Graph:
    """Start from random_regular and apply double-edge swaps (which keep every degree) when they
    do not lower the triangle count and keep the graph connected; return the most clustered."""
    rng = random.Random(seed)
    g = random_regular(n, d, seed)
    edges = sorted(_edges(g))
    cur = triangles(g)
    best = (cur, list(edges))
    for _ in range(steps):
        (a, b), (c, e) = rng.sample(edges, 2)
        if rng.random() < 0.5:
            new1, new2 = (a, c), (b, e)
        else:
            new1, new2 = (a, e), (b, c)
        new1, new2 = (min(new1), max(new1)), (min(new2), max(new2))
        es = set(edges)
        if new1[0] == new1[1] or new2[0] == new2[1] or new1 in es or new2 in es or new1 == new2:
            continue
        es -= {(a, b), (c, e)}
        es |= {new1, new2}
        h = _from_edges("tmp", n, es)
        t = triangles(h)
        if t >= cur and is_connected(h):
            edges, cur = sorted(es), t
            if t > best[0]:
                best = (t, list(edges))
    return _from_edges(f"cluster{d}_{n}", n, best[1])


# ---- metrics ---------------------------------------------------------------------------------

def is_connected(g: Graph) -> bool:
    seen, q = {0}, deque([0])
    while q:
        v = q.popleft()
        for w in g.nbrs[v]:
            if w not in seen:
                seen.add(w)
                q.append(w)
    return len(seen) == g.n


def triangles(g: Graph) -> int:
    s = [set(x) for x in g.nbrs]
    return sum(1 for i in range(g.n) for j in g.nbrs[i] if j > i for k in s[i] & s[j] if k > j)


def girth(g: Graph) -> float:
    best = float("inf")
    for src in range(g.n):
        dist, parent, q = {src: 0}, {src: -1}, deque([src])
        while q:
            v = q.popleft()
            for w in g.nbrs[v]:
                if w not in dist:
                    dist[w], parent[w] = dist[v] + 1, v
                    q.append(w)
                elif parent[v] != w:
                    best = min(best, dist[v] + dist[w] + 1)
    return best


def diameter(g: Graph) -> int:
    out = 0
    for src in range(g.n):
        dist, q = {src: 0}, deque([src])
        while q:
            v = q.popleft()
            for w in g.nbrs[v]:
                if w not in dist:
                    dist[w] = dist[v] + 1
                    q.append(w)
        out = max(out, max(dist.values()))
    return out


def open_wedges(g: Graph) -> int:
    """c(G) in Proposition 3: pairs of neighbours sharing a listener (each listener: C(deg, 2))."""
    return sum(len(x) * (len(x) - 1) // 2 for x in g.nbrs)


def metrics(g: Graph) -> dict:
    degs = [len(x) for x in g.nbrs]
    return {"degree_min": min(degs), "degree_max": max(degs), "triangles": triangles(g),
            "girth": girth(g), "diameter": diameter(g), "wedges": open_wedges(g)}


def build(name: str, seed: int = 0) -> Graph:
    """complete6, ring6, heawood14, rr3_14, cluster3_14 (any n / d in the same patterns)."""
    if name.startswith("complete"):
        return complete(int(name[8:]))
    if name.startswith("ring"):
        return ring(int(name[4:]))
    if name == "heawood14":
        return heawood()
    if name.startswith("rr"):
        d, n = name[2:].split("_")
        return random_regular(int(n), int(d), seed)
    if name.startswith("cluster"):
        d, n = name[7:].split("_")
        return clustered_regular(int(n), int(d), seed)
    raise ValueError(f"unknown graph {name!r}")
