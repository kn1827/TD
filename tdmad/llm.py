"""Client for OpenAI-compatible servers (vLLM on Kaggle), with per-call cost logging.

Adapted from DUS src/agents/base_agent.py (_call_vllm, check_vllm_server), plus what TD-MAD
needs: multi-turn chat contexts (Du et al. debate), n samples per request, per-request seeds,
first-token logprobs, assistant prefill, fixed chat-template variables, round-robin over
replicas, and a cost record for every call (G1 criterion 4).

Errors: HTTP 400/413/422 (prompt longer than --max-model-len, invalid parameters) raise
RequestRejected and are not retried; network errors, 5xx and malformed responses are retried
with back-off, then raise RuntimeError.

FakeClient answers without a server, so every stage and analysis can be dry-run on a laptop.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import requests

from .answers import LETTERS, extract_final, fmt_number
from .utils import JsonlWriter, stable_seed


class RequestRejected(Exception):
    """The server refused the request (4xx); retrying the same request cannot succeed."""


@dataclass
class ChatResult:
    texts: list
    finish: list
    prompt_tokens: int
    completion_tokens: int
    latency: float
    endpoint: str
    top_logprobs: list = field(default_factory=list)  # per choice: {token: logprob} of token 1


class CostLog:
    """calls.jsonl: one line per request. stages.jsonl: one line per stage with wall time and the
    number of GPUs the stage held, so GPU-seconds per call = wall * gpus / calls."""

    def __init__(self, run_dir, tag: str):
        self.calls = JsonlWriter(f"{run_dir}/cost/{tag}_calls.jsonl")
        self.stages = JsonlWriter(f"{run_dir}/cost/{tag}_stages.jsonl")
        self._lock = threading.Lock()
        self._acc = None
        self._count_prefix = ""

    def record(self, **kw) -> None:
        kw["ts"] = time.time()
        self.calls.write(kw)
        with self._lock:
            if self._acc is not None:
                self._acc["requests"] += 1
                if str(kw.get("stage", "")).startswith(self._count_prefix):
                    self._acc["samples"] += kw.get("n", 1)
                self._acc["prompt_tokens"] += kw.get("prompt_tokens", 0)
                self._acc["completion_tokens"] += kw.get("completion_tokens", 0)

    @contextmanager
    def stage(self, stage: str, model: str, gpus: int, unit: str, count_prefix: str = "", **extra):
        """unit: what one "call" is for criterion 4 ("sample" or "turn"); only calls whose stage
        label starts with count_prefix are counted as such (e.g. not the logprob probes)."""
        with self._lock:
            self._count_prefix = count_prefix
            self._acc = dict(requests=0, samples=0, prompt_tokens=0, completion_tokens=0)
        t0 = time.time()
        try:
            yield self
        finally:
            with self._lock:
                acc, self._acc = self._acc, None
            self.stages.write(dict(stage=stage, model=model, gpus=gpus, unit=unit,
                                   wall_s=round(time.time() - t0, 2), t_start=t0, **acc, **extra))


def _messages(user, messages, assistant_prefix) -> list:
    msgs = [dict(m) for m in messages] if messages else [{"role": "user", "content": user}]
    if assistant_prefix:
        msgs.append({"role": "assistant", "content": assistant_prefix})
    return msgs


class LLMClient:
    def __init__(self, model_key: str, hf_id: str, endpoints: list, cost: CostLog | None = None,
                 timeout: float = 900, max_retries: int = 6, template_kwargs: dict | None = None):
        self.model_key, self.hf_id = model_key, hf_id
        self.endpoints = [e.rstrip("/") for e in endpoints]
        self._rr = itertools.cycle(range(len(self.endpoints)))
        self._rr_lock = threading.Lock()
        self.cost, self.timeout, self.max_retries = cost, timeout, max_retries
        self.template_kwargs = dict(template_kwargs or {})
        self._local = threading.local()
        self.headers = {"Content-Type": "application/json"}
        if os.environ.get("VLLM_API_KEY"):
            self.headers["Authorization"] = f"Bearer {os.environ['VLLM_API_KEY']}"

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "s"):
            self._local.s = requests.Session()
        return self._local.s

    def _next_endpoint(self) -> str:
        with self._rr_lock:
            return self.endpoints[next(self._rr)]

    def check(self) -> None:
        """Fail fast if a server is down or serves another model (DUS check_vllm_server)."""
        for base in self.endpoints:
            try:
                r = requests.get(f"{base}/models", timeout=10, headers=self.headers)
                r.raise_for_status()
                served = [m["id"] for m in r.json().get("data", [])]
            except (requests.RequestException, ValueError, KeyError) as e:
                raise RuntimeError(f"no server at {base} for {self.model_key} ({e}); start it with "
                                   f"kaggle/start_vllm.sh and see logs/vllm_<port>.log") from e
            if self.hf_id not in served:
                raise RuntimeError(f"{base} serves {served}, not {self.hf_id}")

    def chat(self, user: str | None = None, *, messages: list | None = None, n: int = 1,
             temperature: float = 0.7, top_p: float = 1.0, max_tokens: int = 1024,
             seed: int | None = None, logprobs: int = 0, assistant_prefix: str | None = None,
             stage: str = "", extra: dict | None = None) -> ChatResult:
        """user: one user message; or messages: a full chat (user/assistant alternating)."""
        body = {"model": self.hf_id, "messages": _messages(user, messages, assistant_prefix),
                "n": n, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
        if seed is not None:
            body["seed"] = int(seed)
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = int(logprobs)
        if assistant_prefix:
            body["add_generation_prompt"] = False
            body["continue_final_message"] = True
        if self.template_kwargs:
            body["chat_template_kwargs"] = self.template_kwargs
        if extra:
            body.update(extra)
        last = None
        for attempt in range(self.max_retries):
            base = self._next_endpoint()
            t0 = time.time()
            try:
                r = self._session().post(f"{base}/chat/completions", json=body,
                                         headers=self.headers, timeout=(10, self.timeout))
                if r.status_code in (400, 413, 422):
                    raise RequestRejected(f"{self.model_key}: HTTP {r.status_code}: {r.text[:400]}")
                r.raise_for_status()
                data = r.json()
                dt = time.time() - t0
                choices = sorted(data["choices"], key=lambda c: c.get("index", 0))
                usage = data.get("usage") or {}
                res = ChatResult(
                    texts=[(c.get("message") or {}).get("content") or "" for c in choices],
                    finish=[c.get("finish_reason") for c in choices],
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    latency=dt, endpoint=base,
                    top_logprobs=[_first_token_logprobs(c) for c in choices] if logprobs else [])
                if self.cost:
                    self.cost.record(model=self.model_key, stage=stage, endpoint=base, n=n,
                                     prompt_tokens=res.prompt_tokens,
                                     completion_tokens=res.completion_tokens,
                                     latency=round(dt, 3), max_tokens=max_tokens)
                return res
            except RequestRejected:
                raise
            except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as e:
                last = e   # ValueError covers a truncated / non-JSON body
                wait = min(60, 5 * (attempt + 1))
                print(f"[{self.model_key}] request failed ({type(e).__name__}: {e}); retry in "
                      f"{wait}s ({attempt + 1}/{self.max_retries})", flush=True)
                time.sleep(wait)
        raise RuntimeError(f"{self.model_key}: failed after {self.max_retries} retries: {last}")


def _first_token_logprobs(choice: dict) -> dict:
    content = ((choice.get("logprobs") or {}).get("content") or [])
    if not content:
        return {}
    return {t["token"]: t["logprob"] for t in content[0].get("top_logprobs") or []}


def letter_probs(top: dict, letters: str) -> tuple:
    """{token: logprob} of the first answer token -> ({letter: p} renormalised, raw letter mass)."""
    acc = {L: 0.0 for L in letters}
    for tok, lp in top.items():
        t = tok.strip().strip("()[]*.:").upper()
        if t in acc:
            acc[t] += math.exp(lp)
    mass = sum(acc.values())
    if mass <= 0:
        return {L: 1.0 / len(letters) for L in letters}, 0.0
    return {L: v / mass for L, v in acc.items()}, mass


# ---------------------------------------------------------------------------------------------
# FakeClient: no server, plausible behaviour, for dry runs and tests
# ---------------------------------------------------------------------------------------------

_PEER = re.compile(r"One agent solution: ```(.*?)```", re.S)


class FakeClient:
    """Imitates a debater. Knows the items registered with `register()` and answers correct /
    lure / other with fixed odds; in a debate turn it copies the most common neighbour answer
    with probability p_follow, else keeps its own previous answer with probability p_stay.
    Token counts are len/4; latency is 0."""

    def __init__(self, model_key: str, cost: CostLog | None = None, p_correct=0.6, p_lure=0.25,
                 p_follow=0.3, p_stay=0.7, seed=0):
        self.model_key, self.hf_id, self.endpoints = model_key, f"fake/{model_key}", ["fake"]
        self.cost = cost
        self.p_correct, self.p_lure = p_correct, p_lure
        self.p_follow, self.p_stay, self.seed = p_follow, p_stay, seed
        self.items = {}

    def register(self, items: list) -> None:
        for it in items:
            self.items[it["question"].strip()[:200]] = it

    def check(self) -> None:
        return None

    def _item_for(self, text: str):
        for stem, it in self.items.items():
            if stem in text:
                return it
        return None

    def _pick(self, rng, it, fmt):
        if it is None:
            return "A" if fmt == "letter" else "0"
        if fmt == "letter":
            letters = LETTERS[: len(it["options"])]
            others = [L for L in letters if L != it["answer"]]
            correct, lure = it["answer"], others[0]
        else:
            correct = it["answer"]
            lure = it.get("lure") or fmt_number(float(correct) + 1)
            others = [fmt_number(float(correct) + d) for d in (2, 3, -1, 10)]
        u = rng.random()
        if u < self.p_correct:
            return correct
        if u < self.p_correct + self.p_lure:
            return lure
        return rng.choice(others)

    def chat(self, user=None, *, messages=None, n: int = 1, temperature: float = 0.7,
             top_p: float = 1.0, max_tokens: int = 1024, seed=None, logprobs: int = 0,
             assistant_prefix=None, stage: str = "", extra=None) -> ChatResult:
        msgs = _messages(user, messages, None)
        first, last = msgs[0]["content"], [m for m in msgs if m["role"] == "user"][-1]["content"]
        rng = random.Random(stable_seed(self.seed, self.model_key, seed, last[-300:], len(msgs)))
        it = self._item_for(first)
        fmt = it["format"] if it else ("letter" if "(X)" in first else "number")
        opts = it.get("options") if it else None
        texts, tops = [], []
        if "Output ONLY a JSON object" in last:  # D1 / D2 writer
            target = re.search(r"Final answer to argue for: (.+)", last).group(1).strip()
            for _ in range(n):
                if '"steps"' in last:
                    steps = [f"From the problem, quantity {k} is {rng.randint(2, 50)}." for k in
                             range(rng.randint(3, 5))]
                    texts.append(json.dumps({"steps": steps, "key_step": 1,
                                             "key_step_bare": "Clearly, quantity 1 is as stated.",
                                             "final_answer": target}))
                else:
                    texts.append(json.dumps({"argument": "It just seems the most natural reading "
                                             "of the situation.", "final_answer": target}))
        elif logprobs:
            letters = LETTERS[: len(opts)] if opts else "ABCD"
            for _ in range(n):
                w = [rng.random() ** 2 for _ in letters]
                texts.append(letters[w.index(max(w))])
                tops.append({L: math.log(x / sum(w) + 1e-9) for L, x in zip(letters, w)})
        else:
            peers = [extract_final(p, fmt, opts)[0] for p in _PEER.findall(last)]
            peers = [p for p in peers if p is not None]
            own = [m["content"] for m in msgs if m["role"] == "assistant"]
            own_ans = extract_final(own[-1], fmt, opts)[0] if own else None
            for _ in range(n):
                if peers and rng.random() < self.p_follow:
                    ans = max(set(peers), key=peers.count)
                elif own_ans is not None and rng.random() < self.p_stay:
                    ans = own_ans
                else:
                    ans = self._pick(rng, it, fmt)
                end = f"({ans})" if fmt == "letter" else f"\\boxed{{{ans}}}"
                texts.append(f"Let me work through it step by step.\nFirst, read the problem.\n"
                             f"Then compute.\nSo the answer is {end}.")
        res = ChatResult(texts=texts, finish=["stop"] * n,
                         prompt_tokens=sum(len(m["content"]) for m in msgs) // 4,
                         completion_tokens=sum(len(t) // 4 for t in texts), latency=0.0,
                         endpoint="fake", top_logprobs=tops)
        if self.cost:
            self.cost.record(model=self.model_key, stage=stage, endpoint="fake", n=n,
                             prompt_tokens=res.prompt_tokens,
                             completion_tokens=res.completion_tokens, latency=0.0,
                             max_tokens=max_tokens)
        return res


def make_client(cfg, model_key: str, cost: CostLog | None, fake: bool = False,
                endpoint: str | None = None):
    if fake:
        return FakeClient(model_key, cost, seed=stable_seed("fake", model_key))
    eps = [endpoint] if endpoint else cfg.endpoints(model_key)
    m = cfg.model(model_key)
    return LLMClient(model_key, m["hf_id"], eps, cost,
                     template_kwargs=m.get("chat_template_kwargs"))
