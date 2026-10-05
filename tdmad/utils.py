"""Helpers shared by every stage: jsonl io, stable seeds, resume, Kaggle deadline guard."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path


def stable_hash(*parts) -> int:
    """Hash that does not change between Python runs (unlike hash())."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:15], 16)


def stable_seed(*parts) -> int:
    return stable_hash(*parts) % (2**31 - 1)


def stable_unit(*parts) -> float:
    """Uniform number in [0, 1) fixed by the parts."""
    return stable_hash(*parts) / float(16**15)


def read_jsonl(path) -> list:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # half-written last line of a killed session
    return out


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


class JsonlWriter:
    """Append-only, thread-safe, flushed after every record: a killed Kaggle session loses at
    most the record being written, and --resume skips everything already on disk."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        needs_newline = self.path.exists() and self.path.stat().st_size > 0 and \
            not self.path.read_bytes().endswith(b"\n")
        self._f = open(self.path, "a", encoding="utf-8")
        if needs_newline:  # previous run died mid-line: do not glue the next record onto it
            self._f.write("\n")
        self._lock = threading.Lock()

    def write(self, rec: dict) -> None:
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            self._f.write(line + "\n")
            self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def done_keys(path, key: str) -> set:
    return {r[key] for r in read_jsonl(path) if key in r}


def deadline_reached(margin_s: float = 0.0) -> bool:
    """True once the wall clock passes $TDMAD_DEADLINE_TS (set by the Kaggle notebook so that
    drivers stop taking new work before the 12-hour session limit)."""
    ts = os.environ.get("TDMAD_DEADLINE_TS")
    return bool(ts) and time.time() + margin_s >= float(ts)


def run_bounded(fn, jobs, workers: int, on_result, label: str = "", every: int = 10) -> int:
    """Run fn(job) with at most `workers` jobs in flight; on_result(job, result) in the caller's
    thread. Stops taking new jobs once deadline_reached() (in-flight jobs still finish). A job
    that raises is reported and skipped; --resume picks it up next time. Returns jobs done."""
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    jobs = list(jobs)
    prog = Progress(len(jobs), label, every)
    it, inflight, n_ok, n_err = iter(jobs), {}, 0, 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        while True:
            while len(inflight) < workers and not deadline_reached():
                job = next(it, None)
                if job is None:
                    break
                inflight[ex.submit(fn, job)] = job
            if not inflight:
                break
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for f in done:
                job = inflight.pop(f)
                try:
                    res = f.result()
                except Exception as e:  # noqa: BLE001 — report and move on
                    n_err += 1
                    print(f"[{label}] job failed: {type(e).__name__}: {e}", flush=True)
                    if n_err >= 20 and n_ok == 0:
                        raise RuntimeError(f"[{label}] first 20 jobs all failed; stopping") from e
                    continue
                on_result(job, res)
                n_ok += 1
                prog.tick()
    if deadline_reached():
        print(f"[{label}] deadline reached: stopped with {len(jobs) - n_ok - n_err} jobs left "
              f"(rerun with the same command to resume)", flush=True)
    return n_ok


class Progress:
    """Prints `done/total`, rate and ETA every `every` items (thread-safe)."""

    def __init__(self, total: int, label: str, every: int = 10):
        self.total, self.label, self.every = total, label, max(1, every)
        self.done = 0
        self.t0 = time.time()
        self._lock = threading.Lock()

    def tick(self, extra: str = "") -> None:
        with self._lock:
            self.done += 1
            if self.done % self.every and self.done != self.total:
                return
            dt = time.time() - self.t0
            rate = self.done / dt if dt > 0 else 0.0
            eta = (self.total - self.done) / rate if rate > 0 else float("inf")
            print(f"[{self.label}] {self.done}/{self.total}  {rate * 60:.1f}/min  "
                  f"eta {eta / 60:.0f} min  {extra}", flush=True)
