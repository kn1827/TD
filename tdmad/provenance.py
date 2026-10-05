"""Provenance for the official run (plan, "Tái lập": fix and publish temperature, seeds, prompts,
model versions, and the vLLM settings that affect determinism).

check_data()      data/*.json must match the sha256 in data/MANIFEST.json (no silent edits)
ensure_protocol() the first run of a stage writes meta/<tag>_protocol.json; a later run with a
                  different protocol (prompt version, sampling, memory, model revision...) is
                  refused, so resumed sessions can never mix two protocols in one file
record_env()      appends meta/<tag>_sessions.jsonl: time, host, python, vLLM version and server
                  arguments of every endpoint used; writes meta/vllm_version.txt the first time
                  (the Kaggle notebook installs that same version in later sessions)
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import time
from pathlib import Path

import requests

from .config import ROOT
from .utils import JsonlWriter, read_json, write_json


def check_data(data_dir) -> None:
    d = Path(data_dir)
    man = d / "MANIFEST.json"
    if not man.exists():
        raise SystemExit(f"{man} missing: data/ must come from scripts.prepare_data")
    for name, info in read_json(man)["files"].items():
        p = d / name
        if not p.exists():
            raise SystemExit(f"{p} missing")
        if hashlib.sha256(p.read_bytes()).hexdigest() != info["sha256"]:
            raise SystemExit(f"{p} differs from data/MANIFEST.json: the item set changed. "
                             f"Restore the original file; never edit pilot data by hand.")


def ensure_protocol(run_dir, tag: str, protocol: dict) -> None:
    p = Path(run_dir) / "meta" / f"{tag}_protocol.json"
    proto = json.loads(json.dumps(protocol))   # normalise tuples, etc.
    if p.exists():
        old = read_json(p)
        if old != proto:
            diff = {k: (old.get(k), proto.get(k)) for k in set(old) | set(proto)
                    if old.get(k) != proto.get(k)}
            raise SystemExit(f"{tag}: protocol differs from the one already used in "
                             f"{p.parent.parent.name} (old, new): {diff}. Use a new run_name "
                             f"instead of mixing protocols in one run.")
        return
    write_json(p, proto)


def _server_info(base: str) -> dict:
    root = base.rsplit("/v1", 1)[0]
    info = {"endpoint": base}
    try:
        info["vllm_version"] = requests.get(f"{root}/version", timeout=5).json().get("version")
    except (requests.RequestException, ValueError):
        info["vllm_version"] = None
    port = root.rsplit(":", 1)[-1]
    args = ROOT / "logs" / f"vllm_{port}.args"
    if args.exists():
        info["server_args"] = args.read_text(encoding="utf-8").strip()
    return info


def record_env(run_dir, tag: str, clients: list, extra: dict | None = None) -> None:
    servers = [dict(_server_info(e), model=c.model_key, hf_id=c.hf_id)
               for c in clients for e in c.endpoints if e.startswith("http")]
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname(),
           "python": platform.python_version(), "platform": platform.platform(),
           "servers": servers, **(extra or {})}
    meta = Path(run_dir) / "meta"
    with JsonlWriter(meta / f"{tag}_sessions.jsonl") as w:
        w.write(rec)
    versions = {s["vllm_version"] for s in servers if s.get("vllm_version")}
    vfile = meta / "vllm_version.txt"
    if versions and not vfile.exists():
        vfile.write_text(sorted(versions)[0] + "\n", encoding="utf-8")


def model_fingerprint(cfg, key: str) -> dict:
    m = cfg.model(key)
    return {"hf_id": m["hf_id"], "revision": m.get("revision"),
            "chat_template_kwargs": m.get("chat_template_kwargs")}
