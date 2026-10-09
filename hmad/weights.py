"""Where a model's weights come from, and how much disk they may take.

Lookup order for model `key` (configs/models.yaml):
  1. a local copy in one of `model_search_paths`: a folder named after the key or after the repo
     name (e.g. ".../Qwen2.5-7B-Instruct"), directly inside a search path or one level below,
     containing config.json. Used as is (e.g. read-only dataset mounts);
  2. the download cache `model_cache_dir/<key>`, filled with huggingface_hub.snapshot_download at
     the pinned revision (safetensors, configs, tokenizer files only). HF_TOKEN from the environment
     is used for gated repos.
The cache is kept under `disk_budget_gb`: before a download, least-recently-used models that are
not protected (the one running, the one being prefetched) are deleted. 12 models in fp16 take
~190 GB, more than a typical cloud session's disk, so models are downloaded again when needed;
hmad.run orders models to reuse what is on disk and downloads the next model while one runs.
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

ALLOW = ["*.json", "*.safetensors", "*.model", "*.txt", "*.py", "*.tiktoken", "tokenizer*", "*.jinja"]
# consolidated*.safetensors: Mistral repos also ship their native-format copy of the weights (2x size)
IGNORE = ["original/*", "*.pth", "*.gguf", "*.bin", "*.pt", "consolidated*"]


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


class WeightStore:
    def __init__(self, registry: dict, cache_dir, budget_gb: float, search_paths=()):
        self.registry = registry
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.budget = budget_gb * 1e9
        self.search = [Path(p) for p in search_paths or []]
        self._threads = {}
        self._errors = {}
        self._used = {}
        self._lock = threading.Lock()

    # -- lookup ------------------------------------------------------------------------------
    def local_copy(self, key: str):
        name = self.registry[key]["hf_id"].split("/")[-1]
        for root in self.search:
            for cand in [root / key, root / name, *root.glob(f"*/{key}"), *root.glob(f"*/{name}")]:
                if (cand / "config.json").exists():
                    return cand
        return None

    def cached(self, key: str) -> bool:
        return (self.cache / key / ".complete").exists()

    def expected_bytes(self, key: str) -> float:
        return 2.0 * float(self.registry[key].get("params") or 8e9)   # fp16 weights

    # -- download ----------------------------------------------------------------------------
    def _download(self, key: str) -> None:
        from huggingface_hub import snapshot_download
        m = self.registry[key]
        dest = self.cache / key
        t0 = time.time()
        snapshot_download(repo_id=m["hf_id"], revision=m.get("revision"), local_dir=str(dest),
                          allow_patterns=ALLOW, ignore_patterns=IGNORE)
        (dest / ".complete").write_text(f"{m['hf_id']}@{m.get('revision')}\n", encoding="utf-8")
        self._used[key] = time.time()
        print(f"[weights] {key}: downloaded {dir_size(dest) / 1e9:.1f} GB in {time.time() - t0:.0f}s",
              flush=True)

    def _make_room(self, key: str, protect: set) -> None:
        need = self.expected_bytes(key)
        while True:
            entries = [p for p in self.cache.iterdir() if p.is_dir()]
            used = sum(dir_size(p) for p in entries)
            if used + need <= self.budget:
                return
            victims = sorted((p for p in entries if p.name not in protect),
                             key=lambda p: self._used.get(p.name, 0.0))
            if not victims:
                print(f"[weights] warning: {used / 1e9:.0f} GB cached + {need / 1e9:.0f} GB for {key} "
                      f"exceeds the {self.budget / 1e9:.0f} GB budget; downloading anyway", flush=True)
                return
            print(f"[weights] deleting {victims[0].name} to make room", flush=True)
            shutil.rmtree(victims[0], ignore_errors=True)

    def _fetch(self, key: str, protect: set) -> None:
        try:
            self._make_room(key, protect | {key})
            try:
                self._download(key)
            except OSError as e:
                # ENOSPC / EROFS: a cloud session's real disk quota can be far below what `df`
                # reports. Free everything not needed right now and try once more.
                print(f"[weights] {key}: {e}; deleting unprotected cached models and retrying",
                      flush=True)
                shutil.rmtree(self.cache / key, ignore_errors=True)
                for p in self.cache.iterdir():
                    if p.is_dir() and p.name not in protect:
                        shutil.rmtree(p, ignore_errors=True)
                self._download(key)
        except Exception as e:  # noqa: BLE001 — reported when the model is needed
            self._errors[key] = e

    def prefetch(self, key: str, protect: set = frozenset()) -> None:
        """Start downloading `key` in the background (no-op if local, cached or in progress)."""
        with self._lock:
            if self.local_copy(key) or self.cached(key) or key in self._threads:
                return
            t = threading.Thread(target=self._fetch, args=(key, set(protect)), daemon=True)
            self._threads[key] = t
            t.start()

    def get(self, key: str, protect: set = frozenset()) -> tuple:
        """-> (path, source). Blocks until the weights are on disk."""
        local = self.local_copy(key)
        if local:
            return str(local), f"local:{local}"
        t = self._threads.get(key)
        if t is None and not self.cached(key):
            self._fetch(key, set(protect))
        elif t is not None:
            t.join()
            self._threads.pop(key, None)
        if key in self._errors:
            raise RuntimeError(f"download of {key} failed: {self._errors.pop(key)}")
        self._used[key] = time.time()
        m = self.registry[key]
        return str(self.cache / key), f"hf:{m['hf_id']}@{m.get('revision')}"
