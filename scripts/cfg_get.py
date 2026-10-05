"""Print one config value for shell scripts.

    python -m scripts.cfg_get configs/g1.yaml lanes.lane0.model        -> qwen2.5-7b
    python -m scripts.cfg_get configs/g1.yaml lanes.lane0.model --hf   -> Qwen/Qwen2.5-7B-Instruct-AWQ
    python -m scripts.cfg_get configs/g1.yaml lanes.lane1.generators   -> phi-4 mistral-7b
    python -m scripts.cfg_get configs/g1.yaml --hf-of phi-4            -> stelterlab/phi-4-AWQ
    python -m scripts.cfg_get configs/g1.yaml --rev-of phi-4           -> 075b93fe... (pinned commit)
"""

import sys

from tdmad.config import Config, get_path


def main():
    path, key = sys.argv[1], sys.argv[2]
    cfg = Config(path)
    if key == "--hf-of":
        print(cfg.hf_id(sys.argv[3]))
        return
    if key == "--rev-of":
        print(cfg.model(sys.argv[3]).get("revision") or "")
        return
    val = get_path(cfg.raw, key)
    vals = val if isinstance(val, list) else [val]
    if "--hf" in sys.argv:
        vals = [cfg.hf_id(v) for v in vals]
    print(" ".join(str(v) for v in vals))


if __name__ == "__main__":
    main()
