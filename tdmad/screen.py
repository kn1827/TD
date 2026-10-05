"""S1 — screening and plausibility pi (plan NC1 steps 1-2, "Đo độ hợp lý").

Per (model, item):
  * k CoT samples in ONE request (vLLM `n`, prompt computed once), with exactly the round-0
    prompt and sampling of the debate (Du et al. question prompt, same temperature and
    max_tokens), so pi is the distribution of round-0 answers. k = 30 for numeric items, 20 for
    multiple choice. The first 10 are the screening sample of the plan; all k give pi
    (analysis/pi_reliability.py). Full texts are kept: NC1 uses a model's own wrong / correct
    solutions as initial states, and assigned round 0 of NC2 too.
  * multiple choice only: letter probabilities of a direct answer over `logprob_perms` cyclic
    option orders (position bias averages out; the halves 0,2 vs 1,3 give a split-half check).
"""

from __future__ import annotations

from .answers import extract_final
from .data import cyclic_perms, option_letters
from .llm import RequestRejected, letter_probs
from .prompts import PROMPT_VERSION, direct_prompt, round0_prompt
from .utils import stable_seed


def screen_item(client, item: dict, cfg: dict, run_id: str) -> dict:
    fmt, opts = item["format"], item.get("options")
    k = cfg["n_samples"][fmt]
    res = client.chat(round0_prompt(item), n=k, temperature=cfg["temperature"],
                      top_p=cfg.get("top_p", 1.0), max_tokens=cfg["max_tokens"],
                      seed=stable_seed(run_id, "screen", client.model_key, item["id"]),
                      stage="screen")
    samples = []
    for text, fin in zip(res.texts, res.finish):
        a, how = extract_final(text, fmt, opts)
        samples.append({"answer": a, "parse": how, "finish": fin, "text": text})
    rec = {"model": client.model_key, "item_id": item["id"], "task": item["task"], "format": fmt,
           "family": item.get("family"), "gold": item["answer"], "lure": item.get("lure"),
           "prompt_version": PROMPT_VERSION, "temperature": cfg["temperature"],
           "max_tokens": cfg["max_tokens"], "samples": samples,
           "tokens": {"prompt": res.prompt_tokens, "completion": res.completion_tokens}}

    if fmt == "letter" and cfg.get("logprob_perms", 0) > 0:
        letters = option_letters(item)
        lp = []
        for perm in cyclic_perms(len(opts), cfg["logprob_perms"]):
            user, prefix = direct_prompt(item, perm)
            try:
                r = client.chat(user, n=1, temperature=0.0, max_tokens=1, logprobs=20,
                                assistant_prefix=prefix, stage="logprob")
                mode = "prefill"
            except RequestRejected:  # server without continue_final_message: plain prompt
                r = client.chat(user, n=1, temperature=0.0, max_tokens=1, logprobs=20,
                                stage="logprob")
                mode = "plain"
            shown, mass = letter_probs(r.top_logprobs[0] if r.top_logprobs else {}, letters)
            # displayed letter i shows stored option perm[i] -> report in stored letters
            probs = {letters[perm[i]]: shown[letters[i]] for i in range(len(perm))}
            lp.append({"perm": perm, "probs": probs, "mass": round(mass, 4), "mode": mode})
        rec["logprob"] = lp
    return rec
