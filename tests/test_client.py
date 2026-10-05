"""LLMClient against a local mock of the vLLM OpenAI API (request body, chat contexts, n,
logprobs, prefill, template kwargs, retries, rejections, cost log).   python -m tests.test_client"""

from __future__ import annotations

import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import tdmad.llm as llm
from tdmad.llm import CostLog, LLMClient, RequestRejected, letter_probs
from tdmad.utils import read_jsonl

SEEN = []
SCRIPT = {"fail": 1, "garbage": 0}   # first request -> 503; then optionally a non-JSON 200


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj, raw: bytes | None = None):
        body = raw if raw is not None else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"data": [{"id": "org/model-awq"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(req)
        if SCRIPT["fail"] > 0:
            SCRIPT["fail"] -= 1
            return self._send(503, {"error": "warming up"})
        if SCRIPT["garbage"] > 0:
            SCRIPT["garbage"] -= 1
            return self._send(200, None, raw=b'{"choices": [')
        if req["messages"][0]["content"] == "too long":
            return self._send(400, {"error": "maximum context length is 10240 tokens"})
        choices = []
        for i in range(req.get("n", 1)):
            c = {"index": i, "message": {"role": "assistant", "content": f"So \\boxed{{{i}}}"},
                 "finish_reason": "stop"}
            if req.get("logprobs"):
                c["logprobs"] = {"content": [{"token": "B", "logprob": -0.2, "top_logprobs": [
                    {"token": "B", "logprob": -0.2}, {"token": " A", "logprob": -1.9},
                    {"token": "**", "logprob": -4.0}]}]}
            choices.append(c)
        self._send(200, {"choices": choices[::-1],   # out of order on purpose
                         "usage": {"prompt_tokens": 11, "completion_tokens": 5 * len(choices)}})


def test_client():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    sleep = llm.time.sleep
    llm.time.sleep = lambda s: None          # no back-off wait in the test
    try:
        with tempfile.TemporaryDirectory() as d:
            cost = CostLog(d, "t")
            c = LLMClient("m", "org/model-awq", [url], cost, max_retries=3,
                          template_kwargs={"date_string": "26 Jul 2024"})
            c.check()
            with cost.stage("screen", "m", 1, "sample", count_prefix="screen"):
                r = c.chat("hi", n=3, seed=5, stage="screen")          # after one 503
                lp = c.chat("q", max_tokens=1, logprobs=20, assistant_prefix="The answer is (",
                            stage="logprob")
            assert r.texts == ["So \\boxed{0}", "So \\boxed{1}", "So \\boxed{2}"]
            assert r.prompt_tokens == 11 and r.completion_tokens == 15
            assert SEEN[-2]["seed"] == 5 and SEEN[-2]["n"] == 3
            assert SEEN[-2]["chat_template_kwargs"] == {"date_string": "26 Jul 2024"}
            body = SEEN[-1]
            assert body["continue_final_message"] and not body["add_generation_prompt"]
            assert body["messages"][-1] == {"role": "assistant", "content": "The answer is ("}
            p, mass = letter_probs(lp.top_logprobs[0], "ABCD")
            assert p["B"] > 0.8 and mass > 0.9

            # multi-turn debate context goes through unchanged
            ctx = [{"role": "user", "content": "q0"}, {"role": "assistant", "content": "a0"},
                   {"role": "user", "content": "These are the solutions ..."}]
            c.chat(messages=ctx, stage="mad_r1")
            assert SEEN[-1]["messages"] == ctx

            # a truncated JSON body is retried (it is a ValueError, not a rejection)
            SCRIPT["garbage"] = 1
            n_before = len(SEEN)
            c.chat("again", stage="screen")
            assert len(SEEN) == n_before + 2

            # 400 is permanent: RequestRejected, no retry
            n_before = len(SEEN)
            try:
                c.chat("too long")
                raise AssertionError("400 must raise RequestRejected")
            except RequestRejected:
                pass
            assert len(SEEN) == n_before + 1

            stages = read_jsonl(f"{d}/cost/t_stages.jsonl")
            assert stages[0]["samples"] == 3 and stages[0]["requests"] == 2  # logprob not a sample
            cost.calls.close()
            cost.stages.close()
    finally:
        llm.time.sleep = sleep
        srv.shutdown()


if __name__ == "__main__":
    test_client()
    print("ok test_client")
