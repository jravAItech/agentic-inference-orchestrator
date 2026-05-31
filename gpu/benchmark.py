"""
Phase 2 benchmark. Drives the Triton endpoint at increasing concurrency and
records throughput + latency percentiles. Run on Colab/Kaggle after the
engine is built and Triton is serving. Output feeds the README table.

    python gpu/benchmark.py --url localhost:8000 --concurrency 1,4,8 \
        --out benchmarks/results.json
"""

from __future__ import annotations
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

PROMPT = ("Summarize the remediation steps for a database connection pool "
          "exhaustion incident in three bullet points.")


def _one_call(url: str, max_tokens: int):
    import tritonclient.http as httpclient
    import numpy as np
    client = httpclient.InferenceServerClient(url=url)
    inp = httpclient.InferInput("text_input", [1], "BYTES")
    inp.set_data_from_numpy(np.array([PROMPT.encode()], dtype=object))
    mt = httpclient.InferInput("max_tokens", [1], "INT32")
    mt.set_data_from_numpy(np.array([max_tokens], dtype=np.int32))
    t0 = time.perf_counter()
    out = client.infer("ensemble", inputs=[inp, mt])
    dt = time.perf_counter() - t0
    text = out.as_numpy("text_output")[0].decode()
    return dt, len(text.split())


def run(url: str, concurrency: int, n: int, max_tokens: int):
    latencies, tokens = [], []
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(_one_call, url, max_tokens) for _ in range(n)]
        for f in futures:
            dt, tok = f.result()
            latencies.append(dt * 1000)
            tokens.append(tok)
    wall = time.perf_counter() - t_start
    latencies.sort()
    return {
        "concurrency": concurrency,
        "requests": n,
        "tokens_per_sec": round(sum(tokens) / wall, 1),
        "p50_ms": round(statistics.median(latencies), 1),
        "p99_ms": round(latencies[int(len(latencies) * 0.99) - 1], 1),
        "mean_ms": round(statistics.mean(latencies), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="localhost:8000")
    ap.add_argument("--concurrency", default="1,4,8")
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", default="benchmarks/results.json")
    args = ap.parse_args()

    results = []
    for c in [int(x) for x in args.concurrency.split(",")]:
        print(f"running concurrency={c} ...")
        results.append(run(args.url, c, args.requests, args.max_tokens))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
