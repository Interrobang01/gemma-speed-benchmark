"""SGLang shared-prefix benchmark for Gemma 4 agents.

The question this answers: when many agents share the SAME long tool/system
prompt (the common case for an agent fleet), how much does SGLang's
RadixAttention prefix caching buy you?

We run the identical concurrency sweep twice:
  * share_prefix=True  -> every agent reuses the cached system+tools prefix
  * share_prefix=False -> each agent's prefix is salted -> no reuse (control)
The gap is the prefix-cache win. We also report SGLang's own cache-hit rate.

Why a separate file from bench_vllm.py: SGLang and vLLM pin conflicting
torch / flashinfer versions. Run this in a FRESH Colab runtime after
`pip install "sglang[all]"` (see RUN_ON_COLAB.md).

Run:  python bench_sglang.py                  # MoE, default sweep
      python bench_sglang.py --model dense     # 31B w4a16
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import torch

import benchlog
from agentic_workload import build_conversations

log = logging.getLogger("bench")

MODELS = {
    "moe": "google/gemma-4-26B-A4B-it",          # fits bf16 on 80GB
    "dense": "google/gemma-4-31B-it-qat-w4a16-ct",  # quantized for single GPU
}


def vram_gb() -> float:
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def _to_prompts(convos, tok):
    """Apply the model's chat template -> plain text prompts (portable across
    SGLang versions; avoids relying on Engine.generate accepting `messages`)."""
    return [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in convos]


def run(model_id: str, concurrencies, max_tokens: int):
    import sglang as sgl
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    quant = "w8a8_int8" if vram_gb() < 60 and "qat" not in model_id else None
    engine_kwargs = dict(model_path=model_id, mem_fraction_static=0.90,
                         context_length=8192, quantization=quant,
                         disable_radix_cache=False)
    benchlog.log_config(log, "sgl.Engine", engine_kwargs)
    log.info("Loading %s (quant=%s) ...", model_id, quant)
    t0 = time.perf_counter()
    # RadixAttention prefix caching is ON by default in SGLang.
    llm = sgl.Engine(**engine_kwargs)
    log.info("  engine ready in %.1fs", time.perf_counter() - t0)
    sp = {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": max_tokens,
          "ignore_eos": True}
    benchlog.log_config(log, "sampling params", sp)

    results = {"id": model_id, "shared": [], "unshared": []}
    for share in (True, False):
        key = "shared" if share else "unshared"
        log.debug("  warmup [%s] (K=2) -- also primes the radix cache when shared", key)
        # Warmup also primes the radix cache when share=True.
        llm.generate(_to_prompts(build_conversations(2, share_prefix=share), tok), sp)
        for K in concurrencies:
            prompts = _to_prompts(build_conversations(K, share_prefix=share), tok)
            t0 = time.perf_counter()
            outs = llm.generate(prompts, sp)
            dt = time.perf_counter() - t0
            toks = sum(o["meta_info"]["completion_tokens"] for o in outs)
            agg = toks / dt
            results[key].append({"agents": K, "agg_tok_s": round(agg, 1),
                                 "per_agent_tok_s": round(agg / K, 1),
                                 "wall_s": round(dt, 2)})
            if K == concurrencies[0] and outs:
                benchlog.log_sample_output(log, f"{key} K={K}", outs[0]["text"])
                log.debug("  meta_info[0]: %s", outs[0].get("meta_info"))
            log.info("  [%-8s] agents=%4d  agg=%8.1f tok/s  per-agent=%7.1f tok/s  wall=%6.2fs",
                     key, K, agg, agg / K, dt)

    # Speedup from prefix sharing at each concurrency.
    log.info("\n  prefix-cache win (shared / unshared aggregate tok/s):")
    for s, u in zip(results["shared"], results["unshared"]):
        win = s["agg_tok_s"] / u["agg_tok_s"] if u["agg_tok_s"] else float("nan")
        log.info("    agents=%4d  %.2fx", s["agents"], win)
        s["prefix_win"] = round(win, 2)

    llm.shutdown()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="moe", choices=list(MODELS))
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 16, 64, 256, 512])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default="results_sglang.json")
    ap.add_argument("--log", default="bench_sglang.log")
    args = ap.parse_args()

    benchlog.setup(args.log)
    benchlog.log_env(log)
    benchlog.log_workload_example(log)
    benchlog.log_config(log, "run args", vars(args))

    res = run(MODELS[args.model], args.concurrency, args.max_tokens)
    res["engine"] = "sglang"
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    log.info("\nwrote %s  (and log -> %s)", args.out, args.log)


if __name__ == "__main__":
    main()
