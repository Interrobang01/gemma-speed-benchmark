"""Context / KV-cache CAPACITY stress test for the Terrarium workload.

Different question from bench_vllm/bench_sglang. Those vary concurrency at a tiny
context to map throughput. This one fills each agent's context toward the model's
max (aiming at 64k) with realistic Terrarium memory -- a ~5k-token shared
rules+tools system prefix plus a big per-agent unique tail (diary/board/DMs) --
and sweeps **context_length x concurrency** to find where you run out of KV cache.

The thing it reveals: for a fleet of long-context agents, KV memory ≈
(agents x context_tokens). On an 80GB A100 the weights eat a fixed chunk and the
rest is the KV budget, so there's a hard frontier in (agents x context). This
maps it.

Note on what "the limit" looks like per engine:
  * vLLM / SGLang page + queue, so too many long agents usually DON'T OOM -- they
    just stop fitting in one step and get scheduled serially, so aggregate tok/s
    flattens or drops. A hard error only happens if a SINGLE sequence's context
    exceeds the whole KV cache (or max_model_len). We catch errors and record
    them, but mostly you read the frontier off the throughput curve.

One file, both engines (run in their own runtimes, like the others):
  python bench_context.py --engine vllm
  python bench_context.py --engine sglang
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import torch
from transformers import AutoTokenizer

import benchlog
from agentic_workload import build_terrarium_conversations, build_terrarium_system

log = logging.getLogger("bench")

MODELS = {
    "moe": "google/gemma-4-26B-A4B-it",
    "dense": "google/gemma-4-31B-it-qat-w4a16-ct",
}


def vram_gb() -> float:
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def make_vllm(model_id, quant, max_len, max_seqs, out_tokens):
    from vllm import LLM, SamplingParams
    kwargs = dict(model=model_id, max_model_len=max_len, gpu_memory_utilization=0.90,
                  max_num_seqs=max_seqs, enable_prefix_caching=True,
                  quantization=quant, disable_log_stats=True)
    benchlog.log_config(log, "vLLM LLM", kwargs)
    log.info("building vLLM engine (max_model_len=%d) ...", max_len)
    t0 = time.perf_counter()
    llm = LLM(**kwargs)
    log.info("  engine ready in %.1fs", time.perf_counter() - t0)
    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=out_tokens, ignore_eos=True)

    def gen(convos):
        outs = llm.chat(convos, sp, use_tqdm=False)
        toks = sum(len(o.outputs[0].token_ids) for o in outs)
        return toks, outs[0].outputs[0].text

    return gen


def make_sglang(model_id, quant, max_len, max_seqs, out_tokens, tok):
    import sglang as sgl
    kwargs = dict(model_path=model_id, context_length=max_len, mem_fraction_static=0.90,
                  quantization=quant, disable_radix_cache=False)
    benchlog.log_config(log, "sgl.Engine", kwargs)
    log.info("building SGLang engine (context_length=%d) ...", max_len)
    t0 = time.perf_counter()
    llm = sgl.Engine(**kwargs)
    log.info("  engine ready in %.1fs", time.perf_counter() - t0)
    sp = {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": out_tokens, "ignore_eos": True}

    def gen(convos):
        prompts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                   for c in convos]
        outs = llm.generate(prompts, sp)
        toks = sum(o["meta_info"]["completion_tokens"] for o in outs)
        return toks, outs[0]["text"]

    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["vllm", "sglang"])
    ap.add_argument("--model", default="moe", choices=list(MODELS))
    ap.add_argument("--context", type=int, nargs="+",
                    default=[4096, 8192, 16384, 32768, 65536],
                    help="per-agent total prompt token targets to sweep")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32, 128],
                    help="number of simultaneous agents to sweep at each context size")
    ap.add_argument("--shared-prefix", type=int, default=5000,
                    help="size of the identical system/tools prefix (prefix-cache target)")
    ap.add_argument("--out-tokens", type=int, default=128,
                    help="tokens generated per agent (the test is about prompt KV, not output)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    if args.quick:
        args.context = [4096, 16384]
        args.concurrency = [1, 16]

    args.out = args.out or f"results_context_{args.engine}.json"
    args.log = args.log or f"bench_context_{args.engine}.log"

    benchlog.setup(args.log)
    benchlog.log_env(log)
    benchlog.log_config(log, "run args", vars(args))

    model_id = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(model_id)

    # Log the shared prefix size we actually achieved (sanity check).
    sys_tokens = len(tok(build_terrarium_system(tok, args.shared_prefix),
                         add_special_tokens=False)["input_ids"])
    log.info("[ctx] shared system prefix = %d tokens (target %d)", sys_tokens, args.shared_prefix)

    max_ctx = max(args.context)
    max_len = max_ctx + args.out_tokens + 256          # room for generation
    max_seqs = max(args.concurrency)
    vram = vram_gb()
    # MoE on a <60GB card needs weight quant to leave any KV room at all.
    quant = None
    if args.model == "moe" and vram < 60:
        quant = "int8_per_channel_weight_only" if args.engine == "vllm" else "w8a8_int8"
        log.info("[ctx] <60GB GPU -> quantizing MoE weights (%s) to free KV room", quant)

    if args.engine == "vllm":
        gen = make_vllm(model_id, quant, max_len, max_seqs, args.out_tokens)
    else:
        gen = make_sglang(model_id, quant, max_len, max_seqs, args.out_tokens, tok)

    results = {"engine": args.engine, "model": args.model, "vram_gb": round(vram, 1),
               "shared_prefix_tokens": sys_tokens, "out_tokens": args.out_tokens, "grid": {}}

    sampled = False
    for C in args.context:
        results["grid"][str(C)] = {}
        skip = False
        for K in args.concurrency:
            if skip:
                log.info("  ctx=%6d  agents=%4d  -> skipped (lower K already failed)", C, K)
                results["grid"][str(C)][str(K)] = "skipped"
                continue

            convos = build_terrarium_conversations(K, tok, target_ctx_tokens=C,
                                                   shared_prefix_tokens=args.shared_prefix)
            actual = len(tok(tok.apply_chat_template(convos[0], tokenize=False,
                                                     add_generation_prompt=True),
                             add_special_tokens=False)["input_ids"])
            kv_tokens = K * actual
            log.debug("  ctx=%d K=%d  per-agent prompt=%d tok  total KV≈%d tok",
                      C, K, actual, kv_tokens)
            try:
                t0 = time.perf_counter()
                toks, sample = gen(convos)
                dt = time.perf_counter() - t0
                agg = toks / dt
                cell = {"agents": K, "ctx_tokens": actual, "kv_tokens": kv_tokens,
                        "agg_tok_s": round(agg, 1), "per_agent_tok_s": round(agg / K, 1),
                        "wall_s": round(dt, 2)}
                results["grid"][str(C)][str(K)] = cell
                log.info("  ctx=%6d  agents=%4d  ctx_tok=%6d  KV≈%9d  agg=%8.1f tok/s  wall=%6.2fs",
                         C, K, actual, kv_tokens, agg, dt)
                if not sampled:
                    benchlog.log_sample_output(log, f"ctx={C} K={K}", sample)
                    sampled = True
            except Exception as e:           # OOM or "context > KV cache" etc.
                log.warning("  ctx=%6d  agents=%4d  -> %s: %s (stopping higher K at this ctx)",
                            C, K, type(e).__name__, e)
                log.debug("  traceback:", exc_info=True)
                results["grid"][str(C)][str(K)] = f"ERROR: {type(e).__name__}: {e}"
                skip = True
                torch.cuda.empty_cache()

    # Compact 2D summary to console (rows=context, cols=concurrency).
    log.info("\n  SUMMARY (aggregate tok/s; ERR=failed, --=skipped):")
    header = "    ctx\\K  " + "".join(f"{K:>10d}" for K in args.concurrency)
    log.info(header)
    for C in args.context:
        row = f"  {C:>8d}  "
        for K in args.concurrency:
            cell = results["grid"][str(C)].get(str(K))
            if isinstance(cell, dict):
                row += f"{cell['agg_tok_s']:>10.0f}"
            elif cell == "skipped":
                row += f"{'--':>10}"
            else:
                row += f"{'ERR':>10}"
        log.info(row)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    log.info("\nwrote %s  (and log -> %s)", args.out, args.log)


if __name__ == "__main__":
    main()
