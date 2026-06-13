"""Map the throughput Pareto frontier of Gemma 4 agents on one A100 (vLLM).

What it produces, per model:
  1. A concurrency sweep: aggregate output tok/s and per-agent tok/s as we add
     more simultaneous agents. This is the Pareto frontier you asked for.
  2. The "sweet spot": the concurrency K* that maximizes AGGREGATE tok/s
     (tps/agent x agents), which is the number we actually care about.
  3. Interventions at/around K*: Multi-Token Prediction (MTP) speculative
     decoding via the Gemma 4 assistant drafters.

Design notes:
  * We hold the engine at a high max_num_seqs and control concurrency by
    submitting exactly K conversations at once. With ignore_eos + fixed
    max_tokens every sequence emits the same token count, so accounting is
    exact and the K prompts start/finish together -> clean per-agent number.
  * When K exceeds what KV cache can hold, vLLM queues -> wall time stops
    improving. That bend IS the frontier; we detect the peak and stop early.
  * Single Colab GPU => no tensor parallel. The 31B dense model does not fit in
    bf16 on 80GB alongside any KV cache, so we use the QAT w4a16 checkpoint.
    The 26B-A4B MoE fits in bf16 on 80GB (int8 on 40GB).

Run:  python bench_vllm.py                 # both models, default sweep
      python bench_vllm.py --models moe    # just the MoE
      python bench_vllm.py --quick         # tiny sweep, ~3 min
Requires: a recent vLLM with Gemma 4 + gemma4_mtp (>= the PR #41745 release),
and `huggingface-cli login` with Gemma 4 license accepted (gated models).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time

import torch

import benchlog
from agentic_workload import build_conversations

log = logging.getLogger("bench")

# ----------------------------------------------------------------------------
# Model registry. Precision is chosen at runtime from detected VRAM.
# ----------------------------------------------------------------------------
MODELS = {
    "moe": {
        "label": "gemma-4-26B-A4B (MoE, 4B active)",
        "id": "google/gemma-4-26B-A4B-it",
        "assistant": "google/gemma-4-26B-A4B-it-assistant",  # MTP drafter
        # 80GB: bf16 fits. 40GB: int8 weight-only (~47% savings per vLLM recipe).
        "quant_40gb": "int8_per_channel_weight_only",
        "tp": 1,
    },
    "dense": {
        "label": "gemma-4-31B (dense)",
        # Dense 31B needs TP=2 in bf16; on ONE GPU use the QAT w4a16 checkpoint
        # (quantization auto-detected from the checkpoint, fits 80GB and 40GB).
        "id": "google/gemma-4-31B-it-qat-w4a16-ct",
        "assistant": "google/gemma-4-31B-it-assistant",
        "quant_40gb": None,  # already quantized
        "tp": 1,
    },
}


def gpu_vram_gb() -> float:
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024**3)


def build_llm(spec: dict, vram_gb: float, mtp_tokens: int = 0):
    """Construct a vLLM engine. mtp_tokens>0 enables MTP speculative decoding."""
    from vllm import LLM

    kwargs = dict(
        model=spec["id"],
        tensor_parallel_size=spec["tp"],
        max_model_len=8192,            # plenty for an agent turn; bounds KV use
        gpu_memory_utilization=0.90,
        max_num_seqs=512,              # high cap; we throttle by submission size
        enable_prefix_caching=True,    # vLLM also caches the shared prefix
        enforce_eager=False,           # keep CUDA graphs for realistic speed
        disable_log_stats=True,
    )
    # On 40GB, the bf16 MoE won't fit -> apply weight-only quant if defined.
    if vram_gb < 60 and spec.get("quant_40gb"):
        kwargs["quantization"] = spec["quant_40gb"]
        log.info("  [<60GB GPU detected -> quantization=%s]", spec["quant_40gb"])
    if mtp_tokens > 0:
        kwargs["speculative_config"] = {
            "method": "mtp",
            "model": spec["assistant"],
            "num_speculative_tokens": mtp_tokens,
        }
    benchlog.log_config(log, f"vLLM LLM (mtp_tokens={mtp_tokens})", kwargs)
    log.info("  building vLLM engine for %s (mtp_tokens=%d) ...", spec["id"], mtp_tokens)
    t0 = time.perf_counter()
    llm = LLM(**kwargs)
    log.info("  engine ready in %.1fs", time.perf_counter() - t0)
    return llm


def time_run(llm, convos, max_tokens: int, sample: bool = False):
    """Run K conversations to completion; return wall seconds and total out toks."""
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=max_tokens,
                        ignore_eos=True)  # force exactly max_tokens each
    t0 = time.perf_counter()
    outs = llm.chat(convos, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    out_toks = sum(len(o.outputs[0].token_ids) for o in outs)
    if sample and outs:
        benchlog.log_sample_output(log, f"K={len(convos)}", outs[0].outputs[0].text)
    return dt, out_toks


def sweep_model(name: str, spec: dict, vram_gb: float, concurrencies, max_tokens: int,
                mtp_tokens: int):
    log.info("\n%s\n%s  (%s)\n%s", "=" * 72, spec["label"], spec["id"], "=" * 72)
    results = {"label": spec["label"], "id": spec["id"], "frontier": [], "mtp": []}

    # ---- 1. Baseline frontier (no MTP) -------------------------------------
    llm = build_llm(spec, vram_gb, mtp_tokens=0)
    # Warmup: triggers CUDA graph capture + prefix-cache fill so it isn't timed.
    log.debug("  warmup (K=2, 16 tokens) + capturing a sample output ...")
    time_run(llm, build_conversations(2), 16, sample=True)

    peak_agg, peak_K = 0.0, 0
    for K in concurrencies:
        convos = build_conversations(K, share_prefix=True)
        dt, toks = time_run(llm, convos, max_tokens, sample=(K == concurrencies[0]))
        agg = toks / dt
        per_agent = agg / K
        results["frontier"].append(
            {"agents": K, "agg_tok_s": round(agg, 1),
             "per_agent_tok_s": round(per_agent, 1), "wall_s": round(dt, 2)}
        )
        log.debug("  detail K=%d  out_toks=%d  wall=%.3fs", K, toks, dt)
        log.info("  agents=%4d  agg=%8.1f tok/s   per-agent=%7.1f tok/s   wall=%6.2fs",
                 K, agg, per_agent, dt)
        if agg > peak_agg:
            peak_agg, peak_K = agg, K
        # Early stop: once aggregate falls well below the peak, we're past the knee.
        elif agg < 0.85 * peak_agg:
            log.info("  (aggregate past its peak -> stopping sweep)")
            break

    results["sweet_spot"] = {"agents": peak_K, "agg_tok_s": round(peak_agg, 1)}
    log.info("  >> sweet spot: %d agents @ %.0f aggregate tok/s", peak_K, peak_agg)
    _free(llm)

    # ---- 2. Intervention: MTP at low concurrency AND at the sweet spot -----
    # MTP helps most when the GPU is underutilized (few agents); at the
    # throughput sweet spot the batch already saturates compute so it may not.
    if mtp_tokens > 0:
        log.info("\n  -- intervention: MTP (num_speculative_tokens=%d) --", mtp_tokens)
        try:
            llm = build_llm(spec, vram_gb, mtp_tokens=mtp_tokens)
            time_run(llm, build_conversations(2), 16)
            for K in sorted({1, 4, peak_K}):
                convos = build_conversations(K, share_prefix=True)
                dt, toks = time_run(llm, convos, max_tokens)
                agg, per_agent = toks / dt, toks / dt / K
                # Compare against the matching baseline point if we have it.
                base = next((r for r in results["frontier"] if r["agents"] == K), None)
                speedup = (per_agent / base["per_agent_tok_s"]) if base else float("nan")
                results["mtp"].append(
                    {"agents": K, "agg_tok_s": round(agg, 1),
                     "per_agent_tok_s": round(per_agent, 1),
                     "speedup_vs_baseline": round(speedup, 2)}
                )
                log.info("  agents=%4d  per-agent=%7.1f tok/s   speedup=%.2fx   agg=%.0f tok/s",
                         K, per_agent, speedup, agg)
            _free(llm)
        except Exception as e:  # MTP wiring is the most fragile part; don't kill the run
            log.warning("  [MTP run failed: %s: %s]", type(e).__name__, e)
            log.debug("  MTP traceback:", exc_info=True)
            results["mtp_error"] = str(e)

    return results


def _free(llm):
    """vLLM holds the GPU until the engine is GC'd; force it between models."""
    try:
        del llm
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["moe", "dense"],
                    choices=list(MODELS), help="which models to benchmark")
    ap.add_argument("--concurrency", type=int, nargs="+",
                    default=[1, 4, 16, 64, 128, 256],
                    help="agent counts to sweep")
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="output tokens per agent (fixed, ignore_eos)")
    ap.add_argument("--mtp-tokens", type=int, default=3,
                    help="num_speculative_tokens for MTP; 0 disables the MTP run")
    ap.add_argument("--quick", action="store_true",
                    help="tiny sweep + short outputs for a ~3 min smoke test")
    ap.add_argument("--out", default="results_vllm.json")
    ap.add_argument("--log", default="bench_vllm.log")
    args = ap.parse_args()

    if args.quick:
        args.concurrency = [1, 8, 64]
        args.max_tokens = 128

    benchlog.setup(args.log)
    benchlog.log_env(log)
    benchlog.log_workload_example(log)

    vram = gpu_vram_gb()
    benchlog.log_config(log, "run args", vars(args))

    all_results = {"engine": "vllm", "vram_gb": round(vram, 1), "models": {}}
    t0 = time.perf_counter()
    for name in args.models:
        all_results["models"][name] = sweep_model(
            name, MODELS[name], vram, args.concurrency, args.max_tokens, args.mtp_tokens)
    all_results["total_wall_s"] = round(time.perf_counter() - t0, 1)

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("\nTotal wall: %ss   ->  wrote %s  (and log -> %s)",
             all_results["total_wall_s"], args.out, args.log)


if __name__ == "__main__":
    main()
