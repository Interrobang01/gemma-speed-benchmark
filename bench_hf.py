"""Plain HuggingFace `transformers` baseline -- the CONTROL for the bake-off.

vLLM and SGLang are optimized *inference engines*: paged/quantized KV cache,
continuous batching, CUDA graphs, prefix reuse. This script is the naive thing
everyone starts with -- `model.generate()` with manual padded batching -- so the
other two have something honest to beat. Expect it to be dramatically slower and
to OOM at a far lower concurrency, because vanilla transformers has:

  * NO paged KV cache  -> KV memory is one big dense rectangle padded to the
    longest sequence, so it explodes with batch size and runs out fast.
  * NO continuous batching -> the whole batch advances in lockstep; one slow
    sequence stalls all of them (here they're equal-length, so it's the KV cost
    that dominates).
  * NO prefix sharing -> the identical multi-agent system prompt is recomputed
    and re-stored per sequence. The thing vLLM/SGLang exploit, thrown away.

That gap is the whole point of the benchmark. We mirror bench_vllm.py's
methodology exactly so the numbers are comparable:

  * same agentic workload (shared system/tool prefix + per-agent task)
  * same fixed-length generation for EXACT token accounting -- we set
    min_new_tokens == max_new_tokens, which forces every sequence to emit
    precisely `max_tokens` tokens (transformers' equivalent of vLLM ignore_eos).
  * concurrency K = batch size; we find the K* that maximizes aggregate tok/s.

OOM is expected and handled: when a batch won't fit we record `oom_at` and stop
the sweep cleanly (the smaller points are already saved).

Run:  python bench_hf.py                  # both models, default sweep
      python bench_hf.py --models moe     # just the MoE
      python bench_hf.py --quick          # tiny smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import benchlog
from agentic_workload import build_conversations

log = logging.getLogger("bench")

# Same checkpoints as bench_vllm.py so the three engines are compared apples--
# to-apples on identical weights.
MODELS = {
    "moe": {
        "label": "gemma-4-26B-A4B (MoE, 4B active)",
        "id": "google/gemma-4-26B-A4B-it",
        # 80GB: bf16. <60GB: bitsandbytes 8-bit (transformers' easy quant path).
        "quant_40gb": "bnb8",
    },
    "dense": {
        "label": "gemma-4-31B (dense)",
        # QAT w4a16 compressed-tensors checkpoint; transformers loads it via the
        # compressed-tensors integration, no extra flag needed. Fits one GPU.
        "id": "google/gemma-4-31B-it-qat-w4a16-ct",
        "quant_40gb": None,
    },
}


def gpu_vram_gb() -> float:
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def build_model(spec: dict, vram_gb: float):
    """Load weights + tokenizer with plain transformers (no engine in sight)."""
    kwargs = dict(torch_dtype=torch.bfloat16, device_map="cuda",
                  attn_implementation="sdpa")
    if vram_gb < 60 and spec["quant_40gb"] == "bnb8":
        from transformers import BitsAndBytesConfig
        kwargs.pop("torch_dtype")
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        log.info("  [<60GB GPU -> bitsandbytes 8-bit weight quant]")

    benchlog.log_config(log, "from_pretrained", {"id": spec["id"], **kwargs})
    log.info("  loading %s (plain transformers) ...", spec["id"])
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(spec["id"], **kwargs).eval()
    log.info("  loaded in %.1fs", time.perf_counter() - t0)

    tok = AutoTokenizer.from_pretrained(spec["id"])
    # Decoder-only batched generation REQUIRES left padding (so all sequences'
    # last real token sits at the same position before generation starts).
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        log.debug("  tokenizer had no pad token; using eos (%s)", tok.eos_token)
    return model, tok


@torch.inference_mode()
def time_run(model, tok, convos, max_tokens: int, n_samples: int = 0):
    """Generate for K conversations as one padded batch. Returns (wall_s,
    out_tokens, prompt_len). Forces exactly max_tokens new tokens per sequence.
    n_samples>0 logs that many full prompt->output pairs to the logfile."""
    prompts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
               for c in convos]
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    prompt_len = enc["input_ids"].shape[1]
    log.debug("  batch=%d  padded_prompt_len=%d  kv_rectangle≈%d tok-slots",
              len(convos), prompt_len, len(convos) * (prompt_len + max_tokens))

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(
        **enc,
        max_new_tokens=max_tokens,
        min_new_tokens=max_tokens,        # == max -> exact accounting (ignore_eos analog)
        do_sample=True, temperature=0.7, top_p=0.9,
        pad_token_id=tok.pad_token_id,
        use_cache=True,
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    gen = out[:, prompt_len:]             # min==max forces full width, all real tokens
    out_toks = gen.shape[0] * gen.shape[1]
    for i in range(min(n_samples, gen.shape[0])):
        benchlog.log_sample_output(
            log, f"K={len(convos)} agent#{i}",
            tok.decode(gen[i], skip_special_tokens=True),
            task=convos[i][1]["content"])
    return dt, out_toks, prompt_len


def sweep_model(name: str, spec: dict, vram_gb: float, concurrencies, max_tokens: int):
    log.info("\n%s\n%s  (%s)\n%s", "=" * 72, spec["label"], spec["id"], "=" * 72)
    results = {"label": spec["label"], "id": spec["id"], "frontier": []}

    model, tok = build_model(spec, vram_gb)
    benchlog.log_mem(log, "after load")

    # Warmup: first generate() pays one-time CUDA/kernel init; don't time it.
    log.debug("  warmup generate (K=2, 16 tokens) ...")
    time_run(model, tok, build_conversations(2), 16)

    peak_agg, peak_K = 0.0, 0
    for K in concurrencies:
        convos = build_conversations(K, share_prefix=True)
        try:
            # Log a few full prompt->output pairs at the first concurrency.
            dt, toks, plen = time_run(model, tok, convos, max_tokens,
                                      n_samples=(3 if K == concurrencies[0] else 0))
        except torch.cuda.OutOfMemoryError:
            log.warning("  agents=%4d  -> CUDA OOM (no paged KV cache); stopping sweep", K)
            results["oom_at"] = K
            torch.cuda.empty_cache()
            break

        agg = toks / dt
        per_agent = agg / K
        results["frontier"].append(
            {"agents": K, "agg_tok_s": round(agg, 1),
             "per_agent_tok_s": round(per_agent, 1),
             "wall_s": round(dt, 2), "prompt_len": plen})
        log.debug("  detail K=%d  out_toks=%d  wall=%.3fs  prompt_len=%d", K, toks, dt, plen)
        log.info("  agents=%4d  agg=%8.1f tok/s   per-agent=%7.1f tok/s   wall=%6.2fs",
                 K, agg, per_agent, dt)

        if agg > peak_agg:
            peak_agg, peak_K = agg, K
        elif agg < 0.85 * peak_agg:
            log.info("  (aggregate past its peak -> stopping sweep)")
            break

    results["sweet_spot"] = {"agents": peak_K, "agg_tok_s": round(peak_agg, 1)}
    log.info("  >> sweet spot: %d agents @ %.0f aggregate tok/s", peak_K, peak_agg)

    # Free THIS model before the next one loads. The del must happen here, where
    # the references actually live -- a helper deleting its own parameter alias
    # leaves these frame-locals alive and frees nothing.
    benchlog.log_mem(log, "before free")
    del model, tok
    benchlog.gpu_gc(log, "after free")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["moe"],
                    choices=list(MODELS),
                    help="dense 31B is opt-in: too big for one A100 (and plain "
                         "transformers dequantizes the w4a16 checkpoint to bf16)")
    ap.add_argument("--concurrency", type=int, nargs="+",
                    # Extend past where naive HF is expected to peak so the sweep
                    # ends on OOM or a real post-peak drop -- never on the list
                    # running out (which would hide whether a higher K is better).
                    default=[1, 4, 16, 64, 128, 256],
                    help="agent counts (batch sizes) to sweep")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results_hf.json")
    ap.add_argument("--log", default="bench_hf.log")
    args = ap.parse_args()

    if args.quick:
        args.concurrency = [1, 4, 16]
        args.max_tokens = 128

    benchlog.setup(args.log)
    benchlog.log_env(log)
    benchlog.log_workload_example(log)

    vram = gpu_vram_gb()
    benchlog.log_config(log, "run args", vars(args))

    all_results = {"engine": "hf-transformers", "vram_gb": round(vram, 1), "models": {}}
    t0 = time.perf_counter()
    for name in args.models:
        all_results["models"][name] = sweep_model(
            name, MODELS[name], vram, args.concurrency, args.max_tokens)
    all_results["total_wall_s"] = round(time.perf_counter() - t0, 1)

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("\nTotal wall: %ss   ->  wrote %s  (and log -> %s)",
             all_results["total_wall_s"], args.out, args.log)


if __name__ == "__main__":
    main()
