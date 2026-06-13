# Gemma 4 agent throughput benchmark — Colab runbook

Maps the **aggregate tokens/sec vs number-of-agents** Pareto frontier for
`gemma-4-26B-A4B` (MoE) and `gemma-4-31B` (dense) on one A100, then measures
two optimizations: **MTP speculative decoding** (vLLM) and **shared-prefix
RadixAttention** (SGLang).

## 0. Get an 80GB A100
Runtime → Change runtime type → **A100 GPU** + **High-RAM**. If you only get a
40GB card the scripts auto-switch to int8/w4a16 — fewer KV slots, lower peak
concurrency, but everything still runs. Confirm with `!nvidia-smi`.

## 1. Auth (Gemma 4 is gated)
Accept the license on the model pages, then:
```python
from huggingface_hub import login; login()   # paste a token with gated-repo read
```
Upload the three files (`agentic_workload.py`, `bench_vllm.py`, `bench_sglang.py`)
or `!git clone` them into the working dir.

## 2. vLLM phase — frontier + MTP  (~10–15 min, or ~3 min with --quick)
```python
!pip install -q "vllm>=0.11"        # needs Gemma 4 + gemma4_mtp (PR #41745)
!python bench_vllm.py               # both models; add --quick for a smoke test
```
Writes `results_vllm.json`. Per model you get the sweep, the aggregate-throughput
**sweet spot** (the K* you asked to stay near), and MTP speedups at K=1, 4, K*.

> MTP is the most version-sensitive bit. If it errors, the baseline frontier is
> already saved and the MTP failure is recorded in the JSON — rerun later with
> `--mtp-tokens 0` to skip it cleanly.

## 3. SGLang phase — shared prefix  (~5 min) — FRESH RUNTIME
SGLang and vLLM pin conflicting torch/flashinfer. **Restart the runtime**
(Runtime → Restart) before installing SGLang:
```python
!pip install -q "sglang[all]"
!python bench_sglang.py             # --model dense for the 31B
```
Writes `results_sglang.json` with shared-vs-unshared aggregate tok/s and the
prefix-cache win at each concurrency.

## 4. Read the results
```python
import json; print(json.dumps(json.load(open("results_vllm.json")), indent=2))
```

## Knobs
- `--quick` — tiny sweep, short outputs, for verifying it runs end to end.
- `--concurrency 1 8 32 128` — custom agent counts.
- `--max-tokens 256` — output length per agent (fixed via ignore_eos).
- `--models moe` / `--model dense` — one model only (halves the time).
- `--mtp-tokens 3` — MTP draft depth (4 is vLLM's recipe default; 0 disables).

## Expectations / gotchas
- **Sweet spot:** aggregate tok/s climbs steeply with concurrency, then flattens
  when KV cache fills; the script stops once it's 15% past the peak.
- **MTP helps at LOW concurrency** (idle compute to verify drafts). At the
  throughput sweet spot the batch is already compute-bound, so MTP may be flat
  or slightly negative — that trade-off is exactly what the run reveals.
- **Dense 31B** can't run bf16 on a single 80GB card (needs TP=2); the script
  uses the QAT w4a16 checkpoint. For a true bf16 31B number you need 2 GPUs.
- If a quantization flag name is rejected, the model lineup moved since this was
  written — check the current vLLM Gemma 4 recipe page and update `MODELS`.
- H100 instead of A100: everything runs as-is and faster; on H100 you could
  additionally try native fp8 (`--quantization fp8`), which A100 lacks.
