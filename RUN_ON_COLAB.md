# Gemma 4 agent throughput benchmark — Colab runbook

Maps the **aggregate tokens/sec vs number-of-agents** Pareto frontier for
`gemma-4-26B-A4B` (MoE) and `gemma-4-31B` (dense) on one A100, across **three
inference paths**, so you can see what the optimized engines actually buy you:

| script           | engine                         | what it isolates                          |
|------------------|--------------------------------|-------------------------------------------|
| `bench_hf.py`    | plain HuggingFace transformers | the naive baseline (the control)          |
| `bench_vllm.py`  | vLLM                           | paged KV + continuous batching + **MTP**  |
| `bench_sglang.py`| SGLang                         | **RadixAttention** shared-prefix caching  |

Each script writes a JSON results file **and** a verbose `.log` file. Console
output stays sparse (so Colab doesn't lag); the full detail — engine config, the
actual prompts, sample model outputs, per-run token counts — goes to the logfile.
**Download the `.log` files afterwards.**

The three engines pin conflicting `torch`/`flashinfer` versions, so each runs in
its **own fresh runtime**. Run HF first (it's already installed), then restart
between each engine.

---

## Cell 0 — auth + clone (run once per fresh runtime)
```python
from huggingface_hub import login; login()   # token with gated-repo read; accept the Gemma 4 license first
!git clone https://github.com/Interrobang01/gemma-speed-benchmark
%cd gemma-speed-benchmark
!nvidia-smi   # note the GPU (want A100 80GB) AND the top-right "CUDA Version" = max CUDA the driver supports
```

## Cell A — HuggingFace baseline (the control)  ~5–10 min
No special engine; transformers ships with Colab. We just make sure it's current
and add bitsandbytes in case you only got a 40GB card.
```python
!pip install -q -U transformers accelerate bitsandbytes compressed-tensors  # compressed-tensors loads the dense w4a16 checkpoint
!python bench_hf.py            # add --quick for a ~2 min smoke test; --models moe for one model
```
Writes `results_hf.json` + `bench_hf.log`. Expect this to OOM at a *low*
concurrency and post much lower aggregate tok/s — that's the baseline the engines
beat. `oom_at` in the JSON records where it fell over.

> **⟳ Runtime → Restart** before the next cell.

## Cell B — vLLM (frontier + MTP)  ~10–15 min
```python
!pip install -q -U vllm
import vllm, torch
print("vllm", vllm.__version__, "| torch", torch.__version__, "| torch.cuda", torch.version.cuda)
from vllm import LLM      # <-- if THIS line throws "libcudart.so.NN: cannot open shared object file", see the box below
!python bench_vllm.py     # both models; --quick smoke test; --mtp-tokens 0 to skip MTP
```
Writes `results_vllm.json` + `bench_vllm.log`. Per model: the sweep, the
aggregate-throughput **sweet spot** (K*), and MTP speedups at K=1, 4, K*.

> **`libcudart.so.13: cannot open shared object file`** (or `.so.12`) means the
> vLLM wheel's compiled extension was built for a CUDA version that doesn't match
> the torch in this runtime. Fix:
> ```python
> !pip uninstall -q -y torch torchvision torchaudio vllm   # then Restart runtime, then:
> !pip install -q vllm    # let vllm pull its OWN matching torch instead of reusing Colab's
> import torch, vllm; print(torch.version.cuda, vllm.__version__)   # cuda must match the wheel; import must succeed
> ```
> If `nvidia-smi` showed a driver CUDA *older* than the wheel needs, the driver
> can't run that wheel at all — pin an older vLLM whose wheel targets your CUDA
> (see the vLLM install page for the matching `--extra-index-url` cu12x build).
> Don't `apt install` a newer CUDA runtime to paper over it — you'll end up with
> two CUDA runtimes fighting.

> **MTP** is the most version-sensitive bit. If it errors, the baseline frontier
> is already saved and the failure is recorded in the JSON + logfile; rerun with
> `--mtp-tokens 0` to skip it cleanly.

> **⟳ Runtime → Restart** before the next cell.

## Cell C — SGLang (shared prefix)  ~5 min
```python
!pip install -q "sglang[all]"
!python bench_sglang.py    # --model dense for the 31B
```
Writes `results_sglang.json` + `bench_sglang.log` with shared-vs-unshared
aggregate tok/s and the prefix-cache win at each concurrency.

## Cell D — read everything back
```python
import json
for f in ("results_hf.json", "results_vllm.json", "results_sglang.json"):
    try: print(f, "\n", json.dumps(json.load(open(f)), indent=2), "\n")
    except FileNotFoundError: print(f, "-> not run yet")
# grab the logfiles too:
from google.colab import files
for f in ("bench_hf.log", "bench_vllm.log", "bench_sglang.log"):
    try: files.download(f)
    except Exception as e: print("skip", f, e)
```

---

## Knobs (all three scripts share these unless noted)
- `--quick` — tiny sweep, short outputs, for verifying it runs end to end.
- `--concurrency 1 8 32 128` — custom agent counts. HF defaults to a shorter
  sweep (`1 2 4 8 16 32 64`) because it OOMs early.
- `--max-tokens 256` — output length per agent (fixed via forced generation length).
- `--models moe` / `--model dense` (sglang) — one model only (halves the time).
- `--mtp-tokens 3` — vLLM only; MTP draft depth (0 disables).
- `--log NAME.log` — where the verbose trace goes.

## Expectations / gotchas
- **HF baseline** has no paged KV cache: KV is a dense rectangle padded to the
  longest sequence, so memory blows up with batch size. It will OOM far sooner
  than vLLM/SGLang — that ceiling *is* the result.
- **Sweet spot:** aggregate tok/s climbs steeply with concurrency, then flattens
  when KV cache fills; each script stops once it's ~15% past the peak.
- **MTP helps at LOW concurrency** (idle compute to verify drafts); at the
  throughput sweet spot the batch is already compute-bound, so MTP may be flat
  or slightly negative — that trade-off is what the run reveals.
- **Dense 31B** can't run bf16 on a single 80GB card (needs TP=2); all three
  scripts use the QAT w4a16 checkpoint. For a true bf16 31B number you need 2 GPUs.
- **H100 instead of A100:** everything runs as-is and faster; on H100 you could
  additionally try native fp8 (`--quantization fp8` for vLLM), which A100 lacks.
- If a quantization flag name is rejected, the model lineup moved since this was
  written — check the current vLLM/SGLang Gemma 4 recipe page and update `MODELS`.
