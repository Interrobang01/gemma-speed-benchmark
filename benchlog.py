"""Shared logging for the three benchmarks.

Philosophy (per request): you can never have enough logging, but you also
can't spam the Colab cell output or the UI lags. So we split:

  * CONSOLE (stdout)  -> sparse INFO: the per-run result lines + headers.
  * LOGFILE (on disk) -> everything at DEBUG: full config dicts, the actual
    prompts we feed the model, sample decoded outputs, token counts, timings.

Download the logfile afterwards to see exactly what happened. The console stays
readable while the file has the forensic detail.

Every bench script does:
    import benchlog, logging
    log = logging.getLogger("bench")          # module-level, used everywhere
    ...
    benchlog.setup(args.log)                   # once, in main()
    benchlog.log_env(log)
    benchlog.log_workload_example(log)
"""

from __future__ import annotations

import json
import logging
import sys


def setup(logfile: str, console_level: int = logging.INFO,
          file_level: int = logging.DEBUG) -> logging.Logger:
    """Configure the shared 'bench' logger. Call once from main()."""
    log = logging.getLogger("bench")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()           # idempotent across re-runs in one kernel
    log.propagate = False

    # File: timestamped, full detail.
    fh = logging.FileHandler(logfile, mode="w")
    fh.setLevel(file_level)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-7s %(message)s", "%H:%M:%S"))
    log.addHandler(fh)

    # Console: clean, no timestamps, INFO+ only.
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(ch)

    log.info("[log] full DEBUG trace -> %s  (download this afterwards)", logfile)
    return log


def _ver(modname: str) -> str:
    try:
        m = __import__(modname)
        return getattr(m, "__version__", "?")
    except Exception as e:
        return f"<not importable: {type(e).__name__}>"


def log_env(log: logging.Logger) -> None:
    """Dump the hardware + library environment. Console gets the GPU one-liner;
    the file gets the full version table (the stuff that explains weird perf)."""
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        cap = torch.cuda.get_device_capability(0)
        log.info("GPU: %s  |  VRAM: %.0f GB", name, vram)
        log.debug("  compute capability: sm_%d%d", cap[0], cap[1])
    else:
        log.warning("NO CUDA DEVICE VISIBLE -- everything below will fail")

    # File-only: the version matrix that decides whether anything imports.
    log.debug("env: torch=%s  torch.cuda=%s  cudnn=%s",
              torch.__version__, torch.version.cuda,
              torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None)
    for mod in ("transformers", "vllm", "sglang", "flashinfer", "bitsandbytes",
                "compressed_tensors", "triton"):
        log.debug("env: %-18s %s", mod, _ver(mod))


def log_workload_example(log: logging.Logger, num_tools: int | None = None) -> None:
    """Write ONE full sample conversation to the logfile so you can see exactly
    what's being fed in (the shared system prefix + a per-agent task)."""
    from agentic_workload import build_conversations, build_system_prompt

    sys_prompt = build_system_prompt() if num_tools is None else build_system_prompt(num_tools)
    convo = build_conversations(1)[0]
    log.debug("workload: shared system prefix is %d chars (~%d tokens rough):\n%s",
              len(sys_prompt), len(sys_prompt) // 4, sys_prompt)
    log.debug("workload: example full conversation:\n%s",
              json.dumps(convo, indent=2))
    log.info("[workload] shared system prefix ~%d chars; see logfile for the full text",
             len(sys_prompt))


def log_config(log: logging.Logger, what: str, cfg: dict) -> None:
    """Log an engine/sampling config dict (file gets the full thing)."""
    log.debug("%s config:\n%s", what, json.dumps(cfg, indent=2, default=str))


def log_sample_output(log: logging.Logger, tag: str, text: str, limit: int = 800) -> None:
    """Log a decoded sample generation to the file so you can eyeball quality."""
    clipped = text[:limit] + (" …[clipped]" if len(text) > limit else "")
    log.debug("sample output [%s]:\n%s", tag, clipped)
