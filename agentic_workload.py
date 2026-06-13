"""Synthetic *agentic* workload for throughput benchmarking.

Why synthetic instead of live web/tool calls:
  The goal is to map a *throughput* Pareto frontier (aggregate tok/s vs number
  of concurrent agents). Real online tool calls inject network latency and
  variance that swamp the GPU signal and make 10-minute runs impossible to
  interpret. So we reproduce the *shape* of agent traffic instead:

    * a long, IDENTICAL system prompt full of tool JSON schemas  -> big shared
      prefix (the thing prefix caching / RadixAttention exploits)
    * a per-agent task that forces planning + a structured tool call
    * bounded, fixed-length generation so token accounting is exact

This is what an agent's first hop looks like: huge shared tool/context prefix,
modest unique suffix, structured reasoning output. It is the realistic stressor
for "many agents sharing the same scaffold."

Knobs: prefix length is driven by NUM_TOOLS; output length by the caller's
max_tokens. Set SHARE_PREFIX=False to defeat prefix caching for an A/B.
"""

from __future__ import annotations

import json

# A realistic agent tool belt. Each schema is verbose on purpose: real agent
# system prompts are 1-4k tokens of tool definitions + instructions.
_TOOLS = [
    {
        "name": "web_search",
        "description": "Search the public web and return ranked result snippets with URLs.",
        "parameters": {"query": "string", "recency_days": "integer", "max_results": "integer"},
    },
    {
        "name": "open_url",
        "description": "Fetch a URL and return cleaned readable text plus extracted links.",
        "parameters": {"url": "string", "render_js": "boolean"},
    },
    {
        "name": "python",
        "description": "Execute Python in a sandbox; returns stdout/stderr. Use for math, parsing, data.",
        "parameters": {"code": "string", "timeout_s": "integer"},
    },
    {
        "name": "read_file",
        "description": "Read a file from the agent workspace by path.",
        "parameters": {"path": "string", "start_line": "integer", "end_line": "integer"},
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a workspace file with the given contents.",
        "parameters": {"path": "string", "contents": "string"},
    },
    {
        "name": "sql_query",
        "description": "Run a read-only SQL query against the analytics warehouse.",
        "parameters": {"sql": "string", "max_rows": "integer"},
    },
    {
        "name": "send_email",
        "description": "Draft and send an email on the user's behalf (requires confirmation).",
        "parameters": {"to": "string", "subject": "string", "body": "string"},
    },
    {
        "name": "finish",
        "description": "Return the final answer to the user and end the episode.",
        "parameters": {"answer": "string", "citations": "array"},
    },
]

# Tasks that genuinely require *agency*: multi-step, need tools / web, can't be
# answered from a single forward pass without planning.
_TASKS = [
    "Find the three most-cited papers on mixture-of-experts routing published since 2024 and summarize their routing differences.",
    "Determine the current best price for an 80GB A100 across the major cloud providers and recommend the cheapest spot option.",
    "Pull our last 7 days of signups from the warehouse, compute week-over-week growth, and flag any day that dropped >15%.",
    "Research whether Gemma 4's multi-token prediction is supported in the latest vLLM release and cite the PR.",
    "Compare the context-window pricing of the top 3 hosted LLM APIs and produce a table of $/1M input tokens.",
    "Find an open dataset of agent trajectories suitable for fine-tuning a tool-use model and verify its license permits commercial use.",
    "Check the workspace for a config file, read the current batch size, and write a tuned version that doubles it.",
    "Search for the latest CUDA driver compatible with an A100 on Ubuntu 22.04 and give the exact install command.",
    "Investigate why nightly ETL job 'orders_rollup' has been slow this week and propose one concrete fix.",
    "Draft an email to the infra team summarizing a proposed migration from HF Transformers to vLLM, with two risks.",
]

_SYSTEM_PREAMBLE = (
    "You are an autonomous software agent operating in a multi-step loop. On each "
    "turn you THINK briefly about the plan, then emit exactly ONE tool call as a "
    "JSON object: {\"tool\": <name>, \"arguments\": {...}}. Never invent tool "
    "results. Prefer the fewest tool calls that solve the task. When the task is "
    "complete, call the `finish` tool. You have access to the following tools:\n\n"
)


def build_system_prompt(num_tools: int = len(_TOOLS)) -> str:
    """The big SHARED prefix: identical across every agent in a run."""
    tools = _TOOLS[:num_tools]
    body = "\n".join(f"- {json.dumps(t)}" for t in tools)
    return (
        _SYSTEM_PREAMBLE
        + body
        + "\n\nAlways think step by step about which tool advances the task, then "
        "produce the single best next tool call. Be concise but explicit about "
        "your reasoning before the JSON call."
    )


def build_conversations(n: int, num_tools: int = len(_TOOLS), share_prefix: bool = True):
    """Return `n` chat conversations (list[ {role, content} ]).

    share_prefix=True  -> identical system block for all (prefix-cache friendly)
    share_prefix=False -> system block salted per-agent (defeats prefix cache)
    """
    base_system = build_system_prompt(num_tools)
    convos = []
    for i in range(n):
        system = base_system
        if not share_prefix:
            # A unique header forces a distinct prefix -> no cache reuse.
            system = f"[session-{i:06d}] " + base_system
        task = _TASKS[i % len(_TASKS)]
        convos.append(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Task: {task}\nBegin."},
            ]
        )
    return convos
