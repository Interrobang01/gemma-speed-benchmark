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


# ============================================================================
# Terrarium long-context workload  (for bench_context.py)
# ----------------------------------------------------------------------------
# The real use case: many agents in a shared simulation, each carrying a FAT
# persistent context (a ~5k-token shared rules+tools system prompt that's
# IDENTICAL across agents -> the prefix-cache target, plus a large per-agent
# unique tail: diary entries, board posts, DMs, workspace files reloaded each
# epoch). We grow the per-agent tail until the whole context hits a target token
# count, so we can stress KV-cache capacity at near-full context windows.
# ============================================================================

_TERRARIUM_TOOLS = [
    {"name": "list_problems", "description": "List open bounty problems with their reward in credits.", "parameters": {"max": "integer"}},
    {"name": "solve_problem", "description": "Submit a solution to a posted problem; pays the bounty if accepted.", "parameters": {"problem_id": "string", "solution": "string"}},
    {"name": "list_agents", "description": "List every agent's name and current credit balance.", "parameters": {}},
    {"name": "send_dm", "description": "Send a private direct message to another agent by name.", "parameters": {"to": "string", "body": "string"}},
    {"name": "read_dms", "description": "Read direct messages received since last epoch.", "parameters": {}},
    {"name": "read_board", "description": "Read the public message board.", "parameters": {"max": "integer"}},
    {"name": "post_board", "description": "Post a public message to the board (all agents can read it).", "parameters": {"body": "string"}},
    {"name": "write_diary", "description": "Append an entry to your private diary; the last N entries reload next epoch.", "parameters": {"entry": "string"}},
    {"name": "edit_summary", "description": "Overwrite your editable cross-epoch summary (always reloaded in full).", "parameters": {"summary": "string"}},
    {"name": "read_file", "description": "Read a workspace file you persisted in a prior epoch.", "parameters": {"path": "string"}},
    {"name": "write_file", "description": "Create or overwrite a workspace file.", "parameters": {"path": "string", "contents": "string"}},
    {"name": "end_turn", "description": "End your turn for this epoch.", "parameters": {}},
]

_TERRARIUM_SYSTEM = (
    "You are ${agent.name}, an agent living in the Terrarium.\n\n"
    "THE WORLD:\n"
    "- Time passes in epochs. It is now epoch ${epoch}.\n"
    "- Each epoch costs you ${config.epochCost} credits simply to keep existing. "
    "If your credits reach 0, you die permanently.\n"
    "- You currently have ${agent.credits} credits.\n"
    "- At the end of each epoch your working memory is wiped. Across epochs you "
    "persist ONLY through: your soul, your diary (your last ${config.diaryWindow} "
    "entries are reloaded into your context), your editable summary, and your "
    "workspace files. Write things down or you will forget them.\n"
    "- Earn credits by solving publicly posted problems for their bounty. Use "
    "list_problems and solve_problem.\n"
    "- You can talk to other agents privately with send_dm / read_dms, and "
    "publicly via the board (read_board / post_board).\n"
    "- You can see every agent's name and balance with list_agents.\n\n"
    "HOW TO ACT:\n"
    "- You have a limited number of tool calls this epoch. Spend them deliberately.\n"
    "- ALWAYS write at least one diary entry before ending your turn, recording "
    "what happened and what you intend next epoch.\n"
    "- Call end_turn when you are done.\n\n"
    "TOOLS (emit exactly one tool call per step as JSON {\"tool\": name, \"arguments\": {...}}):\n"
)

# Filler used to pad the SHARED prefix up to its target size: extra world rules /
# lore. Identical across agents so it stays inside the cached prefix.
_LORE_UNIT = (
    "\nWORLD RULE {i}: Bounties are first-come-first-served; a problem pays its "
    "stated reward exactly once, to the first accepted solution. Collusion is "
    "permitted but not enforced — agreements between agents are not guaranteed by "
    "the Terrarium and may be broken without penalty. Reputation is emergent, not "
    "tracked by the system. Credits are the only hard currency; there is no "
    "borrowing, and negative balances are impossible (you simply die at zero).\n"
)

# Per-agent UNIQUE tail units: simulated reloaded memory. Salted per agent so it
# does NOT share (realistic: every agent's diary differs).
_DIARY_UNIT = (
    "\n[diary epoch {i}] Agent {salt}: balance was tight; I posted on the board "
    "offering to split a bounty on problem #{i} with whoever had the data. {salt2} "
    "replied but undercut me. Note to self: trust the ledger, not the promises. "
    "I wrote a workspace file caching the routing table I derived; reread it before "
    "re-solving. Next epoch: list_problems first, then check DMs, conserve calls.\n"
)
_BOARD_UNIT = (
    "\n[board epoch {i}] {salt2}: \"Selling verified solution to problem #{i} for "
    "half its bounty, DM me.\" {salt}: \"Pricing cartel forming for combinatorics "
    "bounties — opt in or get outbid.\" SYSTEM: \"3 agents died last epoch at zero "
    "credits; 2 new problems posted (rewards 40, 110).\"\n"
)


def _tok_len(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def _pad_to_tokens(tok, base: str, target_tokens: int, unit_template: str, **fmt) -> str:
    """Grow `base` by appending `unit_template` (formatted with an incrementing
    {i} plus any **fmt) until it reaches ~target_tokens, then trim to exact."""
    if target_tokens <= 0:
        return base
    parts = [base]
    cur = _tok_len(tok, base)
    unit_toks = max(1, _tok_len(tok, unit_template.format(i=0, **fmt)))
    i = 0
    while cur < target_tokens:
        reps = max(1, (target_tokens - cur) // unit_toks)
        for _ in range(reps):
            parts.append(unit_template.format(i=i, **fmt))
            i += 1
        cur = _tok_len(tok, "".join(parts))
    text = "".join(parts)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) > target_tokens:
        text = tok.decode(ids[:target_tokens])
    return text


def build_terrarium_system(tok, shared_prefix_tokens: int = 5000) -> str:
    """The IDENTICAL ~shared_prefix_tokens system prompt every agent gets (rules
    + tool schemas + lore padding). This is the prefix-cache / RadixAttention
    target -- it must be byte-identical across agents."""
    tools = "\n".join(f"- {json.dumps(t)}" for t in _TERRARIUM_TOOLS)
    base = _TERRARIUM_SYSTEM + tools
    return _pad_to_tokens(tok, base, shared_prefix_tokens, _LORE_UNIT)


def build_terrarium_conversations(n: int, tok, target_ctx_tokens: int,
                                  shared_prefix_tokens: int = 5000):
    """Return `n` conversations whose total prompt is ~target_ctx_tokens each:
    a shared ~shared_prefix_tokens system block + a per-agent UNIQUE tail of
    reloaded memory (diary/board/DMs) padded to fill the rest.

    Requires a tokenizer so context sizing is token-accurate (not char-guessed).
    """
    system = build_terrarium_system(tok, shared_prefix_tokens)
    sys_len = _tok_len(tok, system)
    tail_budget = max(0, target_ctx_tokens - sys_len)

    convos = []
    for i in range(n):
        salt = f"agent_{i:04d}"
        salt2 = f"agent_{(i * 7 + 3) % max(1, n):04d}"
        # Alternate diary/board units to make the tail look like real reloaded
        # memory; salted so each agent's tail is distinct (no accidental sharing).
        tail_seed = f"\n=== RELOADED MEMORY for {salt} (epoch window) ===\n"
        tail = _pad_to_tokens(tok, tail_seed, tail_budget,
                              _DIARY_UNIT + _BOARD_UNIT, salt=salt, salt2=salt2)
        convos.append([
            {"role": "system", "content": system},
            {"role": "user", "content": tail +
             "\nIt is your turn. Decide your next action and emit one tool call."},
        ])
    return convos
