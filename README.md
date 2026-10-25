# ⚡ pydantic-ai-durable

[![CI](https://github.com/matheusbuniotto/pydantic-ai-durable/actions/workflows/ci.yml/badge.svg)](https://github.com/matheusbuniotto/pydantic-ai-durable/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pydantic-ai-durable)](https://pypi.org/project/pydantic-ai-durable/)
[![Python](https://img.shields.io/pypi/pyversions/pydantic-ai-durable)](https://pypi.org/project/pydantic-ai-durable/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Crash-proof, stateful, and malleable agents for [PydanticAI](https://github.com/pydantic/pydantic-ai).**  
> Complete Python implementation of Earendil's [@earendil-works/pi-durable](https://github.com/earendil-works) architecture.

---

### 🧠 The Problem vs The Fix

| Without pydantic-ai-durable 💥 | With pydantic-ai-durable 🛡️ |
| :--- | :--- |
| **Process restarts or crashes?** Everything is lost. | **Turns auto-resume** from their last checkpoint. |
| **Crashes during a tool call?** Retrying double-charges cards. | **Replay safety:** Safe tools re-fetch; mutating tools never duplicate. |
| **Exploring "what-if" speculations?** Re-run entire history. | **Fork instantly** in `O(1)` without copying database rows. |
| **Distributed multi-step task fails?** Dirty resources leaked. | **Durable Sagas:** Automatic reverse rollback compensation. |
| **AI wants to run a risky action?** Hard to intercept cleanly. | **Approval Gates:** Pauses run for human review & argument override. |

---

### 💥 See it survive a crash (offline, 10 seconds)

```bash
git clone https://github.com/matheusbuniotto/pydantic-ai-durable && cd pydantic-ai-durable
uv run python examples/crash_recovery.py
```

```text
Run 1: starting (will crash during deploy)
  [tool] running step: provision
  [tool] running step: migrate
  [tool] running step: deploy
  [tool] 💥 process killed mid-deploy (simulated kill -9)

Run 2: restarting the same process
  ♻️  recovered task: unsafe tools sealed (NOT re-run): 1, safe tools replayed: 0
```

The mutating `deploy` step is never silently repeated; safe read-only tools are replayed.
More in [`examples/`](examples/): [crash recovery](examples/crash_recovery.py), [forking a conversation](examples/fork_conversation.py), [approval gate with argument override](examples/approval_override.py) (an overeager agent tries to `DROP TABLE users`; the operator redirects it to `users_backup`).

---

## 🚀 30-Second Quickstart

```bash
uv pip install pydantic-ai-durable   # or: pip install pydantic-ai-durable

# Latest from GitHub (works before/without a PyPI release)
uv pip install "git+https://github.com/matheusbuniotto/pydantic-ai-durable"
```

Both package imports work interchangeably:
```python
import pydantic_ai_durable as pad

# or
import pya_durable as pad
```

### Complete Bootstrap in 15 Lines

```python
import asyncio
from pydantic_ai_durable import (
    Harness,
    SqliteStorage,
    create_models,
    openrouter_provider,
)


async def main():
    # 1. Pluggable Auth & Providers (OpenRouter, OpenAI, Anthropic, Gemini, Groq, Ollama)
    models = create_models()
    models.set_provider(openrouter_provider(api_key="sk-or-..."))
    # or models.set_provider("openai", api_key="sk-...")

    # 2. Open durable harness over SQLite WAL storage
    harness = await Harness.open(SqliteStorage("agent.db"), models=models)

    # 3. Resume any unfinished work from previous runs
    await harness.resume()

    # 4. Open root channel (shortcuts like "gpt-luna 6", dict specs, or Agent instances)
    root = await harness.root({"agent": {"model": "openrouter:openai/gpt-6-luna"}})

    # 5. Submit with idempotent requestId (safe against network retries)
    result = await root.submit({"content": "Audit cluster health", "requestId": "job-001"})
    print("Agent:", result.output)

    await harness.close()


asyncio.run(main())
```

---

## 🎯 Cheat Sheet: Which Feature Do I Need?

| If you want to... | Use this | What it does |
| :--- | :--- | :--- |
| **Prevent double-charging on crash** | `@durable_tool(replay="unsafe")` | Halts duplicate runs; preserves partial output |
| **Auto-rerun safe lookups on crash** | `@durable_tool(replay="safe")` | Re-runs read-only tools without user prompt |
| **Ask a human before dropping a table** | `@approval_gate` | Pauses turn, generates challenge token |
| **Roll back multi-step cloud steps** | `@define_saga` | Fan-out child tasks + reverse rollback |
| **Branch into a "what-if" experiment** | `conv.fork()` | Zero-copy transcript DAG off any message |
| **Inject mid-turn steering message** | `harness.steer("...")` | Prioritizes instructions in next step |
| **Build a live dashboard or TUI** | `harness.view_state()` | Emits real-time reactive transcript & tasks |
| **Store shared typed state** | `define_doc(...)` | Atomic document state alongside transcript |

---

## 🧱 Core Patterns (Copy & Paste)

### 1. 🛡️ Durable Tools: Safe Replay vs Unsafe Mutating Guard

Declare tool replay safety so crashes never cause accidental side-effects:

```python
from pydantic_ai_durable import durable_tool, set_tool_partial_output


# SAFE: Read-only operations automatically rerun on crash recovery
@durable_tool(replay="safe")
async def search_database(query: str) -> str:
    return f"Results for: {query}"


# UNSAFE: Mutating calls are NEVER repeated automatically
@durable_tool(replay="unsafe")
async def charge_customer(amount: int, customer_id: str) -> str:
    set_tool_partial_output("Connecting to payment gateway...")
    # charge logic...
    return f"Charged ${amount} to {customer_id}"
```

---

### 2. ⏸️ Human Approval Gates & Argument Overrides

Pause turn execution for sensitive operations and wait for human operator intervention:

```python
from pydantic_ai_durable import approval_gate, DurableHarness


# Intercept sensitive actions and generate challenge token
@approval_gate(description="Approve database migration on production")
async def drop_production_table(table_name: str) -> str:
    return f"Dropped {table_name}"


# The agent turn pauses with TaskStatus.WAITING ("waiting_approval").
# External operator reviews challenge token and approves (with optional argument rewrite):
await harness.resolve_approval(
    token=challenge_token,
    approved=True,
    override_args={"table_name": "users_backup"},  # Redirect to safe table
)
```

---

### 3. 🔁 Durable Task Sagas with Reverse Compensation

Orchestrate complex distributed workflows with parallel fan-out and automatic reverse rollback:

```python
from pydantic_ai_durable import define_saga, SagaContext, SagaResult, SagaPolicy


@define_saga(name="cloud.provision_cluster", initial_step="reserve_nodes")
class ProvisionClusterSaga:
    async def reserve_nodes(self, ctx: SagaContext) -> SagaResult:
        # Parallel fan-out across multiple child tasks
        w1 = await ctx.spawn_child("cloud.reserve", {"type": "gpu-h100"})
        w2 = await ctx.spawn_child("cloud.reserve", {"type": "cpu-m6i"})

        # Await with fail-fast cascade
        await ctx.wait_for_children(policy=SagaPolicy.FAIL_FAST)
        return ctx.transition_to("configure_network")

    async def configure_network(self, ctx: SagaContext) -> SagaResult:
        # If this step raises an error, compensate_reserve_nodes runs automatically!
        ctx.state["cluster_ready"] = True
        return ctx.complete(result="Cluster active")

    async def compensate_reserve_nodes(self, ctx: SagaContext) -> None:
        # Releases reserved compute instances bottom-up
        print("Rollback: Releasing reserved nodes...")
```

---

### 4. 🌲 Zero-Copy Transcript Branching (Forking)

Branch a conversation off any historical message in `O(1)` time without duplicating database rows:

```python
# Fork off specific answer entry
thread = await conv.fork(
    entry_id=entry.entry_id,
    new_conversation_id="conv-speculation-1",
)

# Both proceed concurrently with zero interference
await asyncio.gather(
    conv.run("Proceed with standard rollout"),
    thread.run("What happens if we roll back immediately?"),
)
```

---

### 5. 👥 Real-Time Multiplayer Streaming & View State

Connect TUIs, web dashboards, and mobile clients with reactive state subscriptions:

```python
# Obtain reactive view state observable
view = await harness.view_state()
print("Current turn count:", len(view.value.transcript))

# Subscribe to real-time incremental updates
unsubscribe = view.subscribe(lambda state: print("Updated active tasks:", state.active_tasks))

# Or consume high-level agent event stream
async for event in harness.watch_events():
    print(f"[{event.event_type}] {event.payload}")
```

---

### 6. 🗄️ Pluggable Storage Backends

Switch backends effortlessly based on your deployment tier:

```python
from pydantic_ai_durable import SqliteStorage, MemoryStorage, JsonlStorage

# 1. Production SQLite with WAL mode (Recommended default)
storage = SqliteStorage("agent.sqlite")

# 2. Zero-disk in-memory storage for lightning-fast tests
test_storage = MemoryStorage()

# 3. Append-only JSONL ledgers for streaming audit logs
audit_storage = JsonlStorage("audit_trail.jsonl")
```

---

## 🏛️ Architecture

```
                      ┌────────────────────────────────────────────────────────┐
                      │                     DURABLE HARNESS                    │
                      │  Storage: SQLite WAL / In-Memory / Append-Only JSONL   │
                      │  Extension Registry: Tools, Hooks, Prompt Sections     │
                      │  Multiplayer Hub: Presence, View State, Event Stream   │
                      └───────────────────────────┬────────────────────────────┘
                                                  │
                      ┌───────────────────────────┴────────────────────────────┐
                      │                                                        │
             ┌────────▼────────┐                                      ┌────────▼────────┐
             │  Conversation 1 │ (Root Channel)                       │  Conversation 2 │ (Fork / Thread)
             │  - Agent Config │                                      │  - Shared Tree  │ (O(1) Branching)
             │  - Transcript   │ (Parent-pointer DAG)                 │  - Sandbox Env  │
             └────────┬────────┘                                      └─────────────────┘
                      │
            ┌─────────┴──────────────────────────────────────┐
            │ Tasks & Ownership Hierarchy                    │
            ├───────────────────────┬────────────────────────┤
            │ Model Turn Task       │ Durable Replay Tool    │
            │ Multi-Step Saga Task  │ Child Subagent Run     │
            │ Background Compaction │ Human Approval Gate    │
            └───────────────────────┴────────────────────────┘
```

---

## ⚖️ Full Pi Durable Parity Table

`pydantic-ai-durable` brings 100% of Earendil's TypeScript framework to Python and PydanticAI:

| Architectural Pillar | Pi Durable (TypeScript) | PyA Durable (Python) | Status |
| :--- | :--- | :--- | :---: |
| **Model Providers & Auth** | `createModels()` + `setProvider()` | `create_models()` + `*_provider()` | Complete |
| **Harness Bootstrap** | `Harness.open()` + `resume()` | `Harness.open()` + `resume()` | Complete |
| **Storage Engine** | Node SQLite WAL / DO | `SqliteStorage` (WAL mode) | Complete |
| **Zero-Disk Storage** | `MemoryStorage` | `MemoryStorage` | Complete |
| **Append-Only Ledger** | `JsonlStorage` | `JsonlStorage` | Complete |
| **Turn Execution Loop** | Durable Engine Loop | `DurableStepEngine` on `Agent.iter()` | Complete |
| **Replay Policies** | `safe` / `unsafe` | `@durable_tool(replay="safe")` | Complete |
| **Hierarchical Tasks** | `defineTask` + ownership | `define_task` + `TaskRunner` | Complete |
| **Task Trees & Aborts** | Bottom-up cascade | `abort_task` bottom-up cascade | Complete |
| **Subagents** | Tool-owned conversations | `create_handoff_tool` / Subagents | Complete |
| **Branching (Forking)** | Non-copying parent DAG | `DurableConversation.fork()` | Complete |
| **Active Steering** | `whenBusy: "steer"` | `when_busy="steer"` + `SteeringQueue` | Complete |
| **Context Compaction** | Background summarization | `BackgroundCompactor` + `search_history` | Complete |
| **Extensions Registry** | `defineExtension` | `define_extension` + `ExtensionRegistry` | Complete |
| **Prompt Sections** | `section(...)` dynamic eval | `define_section(...)` dynamic eval | Complete |
| **Tool Wrapping & Hooks**| `wrapTool`, `beforeTool`, `afterTool` | `wrap_tool`, `LifecycleHook` | Complete |
| **Execution Envs** | `NodeExecutionEnv` + cwd | `LocalExecutionEnv`, `IsolatedSandboxEnv` | Complete |
| **Multiplayer Sync** | `viewState` + `watchEvents` | `ReactiveViewState` + `watch_events` | Complete |
| **Durable Documents** | `defineDoc` (`scope="conversation"`) | `define_doc` + `ReactiveDocumentState` | Complete |
| **Human Approvals** | Challenge tokens | `@approval_gate` + `ApprovalManager` | Complete |
| **Durable Sagas** | Parallel wait + abort rollback | `define_saga` + reverse compensation | Complete |

---

## 🧪 Testing & Verification

Every pull request is verified across unit test suites, and real AI provider evaluations:

```bash
# Run 148 unit & behavioral tests
uv run pytest

# Run the 9 core behavioral evaluations (high-fidelity fallback)
uv run python evals/run_evals.py

# Run evaluations with OpenRouter & GPT-6 Luna
OPENROUTER_API_KEY=sk-or-v1-... uv run python evals/run_evals.py

# Or specify model shortcuts directly
uv run python evals/run_evals.py --model "gpt-luna 6"
```

---

## License

[MIT](LICENSE). See [CONTRIBUTING](CONTRIBUTING.md), [SECURITY](SECURITY.md) and the [CHANGELOG](CHANGELOG.md).
