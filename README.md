# pydantic-ai-durable

**Crash-resilient, durable agent harness for [PydanticAI](https://github.com/pydantic/pydantic-ai).**  
Inspired by Earendil's [Pi Durable](https://github.com/earendil-works) architectural framework.

---

Traditional AI agent loops die when a process restarts, crashes, or hits an unhandled exception. State is lost, long-running tool runs disappear, and resuming requires either starting from scratch or re-executing expensive external side-effects.

`pydantic-ai-durable` turns PydanticAI agents into crash-proof, stateful state machines. Every model turn, tool call, context compaction, and multi-step business workflow is an atomic, checkpointed task backed by SQLite WAL (or zero-disk in-memory storage, or append-only JSONL ledgers).

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

## Highlights

- **Crash Resilience & Resumption:** Interrupted turns automatically resume on restart.
- **Durable Tool Replay:** Safe read-only tools re-execute automatically; mutating tools never repeat destructively and preserve partial output.
- **Hierarchical Tasks & Subagents:** Recursive task trees with bottom-up abort cascades and crash-safe subagents.
- **O(1) Transcript Branching:** Fork conversations at any message without copying history via content-addressed parent-pointer trees.
- **Durable Task Sagas:** Declarative multi-step workflows with parallel fan-out (`failFast` / `allSettled`) and automatic reverse-order compensation rollback.
- **Human Approval Gates:** Challenge tokens for sensitive operations with out-of-band resolution, parameter overrides, and dynamic deadlines.
- **Pluggable Storage:** Production SQLite WAL, zero-disk `MemoryStorage` for tests, and append-only `JsonlStorage` for audit ledgers.
- **Real-Time Multiplayer:** Observable reactive view states (`view_state`), presence rosters, and streaming agent event feeds.
- **Durable Documents:** Shared typed state (`define_doc`) mutated atomically alongside message transcripts.
- **Execution Environments:** Sandboxed workspaces (`LocalExecutionEnv`, `IsolatedSandboxEnv`) with path-traversal containment.

---

## Installation

```bash
pip install pydantic-ai-durable
```

Both package names are supported:
```python
import pydantic_ai_durable as pad
# or
import pya_durable as pad
```

---

## Quickstart

### 1. Minimal Durable Harness

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai_durable import DurableHarness, AioSqliteStorage

async def main():
    agent = Agent("openai:gpt-4o")

    async with AioSqliteStorage("agent.sqlite") as storage:
        # Create or resume harness
        harness = DurableHarness(
            agent=agent,
            storage=storage,
            conversation_id="conv-main",
        )

        # Run with idempotent request_id
        result = await harness.run(
            "Summarize the latest deployment status.",
            request_id="req-status-001",
        )
        print("Agent:", result.output)

asyncio.run(main())
```

---

### 2. Durable Tools with Replay Policies

Declare tool replay safety to protect mutating operations from re-executing after a crash:

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

### 3. Durable Task Sagas with Reverse Compensation

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
        # If this step raises an error, reserve_nodes compensation executes!
        ctx.state["cluster_ready"] = True
        return ctx.complete(result="Cluster active")

    async def compensate_reserve_nodes(self, ctx: SagaContext) -> None:
        # Releases reserved compute instances bottom-up
        print("Rollback: Releasing reserved nodes...")
```

---

### 4. Human Approval Gates & Deferred Execution

Pause turn execution for sensitive operations and wait for human operator intervention:

```python
from pydantic_ai_durable import approval_gate, DurableHarness

# Intercept sensitive actions and generate challenge token
@approval_gate(description="Approve database migration on production")
async def drop_production_table(table_name: str) -> str:
    return f"Dropped {table_name}"

# The agent turn pauses with TaskStatus.WAITING ("waiting_approval").
# External operator approves via API or TUI:
await harness.resolve_approval(
    token=challenge_token,
    approved=True,
    override_args={"table_name": "users_backup"},  # Optional parameter rewrite
)

# Resumes execution seamlessly!
```

---

### 5. Multi-Conversation Branching (Forking)

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

### 6. Real-Time Multiplayer Streaming & View State

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

## Comparison with Pi Durable (`@earendil-works/pi-durable`)

`pydantic-ai-durable` brings every architectural pillar from Earendil's TypeScript framework to Python and PydanticAI:

| Architectural Pillar | Pi Durable (TypeScript) | PyA Durable (Python) | Status |
| :--- | :--- | :--- | :---: |
| **Storage Engine** | Node SQLite WAL / DO | `AioSqliteStorage` (WAL mode) | Complete |
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

## Testing & Verification

Every pull request is verified across unit test suites, live swarm scenarios, and sub-millisecond performance probes:

```bash
# Run 129 unit tests
uv run pytest

# Run swarm verification lanes
uv run python scripts/verify_pr12.py
```

---

## License

MIT License. Designed with human-first clarity and pragmatism.
