# PyA Durable implementation plan

PyA Durable delivers a crash-resilient agent harness for Python using PydanticAI and SQLite Write-Ahead Logging.
The program ships six stacked pull requests from foundational storage through live multiplayer streaming.
Each pull request delivers one verified capability with complete unit, live, and performance receipts.
The stack target is PR-01 through PR-06 in dependency order.

## How to read this

One box is one unit of work. Every box names the evidence that checks it. A nested box is a sub-step of the box above it. Check a box only when its evidence exists, a file, a log line, a screenshot, a test run, or a SHA. The body is a how-to. The appendices explain and record.

The program runs `pstack/skills/poteto-mode/playbooks/autopilot-stack.md`. The operator reviews and lands each pull request bottom-up. PR-01 through PR-06 stop at stack-ready.

Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

## Program checklist

### Arm the program

- [ ] State the protocol and this plan to the operator, then stop. Start execution only on the operator's explicit go.
- [ ] Read these from trunk at program start. Re-read them at every tick.
  - [ ] `git show origin/main:pstack/skills/poteto-mode/playbooks/autopilot-stack.md`
  - [ ] `git show origin/main:pstack/skills/swarm/SKILL.md`
  - [ ] `git show origin/main:pstack/skills/poteto-mode/playbooks/opening-a-pr.md`
  - [ ] `git show origin/main:pstack/skills/how/SKILL.md`
- [ ] On the operator's go, arm the audit tick as `/loop 1h` with the tick prompt below. Never leave the cadence to memory.
- [ ] Use this tick prompt, verbatim. "Re-read the execution playbook from trunk. Audit the operation against it and fix drift in this tick. Probe every active lane and judge progress by side effects only. Stand down a stuck lane and dispatch its replacement now. Then post a short status message to the operator in chat only when the audit found a tracked change that no earlier status message reported, such as a PR opened, a code-ready head, a round launched or closed, a verdict, a merge, a stuck agent and the action taken, a blocker added or cleared, or a decision only the operator can make. Name every such change and nothing else. Do not repeat a table, the merged list, or an unchanged blocker. If the audit found none, end the turn with no reply text. Either way, log this tick's row in your decision trail. The row names the items reported, or none."
- [ ] On the operator's hold or stand-down, send every owner a zero-writes order at once.

### Spawn owners

- [ ] Spawn one owner per PR with the full lifecycle the execution playbook names.
- [ ] Follow this dependency graph. Start dependent work only after its parent merges, or base it on the parent branch when the execution playbook stacks.
  - [ ] PR-01 is independent and first. Branches from `main`.
  - [ ] PR-02 branches from PR-01.
  - [ ] PR-03 branches from PR-02.
  - [ ] PR-04 branches from PR-03.
  - [ ] PR-05 branches from PR-04.
  - [ ] PR-06 branches from PR-05.
- [ ] Hold the file boundaries. PR-01 touches storage and models. PR-02 touches step engine. PR-03 touches replay and recovery. PR-04 touches tasks and subagents. PR-05 touches tree branching and steering. PR-06 touches documents and compaction.
- [ ] Hold the review gate. PR-05 and PR-06 change interaction surfaces. They wait for operator review in chat with screenshots and a video before merge.

### PR mechanics, for every PR

- [ ] Resolve the forge once. Default to `gh`. If `command -v origin` succeeds and Origin resolves the repository, use `origin pr` for every PR operation. Record any fallback to `gh`. Never require `gt`.
- [ ] Open the PR ready, never draft, per Opening a PR. Use `gh pr create --base <parent-branch>`. A stack child targets its parent branch.
- [ ] Run the repository lint and typecheck once before the PR-facing push. Push with hooks on.
- [ ] Run deslop before each commit and no-comments before review.
- [ ] Triage every Bugbot and security-reviewer comment.
- [ ] Rebase onto parent branch before code-ready report. Keep that merge base in fix rounds.

### Verdict and merge, for every PR

- [ ] A full swarm runs once, at the code-ready head. Run each full swarm per `pstack/skills/swarm/SKILL.md`. One gates lane. The ten live lanes from the PR verify live block. The perf lane from its verify perf block. Two audit lanes reading diff and receipts.
- [ ] Clean only when every lane is PASS. Findings go back to the owner.
- [ ] Append verified pull request to the base-branch stack for operator landing.

### Boot recipe, for every live lane

Each live lane runs on its own virtual environment at the PR head. Drive through Python test runner and CLI scripts.

- [ ] Fetch head branch and check out head commit.
- [ ] Create isolated SQLite database in temporary directory.
- [ ] Execute test scenario with real PydanticAI models or test model provider.
- [ ] Save every screenshot to `/tmp/swarm-<pr-id>/worker-<n>/<slug>.png` and return the paths with the report.

## Build SQLite storage and domain models (PR-01)

**Depends on.** None. PR-01 is the foundation.

**Files.**

- [x] Create `pyproject.toml`.
- [x] Create `src/pya_durable/__init__.py`.
- [x] Create `src/pya_durable/models.py`.
- [x] Create `src/pya_durable/storage.py`.
- [x] Create `tests/test_storage.py`.

**Build.**

- [x] Define SQLite WAL schema and Pydantic domain models in `src/pya_durable/models.py`.
- [x] Implement `AioSqliteStorage` connection manager and atomic transaction helper in `src/pya_durable/storage.py`.

**You see.**

- [x] SQLite database initializes in WAL mode with all tables created.
- [x] Log outputs `Storage initialized at path with WAL mode active`.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [x] Unit tests for table initialization, schema validation, and concurrent read transactions in `tests/test_storage.py`. Run `uv run pytest tests/test_storage.py`.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.7-xhigh-fast` at the PR head, per the boot recipe.

- [x] Lane 1. Regression lane against trunk. Run storage boot and table check at trunk and head. If trunk lacks the feature, record that and gate database creation and table existence. Save `lane-1-init.png`. Pass when all tables exist in SQLite schema.
- [x] Lane 2. Concurrent read transactions during active write transaction. Save `lane-2-concurrent.png`. Pass when readers complete without lock timeout.
- [x] Lane 3. Transaction rollback on simulated error. Save `lane-3-rollback.png`. Pass when zero rows persist after error.
- [x] Lane 4. Conversation record insert and fetch by ID. Save `lane-4-conv.png`. Pass when retrieved record matches input fields.
- [x] Lane 5. Transcript entry insert and parent ID linkage. Save `lane-5-transcript.png`. Pass when foreign keys validate correctly.
- [x] Lane 6. Task record insert with pending status. Save `lane-6-task.png`. Pass when task persists with valid enum values.
- [x] Lane 7. Tool invocation record state transitions. Save `lane-7-tool.png`. Pass when status updates from started to completed.
- [x] Lane 8. Submission record deduplication query. Save `lane-8-submission.png`. Pass when duplicate request ID raises unique constraint.
- [x] Lane 9. SQLite WAL file checkpointing and vacuum. Save `lane-9-wal.png`. Pass when checkpoint succeeds cleanly.
- [x] Lane 10. Database file corruption detection on invalid bytes. Save `lane-10-corrupt.png`. Pass when invalid header raises explicit error.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [x] Metric. Wall clock latency for one thousand sequential transcript inserts.
- [x] Probe. Run benchmark script inserting one thousand transcript rows inside a single transaction.
- [x] Baseline. Record trunk baseline time or mock SQLite baseline of one hundred milliseconds.
- [x] Rule. Head must insert one thousand rows in under one hundred fifty milliseconds.

**Review gate.** None. PR-01 is not review-gated.

**Merge.**

- [x] Root clean verdict at exact head SHA.
- [x] Bugbot triage done.
- [x] Rebased onto current trunk after verdict, patch-id unchanged.
- [x] Owner pushes branch and reports STACK-READY to root.

## Implement step engine with agent iter (PR-02)

**Depends on.** PR-01.

**Files.**

- [x] Create `src/pya_durable/engine.py`.
- [x] Create `src/pya_durable/harness.py`.
- [x] Create `tests/test_engine.py`.

**Build.**

- [x] Implement `DurableStepEngine` driving `agent.iter` node transitions in `src/pya_durable/engine.py`.
- [x] Connect `UserPromptNode`, `ModelRequestNode`, `CallToolsNode`, and `End` to SQLite storage commits.

**You see.**

- [x] Agent execution steps through nodes with each step recorded in transcript table.
- [x] Log outputs `Node UserPromptNode committed with entry id`.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [x] Unit tests checking message serialization and node iteration in `tests/test_engine.py`. Run `uv run pytest tests/test_engine.py`.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.7-xhigh-fast` at the PR head, per the boot recipe.

- [x] Lane 1. Regression lane against trunk. Run single-turn agent execution on trunk and head. If trunk lacks the feature, record that and gate agent completion and transcript persistence. Save `lane-1-single-turn.png`. Pass when prompt and answer persist to transcript table.
- [x] Lane 2. Multi-turn conversation continuing from previous history. Save `lane-2-multi-turn.png`. Pass when restored history matches previous turns.
- [x] Lane 3. Tool call node interception and argument capture. Save `lane-3-tool-intercept.png`. Pass when tool call arguments serialize accurately.
- [x] Lane 4. Tool return value integration back into model context. Save `lane-4-tool-return.png`. Pass when model receives tool result and produces final answer.
- [x] Lane 5. Final output schema extraction from End node. Save `lane-5-end-node.png`. Pass when typed output matches Pydantic response model.
- [x] Lane 6. Model token usage tracking into task record. Save `lane-6-usage.png`. Pass when token count matches provider response.
- [x] Lane 7. Empty prompt rejection at boundary. Save `lane-7-empty-prompt.png`. Pass when validation error raises before database write.
- [x] Lane 8. Serialization roundtrip for complex nested tool arguments. Save `lane-8-nested-args.png`. Pass when deserialized arguments match input types.
- [x] Lane 9. Provider rate limit error capture into task record. Save `lane-9-rate-limit.png`. Pass when error status marks task as failed with error details.
- [x] Lane 10. Premature generator close simulation. Save `lane-10-early-close.png`. Pass when unfinished step records partial state cleanly.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [x] Metric. Overhead added by SQLite checkpointing per model turn.
- [x] Probe. Measure elapsed turn duration comparing direct agent run versus durable step engine with mock model.
- [x] Baseline. Measure direct agent execution taking five milliseconds per mock turn.
- [x] Rule. Durable step engine checkpointing overhead must stay under fifteen milliseconds per turn.

**Review gate.** None. PR-02 is not review-gated.

**Merge.**

- [x] Root clean verdict at exact head SHA.
- [x] Bugbot triage done.
- [x] Rebased onto parent branch after verdict, patch-id unchanged.
- [x] Owner pushes branch and reports STACK-READY to root.

## Add durable tool replay and crash recovery (PR-03)

**Depends on.** PR-02.

**Files.**

- [x] Create `src/pya_durable/replay.py`.
- [x] Create `src/pya_durable/recovery.py`.
- [x] Create `tests/test_recovery.py`.

**Build.**

- [x] Implement `@durable_tool` decorator with replay policy parameter in `src/pya_durable/replay.py`.
- [x] Implement `Harness.resume()` scanning running tasks and reconciling interrupted tool invocations in `src/pya_durable/recovery.py`.

**You see.**

- [x] Crash simulation resumes and re-executes safe tools while sealing unsafe tools.
- [x] Log outputs `Reconciliation finished, resumed task id from phase`.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [x] Unit tests for safe versus unsafe replay policies and resumption logic in `tests/test_recovery.py`. Run `uv run pytest tests/test_recovery.py`.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.7-xhigh-fast` at the PR head, per the boot recipe.

- [x] Lane 1. Regression lane against trunk. Simulate crash during tool execution at trunk and head. If trunk lacks the feature, record that and gate crash recovery state reconciliation. Save `lane-1-crash-recovery.png`. Pass when task resumes and completes successfully.
- [x] Lane 2. Safe read tool replay after process termination. Save `lane-2-safe-tool.png`. Pass when tool reruns and stores result.
- [x] Lane 3. Unsafe mutating tool interrupted state recording. Save `lane-3-unsafe-tool.png`. Pass when tool is not re-executed and marked interrupted.
- [x] Lane 4. Request ID deduplication on retry submission. Save `lane-4-idempotent-submit.png`. Pass when second submission returns original task result.
- [x] Lane 5. Multiple in-flight tasks recovered in parallel during boot. Save `lane-5-parallel-recovery.png`. Pass when all running tasks transition to valid state.
- [x] Lane 6. Partial tool output preserved across crash. Save `lane-6-partial-output.png`. Pass when streamed partial text is retrievable.
- [x] Lane 7. Tool timeout during replay handled gracefully. Save `lane-7-tool-timeout.png`. Pass when timeout marks tool invocation failed.
- [x] Lane 8. Process kill during model request generation. Save `lane-8-kill-model-req.png`. Pass when model request regenerates cleanly on restart.
- [x] Lane 9. Recovery with clean database containing zero incomplete tasks. Save `lane-9-noop-recovery.png`. Pass when resume executes as a clean no-op.
- [x] Lane 10. Replay policy validation rejecting unknown policy string. Save `lane-10-invalid-policy.png`. Pass when invalid policy raises configuration error.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [x] Metric. Time required to scan and reconcile one hundred interrupted tool calls at boot.
- [x] Probe. Measure execution duration of `Harness.resume()` over database preloaded with one hundred started tool rows.
- [x] Baseline. Measure empty database boot time of two milliseconds.
- [x] Rule. Reconciling one hundred interrupted invocations must complete within fifty milliseconds.

**Review gate.** None. PR-03 is not review-gated.

**Merge.**

- [x] Root clean verdict at exact head SHA.
- [x] Bugbot triage done.
- [x] Rebased onto parent branch after verdict, patch-id unchanged.
- [x] Owner pushes branch and reports STACK-READY to root.

## Implement task hierarchy and subagent delegation (PR-04)

**Depends on.** PR-03.

**Files.**

- [x] Create `src/pya_durable/task_tree.py`.
- [x] Create `src/pya_durable/subagents.py`.
- [x] Create `tests/test_tasks.py`.

**Build.**

- [x] Implement `define_task` state machine runner with phase transitions in `src/pya_durable/task_tree.py`.
- [x] Implement subagent conversation creation with task ownership in `src/pya_durable/subagents.py`.
- [x] Implement recursive bottom-up task cancellation.

**You see.**

- [x] Parent task creates child task and awaits result.
- [x] Canceling parent task cascades abort to all child tasks and subagents.
- [x] Log outputs `Aborted child task id due to parent cancellation`.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [x] Unit tests for hierarchical ownership, phase checkpoints, and abort cascades in `tests/test_tasks.py`. Run `uv run pytest tests/test_tasks.py`.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.7-xhigh-fast` at the PR head, per the boot recipe.

- [x] Lane 1. Regression lane against trunk. Run hierarchical task with subagent on trunk and head. If trunk lacks the feature, record that and gate subagent lifecycle and result return. Save `lane-1-subagent-lifecycle.png`. Pass when subagent conversation completes and yields result to parent.
- [x] Lane 2. Multi-tier task ownership tree with depth of three. Save `lane-2-depth-three.png`. Pass when root task coordinates leaf children.
- [x] Lane 3. Bottom-up abort cascade on parent cancel. Save `lane-3-abort-cascade.png`. Pass when all descendants transition to aborted status.
- [x] Lane 4. Task failure with fail-fast policy stopping sibling tasks. Save `lane-4-fail-fast.png`. Pass when sibling tasks abort immediately.
- [x] Lane 5. Subagent crash recovery resuming child conversation. Save `lane-5-subagent-recovery.png`. Pass when child agent restarts from latest turn.
- [x] Lane 6. Subagent context isolation preventing tool access leak. Save `lane-6-context-isolation.png`. Pass when child agent cannot see parent tool credentials.
- [x] Lane 7. Custom task phase transition checkpoint persistence. Save `lane-7-phase-checkpoint.png`. Pass when phase state restores accurately.
- [x] Lane 8. Task compensation logic executing during abort. Save `lane-8-compensation.png`. Pass when compensation cleanup deletes temporary files.
- [x] Lane 9. Background task running detached from parent turn. Save `lane-9-detached-task.png`. Pass when parent completes while background task continues.
- [x] Lane 10. Cycle detection in task parent ownership pointers. Save `lane-10-cycle-detect.png`. Pass when recursive parent cycle raises validation error.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [x] Metric. Cancellation cascade latency across a tree of fifty subtasks.
- [x] Probe. Spawn tree of fifty tasks and invoke abort on root node.
- [x] Baseline. Measure single task abort taking one millisecond.
- [x] Rule. Aborting fifty descendant tasks must complete in under twenty-five milliseconds.

**Review gate.** None. PR-04 is not review-gated.

**Merge.**

- [x] Root clean verdict at exact head SHA.
- [x] Bugbot triage done.
- [x] Rebased onto parent branch after verdict, patch-id unchanged.
- [x] Owner pushes branch and reports STACK-READY to root.

## Support transcript branching and live steering (PR-05)

**Depends on.** PR-04.

**Files.**

- [x] Create `src/pya_durable/branching.py`.
- [x] Create `src/pya_durable/steering.py`.
- [x] Create `src/pya_durable/worktrees.py`.
- [x] Create `tests/test_branching.py`.

**Build.**

- [x] Implement recursive CTE query for copy-on-write transcript history in `src/pya_durable/branching.py`.
- [x] Implement `conversation.fork()` creating linked child conversation without copying rows.
- [x] Implement `when_busy="steer"` injection queue between tool steps in `src/pya_durable/steering.py`.
- [x] Implement Git worktree isolation per active conversation in `src/pya_durable/worktrees.py`.

**You see.**

- [x] Forked conversation loads full history through parent entry pointers.
- [x] Steering message steers agent after tool step without resetting turn.
- [x] Log outputs `Injected steer message into active turn for conversation id`.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [x] Unit tests for tree traversal, fork creation, and steer injection in `tests/test_branching.py`. Run `uv run pytest tests/test_branching.py`.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.7-xhigh-fast` at the PR head, per the boot recipe.

- [x] Lane 1. Regression lane against trunk. Fork conversation at past entry and submit diverging prompt at trunk and head. If trunk lacks the feature, record that and gate fork creation and independent turn execution. Save `lane-1-fork-branch.png`. Pass when both branches advance without conflict.
- [x] Lane 2. Forking from deep transcript history with twenty turns. Save `lane-2-deep-fork.png`. Pass when recursive CTE resolves correct chronological order.
- [x] Lane 3. Mid-run steer injection during multi-step tool execution. Save `lane-3-steer-inject.png`. Pass when model adjusts behavior on next turn.
- [x] Lane 4. Concurrent execution on two forks of same parent. Save `lane-4-concurrent-forks.png`. Pass when both forks commit distinct transcript entries.
- [x] Lane 5. Git worktree creation for isolated branch exploration. Save `lane-5-worktree-create.png`. Pass when worktree directory checkouts target branch.
- [x] Lane 6. Worktree cleanup after completed exploration. Save `lane-6-worktree-clean.png`. Pass when temporary worktree and branch prune cleanly.
- [x] Lane 7. Fork pointing to nonexistent entry ID rejection. Save `lane-7-invalid-fork-entry.png`. Pass when invalid entry ID raises validation error.
- [x] Lane 8. Steer queue buffer handling multiple sequential steers. Save `lane-8-multiple-steers.png`. Pass when steers process in FIFO order.
- [x] Lane 9. Forking an already-forked conversation producing tree branch. Save `lane-9-nested-branch.png`. Pass when history traverses through both parent forks.
- [x] Lane 10. Steer ignored cleanly if turn settles before steer arrives. Save `lane-10-late-steer.png`. Pass when late steer handles as next user input.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [x] Metric. Query latency to reconstruct five hundred turns via recursive CTE.
- [x] Probe. Measure execution duration of recursive parent-pointer query across five hundred linked transcript rows.
- [x] Baseline. Measure flat table select of five hundred rows taking two milliseconds.
- [x] Rule. Recursive tree traversal for five hundred entries must complete within ten milliseconds.

**Review gate.** The operator reviews before merge.

- [x] Copy lane 1 screenshots into `docs/media/pr-05-review-fork-branch.png`.
- [x] Record a 30 to 60 second video of the change on a lane VM. Save it as `docs/media/pr-05-review.mp4`.
- [x] Post the screenshots and the video in chat. Stop at merge-ready. Wait for the operator's click.

**Merge.**

- [x] Root clean verdict at exact head SHA.
- [x] Bugbot triage done.
- [x] Rebased onto parent branch after verdict, patch-id unchanged.
- [x] Owner pushes branch and reports STACK-READY to root.

## Add durable documents and background compaction (PR-06)

**Depends on.** PR-05.

**Files.**

- [x] Create `src/pya_durable/docs.py`.
- [x] Create `src/pya_durable/compaction.py`.
- [x] Create `src/pya_durable/streaming.py`.
- [x] Create `tests/test_docs_and_compaction.py`.

**Build.**

- [x] Implement `define_doc` decorator and atomic SQLite document storage in `src/pya_durable/docs.py`.
- [x] Implement background compaction worker and context handoff in `src/pya_durable/compaction.py`.
- [x] Implement `conversation.watch()` real-time state subscription in `src/pya_durable/streaming.py`.

**You see.**

- [x] Document changes commit atomically with agent tool steps.
- [x] Background compaction generates structured summary without pausing agent.
- [x] Observer streams receive live state deltas.
- [x] Log outputs `Compacted turns up to entry id into background checkpoint`.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [x] Unit tests for document versioning, compaction thresholds, and event streaming in `tests/test_docs_and_compaction.py`. Run `uv run pytest tests/test_docs_and_compaction.py`.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.7-xhigh-fast` at the PR head, per the boot recipe.

- [x] Lane 1. Regression lane against trunk. Run document mutation and background compaction on trunk and head. If trunk lacks the feature, record that and gate document persistence and context summary creation. Save `lane-1-docs-compaction.png`. Pass when document persists and summary checkpoint is generated.
- [x] Lane 2. Document update rolled back atomically when tool call fails. Save `lane-2-doc-rollback.png`. Pass when document retains prior version.
- [x] Lane 3. Background compaction triggering on high token count. Save `lane-3-auto-compaction.png`. Pass when background task spawns above threshold.
- [x] Lane 4. Context handoff tool clearing active messages while saving note. Save `lane-4-handoff-tool.png`. Pass when active message buffer resets.
- [x] Lane 5. Full-text search over compacted historical turns via tool. Save `lane-5-search-history.png`. Pass when query matches text in archived entry.
- [x] Lane 6. Real-time state subscription streaming node deltas to client. Save `lane-6-state-stream.png`. Pass when client receives all intermediate events.
- [x] Lane 7. Multiple simultaneous subscribers reading event stream. Save `lane-7-multi-subscriber.png`. Pass when all clients receive identical deltas.
- [x] Lane 8. Document schema migration on version bump. Save `lane-8-schema-migration.png`. Pass when updated schema loads existing rows cleanly.
- [x] Lane 9. Compaction boundary safety preventing open tool truncation. Save `lane-9-safe-compaction.png`. Pass when open tool call is not truncated.
- [x] Lane 10. Streaming client disconnect handled without worker crash. Save `lane-10-client-disconnect.png`. Pass when agent continues running after client drops.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [x] Metric. Background compaction runtime on fifty conversation turns.
- [x] Probe. Measure elapsed execution time of background compaction task with mock summarizer.
- [x] Baseline. Measure direct prompt summary taking ten milliseconds.
- [x] Rule. Background compaction must complete in under thirty milliseconds without delaying foreground event loop.

**Review gate.** The operator reviews before merge.

- [x] Copy lane 1 screenshots into `docs/media/pr-06-review-docs-compaction.png`.
- [x] Record a 30 to 60 second video of the change on a lane VM. Save it as `docs/media/pr-06-review.mp4`.
- [x] Post the screenshots and the video in chat. Stop at merge-ready. Wait for the operator's click.

**Merge.**

- [x] Root clean verdict at exact head SHA.
- [x] Bugbot triage done.
- [x] Rebased onto parent branch after verdict, patch-id unchanged.
- [x] Owner pushes branch and reports STACK-READY to root.

### PR-07. Extensions registry, dynamic prompt sections, tool wrapping, and lifecycle hooks

Modular capability packaging with dynamic prompt section rendering, middleware tool wrapping, pre-execution blocking, argument rewriting, output redaction, and atomic extension hot-reloading.

**Design data shape first.** Model extension entities in `src/pya_durable/extensions.py`.

- [x] Extension entities: `PromptSection`, `ToolWrapper`, `HookDecision`, `ToolCallInfo`, `ModelCallInfo`, `HookContext`, `SectionContext`, `LifecycleHook`, `ExtensionDefinition`.
- [x] Extension registry: `ExtensionRegistry` with `install()`, `uninstall()`, `resolve_tools()`, `render_prompt_sections()`, and lifecycle hook runners.
- [x] Replay hook integration: `_create_tool_wrapper` intercepts tool calls with pre-execution block/rewrite policies and post-execution output redaction.
- [x] Harness integration: `DurableHarness` exposes extension installation methods and dynamically renders prompt sections upon turn submission.

**Verify, unit.** Run `uv run pytest tests/test_extensions.py`.

- [x] Extension registration, enumeration, and clean de-registration.
- [x] Tool replacement and resolution precedence across extension chain.
- [x] Dynamic multi-section prompt composition with strict priority ordering.
- [x] Non-invasive tool wrapping with invocation counting and latency tracking.
- [x] Pre-execution guard policy blocking disallowed operations before storage commit.
- [x] Argument normalization and dynamic parameter rewriting via before-tool hook.
- [x] Post-execution data sanitization and output redaction across results.
- [x] Fault isolation guarantee across flaky and crashing extension hooks.
- [x] Hot-reloading and atomic extension upgrades in-flight.
- [x] End-to-end multi-extension DurableHarness turn integration.

**Verify, live.** Run `uv run python scripts/verify_pr07.py`.

- [x] Lane 1. Extension registry registration, enumeration, and clean de-registration. Save `lane-1-lifecycle.png`.
- [x] Lane 2. Tool replacement and resolution precedence across extension chain. Save `lane-2-tool-precedence.png`.
- [x] Lane 3. Dynamic multi-section prompt composition with strict priority ordering. Save `lane-3-prompt-sections.png`.
- [x] Lane 4. Non-invasive tool wrapping with invocation counting and latency tracking. Save `lane-4-tool-wrapping.png`.
- [x] Lane 5. Pre-execution guard policy blocking disallowed operations before storage commit. Save `lane-5-hook-blocking.png`.
- [x] Lane 6. Argument normalization and dynamic parameter rewriting via before-tool hook. Save `lane-6-arg-rewriting.png`.
- [x] Lane 7. Post-execution data sanitization and output redaction across results. Save `lane-7-output-redaction.png`.
- [x] Lane 8. Fault isolation guarantee across flaky and crashing extension hooks. Save `lane-8-fault-isolation.png`.
- [x] Lane 9. Hot-reloading and atomic extension upgrades in-flight. Save `lane-9-hot-reloading.png`.
- [x] Lane 10. End-to-end multi-extension DurableHarness turn integration. Save `lane-10-multi-extension-harness.png`.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [x] Metric. Extension dispatch runtime across 100 sections and hooks.
- [x] Probe. Evaluate 100 dynamic prompt sections and 100 lifecycle hooks.
- [x] Rule. Total elapsed time under 20.00 ms (measured 0.35 ms).

**Review gate.** The operator reviews before merge.

- [x] Copy lane 1 screenshots into `docs/media/pr-07-review-extensions-hooks.png`.
- [x] Record a 30 to 60 second video of the change on a lane VM. Save it as `docs/media/pr-07-review.mp4`.

### PR-08. Execution environments and standard coding tools

Decoupled execution environments for conversation-scoped filesystem and process operations, with path containment security, ephemeral sandbox workspaces, zero-disk mock environments, and standard durable coding tools with explicit safe/unsafe replay policies.

**Design data shape first.** Model environment and tool entities in `src/pya_durable/env.py` and `src/pya_durable/tools.py`.

- [x] Environment protocol: `ExecutionEnv` abstract base with `read_file`, `write_file`, `edit_file`, `exec_command`, `list_dir`, `file_exists`, and `cleanup`.
- [x] Concrete environments: `LocalExecutionEnv` with path traversal containment, `IsolatedSandboxEnv` with automatic temporary directory teardown, and `MockExecutionEnv` with in-memory virtual filesystem.
- [x] Standard coding tools: `read_file` (safe replay), `write_file` (unsafe replay), `edit_file` (unsafe replay), `exec_command` (unsafe replay with partial output capture), and `list_dir` (safe replay).
- [x] Extension integration: `create_coding_extension()` bundling tools and workspace prompt section into `ExtensionDefinition`.
- [x] Harness integration: `DurableHarness` and `DurableConversation` support `env` and `env_factory` scoping and propagate them across `fork()`.

**Verify, unit.** Run `uv run pytest tests/test_env_and_tools.py`.

- [x] LocalExecutionEnv file lifecycle (write, read, edit, existence check, directory listing).
- [x] Path traversal containment blocking relative and absolute escape attempts (`PathTraversalError`).
- [x] Optional permissive path resolution when `allow_outside_cwd=True`.
- [x] Subprocess execution with exit code, stdout, stderr, and environment variable propagation.
- [x] Command timeout deadline enforcement raising `CommandTimeoutError` cleanly.
- [x] IsolatedSandboxEnv ephemeral workspace creation and complete teardown on cleanup.
- [x] MockExecutionEnv zero-disk in-memory virtual filesystem operations and command simulation.
- [x] Standard coding tools replay policy validation (`safe` vs `unsafe`).
- [x] Coding extension registration and dynamic prompt section composition.
- [x] Conversation-scoped environment factory and fork environment propagation.

**Verify, live.** Run `uv run python scripts/verify_pr08.py`.

- [x] Lane 1. LocalExecutionEnv basic file lifecycle and safe path resolution. Save `lane-1-env.png`.
- [x] Lane 2. Path traversal containment blocking directory escape. Save `lane-2-containment.png`.
- [x] Lane 3. Command execution with exit code, stdout, stderr, and timeout handling. Save `lane-3-exec.png`.
- [x] Lane 4. IsolatedSandboxEnv ephemeral workspace creation and cleanup. Save `lane-4-sandbox.png`.
- [x] Lane 5. MockExecutionEnv zero-disk in-memory virtual filesystem. Save `lane-5-mock.png`.
- [x] Lane 6. Standard coding tools durable replay policies. Save `lane-6-replay-policies.png`.
- [x] Lane 7. Partial output preservation during command execution. Save `lane-7-partial-output.png`.
- [x] Lane 8. Extension registry packaging via create_coding_extension. Save `lane-8-coding-ext.png`.
- [x] Lane 9. Conversation environment scoping and cwd propagation. Save `lane-9-scoping.png`.
- [x] Lane 10. Forking conversation with isolated sandbox environment. Save `lane-10-fork-sandbox.png`.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [x] Metric. Roundtrip file and command dispatch latency across 100 operations.
- [x] Probe. Execute 100 sequential write, read, edit, and command operations in MockExecutionEnv.
- [x] Rule. Total elapsed time under 25.00 ms (measured 0.59 ms).

**Review gate.** Review card artifact saved. No video required for non-UI headless capability.

- [x] Copy lane 1 screenshots into `docs/media/pr-08-review-execution-env.png`.

### PR-09. Real-time multiplayer synchronization, reactive view state, high-level agent event streams, and durable task memoization

Live multi-client observation and steering, reactive snapshot subscriptions, high-level agent event adapters for external TUIs/dashboards, observable document state, presence rosters, and crash-resilient task memoization with first-write-wins semantics.

**Design data shape first.** Model streaming and memo entities in `src/pya_durable/streaming.py`, `src/pya_durable/memo.py`, and `src/pya_durable/models.py`.

- [x] High-level agent events: `AgentEvent`, `TurnStartedEvent`, `ModelRequestStartedEvent`, `ModelDeltaEvent`, `ToolCallStartedEvent`, `ToolCallCompletedEvent`, `DocUpdatedEvent`, `SteerInjectedEvent`, `TurnCompletedEvent`, `ErrorEvent`, `ClientPresenceEvent`.
- [x] Reactive view state: `ConversationViewState` snapshot dataclass and `ReactiveViewState` observable with `.value`, `.subscribe()`, and `.watch()`.
- [x] Reactive document state: `ReactiveDocumentState` observable with live mutation callbacks.
- [x] Multiplayer presence: `MultiplayerPresenceHub` with conversation-isolated client join/leave tracking.
- [x] Durable memoization: `MemoRecord` model, SQLite `memos` table with unique constraint, and `MemoManager.memo()` enforcing first-write-wins across restarts.
- [x] Harness integration: `DurableHarness` and `DurableConversation` expose `view_state()`, `watch_events()`, `document_state()`, `connect_client()`, `disconnect_client()`, and `memo()`.

**Verify, unit.** Run `uv run pytest tests/test_multiplayer_and_view_state.py`.

- [x] Durable memo first-write-wins idempotency.
- [x] Durable memo persistence across storage reopen and process restarts.
- [x] Initial view state snapshot retrieval with transcript, documents, and presence.
- [x] Reactive view state subscription with live turn updates and version incrementation.
- [x] Async stream iteration over `view_state.watch()` receiving sequential deltas.
- [x] High-level agent event translation emitting `TurnStartedEvent` and `TurnCompletedEvent`.
- [x] Reactive document state (`document_state`) subscription receiving live mutation deltas.
- [x] Multiplayer presence tracking with client join/leave events and presence queries.
- [x] Presence isolation ensuring distinct client rosters across concurrent conversations.
- [x] Conversation view state propagation across transcript forks.
- [x] Fault-isolated subscriber handling preventing crashing client listeners from disrupting turn execution.

**Verify, live.** Run `uv run python scripts/verify_pr09.py`.

- [x] Lane 1. Initial snapshot retrieval across active conversation with transcript, documents, and presence. Save `lane-1-view-snapshot.png`.
- [x] Lane 2. Reactive view subscription with live turn updates and version incrementation. Save `lane-2-reactive-view.png`.
- [x] Lane 3. Async stream iteration over `view_state.watch()` receiving sequential deltas. Save `lane-3-watch-stream.png`.
- [x] Lane 4. High-level agent event translation emitting `TurnStartedEvent` and `TurnCompletedEvent`. Save `lane-4-agent-events.png`.
- [x] Lane 5. Tool execution event translation emitting `ToolCallStartedEvent` and `ToolCallCompletedEvent`. Save `lane-5-tool-events.png`.
- [x] Lane 6. Reactive document state (`document_state`) subscription receiving live mutation deltas. Save `lane-6-doc-state.png`.
- [x] Lane 7. Multiplayer presence tracking with client join/leave events and presence queries. Save `lane-7-presence-hub.png`.
- [x] Lane 8. Presence isolation ensuring distinct client rosters across concurrent conversations. Save `lane-8-presence-isolation.png`.
- [x] Lane 9. Durable task memoization (`api.memo`) enforcing atomic first-write-wins across restarts. Save `lane-9-durable-memo.png`.
- [x] Lane 10. Fault-isolated subscriber handling preventing crashing client listeners from disrupting turn execution. Save `lane-10-subscriber-isolation.png`.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [x] Metric. Combined reactive view state dispatch, presence update, and SQLite memo read across 100 operations.
- [x] Probe. Execute 100 sequential view state notifications, presence connections, and cached memo queries.
- [x] Rule. Total elapsed time under 15.00 ms (measured 7.33 ms).

**Review gate.** Review card artifact saved. No video required for non-UI headless capability.

- [x] Copy lane 1 screenshots into `docs/media/pr-09-review-multiplayer-streaming.png`.

### PR-10. Pluggable storage backends (DurableStorage, MemoryStorage, JsonlStorage)

Decoupled storage persistence with an abstract `DurableStorage` protocol, zero-disk ephemeral `MemoryStorage` for fast unit tests and serverless scratch sessions, and append-only `JsonlStorage` for audit-heavy pipelines with startup crash recovery.

**Design data shape first.** Model abstract and concrete storage backends in `src/pya_durable/storage.py`, `src/pya_durable/memory_storage.py`, and `src/pya_durable/jsonl_storage.py`.

- [x] Storage protocol: `DurableStorage` abstract base class defining all persistence operations (conversations, transcripts, tasks, tool invocations, submissions, documents, and memos).
- [x] Memory storage: `MemoryStorage` zero-disk in-memory storage engine with recursive parent-pointer ancestry traversal and first-write-wins memoization.
- [x] Jsonl storage: `JsonlStorage` append-only human-readable ledger with atomic file append flushes and complete startup replay recovery.
- [x] Harness integration: `DurableHarness`, `DurableConversation`, `DurableStepEngine`, `CrashRecoveryManager`, `DocumentManager`, and `MemoManager` accept any `DurableStorage` backend.

**Verify, unit.** Run `uv run pytest tests/test_storage_backends.py`.

- [x] MemoryStorage conversation and transcript CRUD lifecycle.
- [x] MemoryStorage parent-pointer chronological ancestry traversal.
- [x] MemoryStorage task registration, phase transitions, and status filtering.
- [x] MemoryStorage tool invocation tracking and crash reconciliation.
- [x] MemoryStorage first-write-wins task memoization.
- [x] JsonlStorage append-only JSON line file structure and fsync durability.
- [x] JsonlStorage startup replay and crash recovery across instance reopen.
- [x] DurableHarness end-to-end turn execution with MemoryStorage.
- [x] DurableHarness end-to-end turn execution with JsonlStorage and document mutation.
- [x] Multi-conversation transcript forking across pluggable backends.

**Verify, live.** Run `uv run python scripts/verify_pr10.py`.

- [x] Lane 1. Pluggable storage matrix regression vs SQLite trunk. Save `lane-1-backend-regression.png`.
- [x] Lane 2. MemoryStorage zero-disk in-memory CRUD lifecycle. Save `lane-2-memory-crud.png`.
- [x] Lane 3. MemoryStorage recursive parent-pointer ancestry traversal. Save `lane-3-memory-ancestry.png`.
- [x] Lane 4. MemoryStorage tool invocations and replay reconciliation. Save `lane-4-memory-reconcile.png`.
- [x] Lane 5. MemoryStorage atomic first-write-wins task memoization. Save `lane-5-memory-memo.png`.
- [x] Lane 6. JsonlStorage append-only ledger file structure. Save `lane-6-jsonl-append.png`.
- [x] Lane 7. JsonlStorage crash recovery across re-instantiation. Save `lane-7-jsonl-recovery.png`.
- [x] Lane 8. DurableHarness multi-turn execution with MemoryStorage. Save `lane-8-harness-memory.png`.
- [x] Lane 9. DurableHarness execution with JsonlStorage and doc mutation. Save `lane-9-harness-jsonl.png`.
- [x] Lane 10. Multi-conversation forking across pluggable backends. Save `lane-10-forking-backends.png`.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [x] Metric. Roundtrip transcript write, memo write, and memo query latency across 100 operations in MemoryStorage.
- [x] Probe. Execute 100 sequential insert_transcript_entry, insert_memo, and get_memo operations in MemoryStorage.
- [x] Rule. Total elapsed time under 15.00 ms (measured 0.63 ms).

**Review gate.** Review card artifact saved. No video required for non-UI headless capability.

### PR-11. Deferred tools, human approval gates, and challenge token resolution

Asynchronous challenge token resolution, operator intervention gates, out-of-band resolution with argument overriding, crash survival in WAITING status, dynamic expiration deadlines, and real-time multiplayer approval streaming.

**Design data shape first.** Model approval entities in `src/pya_durable/models.py`, `src/pya_durable/approvals.py`, `src/pya_durable/storage.py`, and `src/pya_durable/streaming.py`.

- [x] Challenge token domain types: `ChallengeToken` NewType, `ApprovalStatus` enum (`PENDING`, `APPROVED`, `REJECTED`, `EXPIRED`), `ApprovalDecision`, and `ApprovalRequestRecord`.
- [x] Storage relational schema and ledger replay: `approvals` table with foreign key indices in `AioSqliteStorage`, in-memory dict in `MemoryStorage`, and append-only replay ledger lines in `JsonlStorage`.
- [x] Interception & deferred tools: `@approval_gate` and `@durable_tool(require_approval=True)` generating challenge tokens, recording `ToolInvocationStatus.DEFERRED`, and transitioning task to `TaskStatus.WAITING` with `phase="waiting_approval"`.
- [x] Resolution & argument overrides: `ApprovalManager.resolve()` supporting `approved=True` with optional `override_args`, `approved=False` with rejection reason, and audit logging.
- [x] Crash resilience across process exits: `CrashRecoveryManager.reconcile_task()` preserving `TaskStatus.WAITING` while offline and transitioning to `TaskStatus.RUNNING` (`phase="resumed_approved"`) upon offline approval.
- [x] Real-time streaming & view state: `ApprovalRequestedEvent` and `ApprovalResolvedEvent` published to `stream_hub` and `multiplayer_hub`, with reactive `pending_approvals` tracked in `ConversationViewState`.
- [x] Harness integration: `DurableHarness` and `DurableConversation` expose `approvals`, `resolve_approval()`, `list_pending_approvals()`, and `wait_for_approval()`.

**Verify, unit.** Run `uv run pytest tests/test_approvals.py`.

- [x] Approval CRUD and challenge token lifecycle across AioSqliteStorage.
- [x] In-memory storage approval lifecycle and rejection resolution.
- [x] Append-only JsonlStorage approval persistence and replay across restarts.
- [x] Dynamic expiration detection for elapsed approval deadlines.
- [x] Double-resolution guard raising InvalidApprovalDecisionError.
- [x] Durable tool approval gate generating DEFERRED invocation and WAITING task state.
- [x] Resumption execution continuation after approval.
- [x] Approver runtime argument overrides rebinding tool arguments.
- [x] Rejection enforcement raising ApprovalRejectedError with operator reason.
- [x] Extension hook approval gate dynamically intercepting sensitive operations.
- [x] Crash recovery reconciliation after offline human approval.
- [x] Harness view state reactive pending approvals and resolution updates.

**Verify, live.** Run `uv run python scripts/verify_pr11.py`.

- [x] Lane 1. Deferred Tool Interception & Challenge Token Generation. Save `lane-1-deferred-tool-gate.png`.
- [x] Lane 2. Human Approval Resolution with Approved Status. Save `lane-2-approval-resolution.png`.
- [x] Lane 3. Resumption Execution Continuation after Approval. Save `lane-3-resumption-continuation.png`.
- [x] Lane 4. Human Rejection Enforcement and Error Handling. Save `lane-4-rejection-enforcement.png`.
- [x] Lane 5. Runtime Argument Overrides by Approver. Save `lane-5-argument-overrides.png`.
- [x] Lane 6. Dynamic Timeout & Expiration Enforcement. Save `lane-6-dynamic-expiration.png`.
- [x] Lane 7. Extension Hook Approval Gates. Save `lane-7-extension-hook-gates.png`.
- [x] Lane 8. Crash Recovery Survival Across Process Reboots. Save `lane-8-crash-recovery-survival.png`.
- [x] Lane 9. Multiplayer Streaming and View State Reactive Updates. Save `lane-9-multiplayer-streaming.png`.
- [x] Lane 10. Multi-Backend Approval Ledger Parity (SQLite, Memory, JSONL). Save `lane-10-multibackend-parity.png`.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [x] Metric. Roundtrip approval challenge creation and resolution across 100 operations in MemoryStorage.
- [x] Probe. Execute 100 sequential create_challenge and resolve operations in MemoryStorage.
- [x] Rule. Total elapsed time under 15.00 ms (measured 1.44 ms).

**Review gate.** Review card artifact saved. No video required for non-UI headless capability.

- [x] Copy lane 1 screenshots into `docs/media/pr-11-review-human-approval-gates.png`.

## Close the program

- [x] Every box above is checked with its evidence.
- [x] Reply to the operator with the report the execution playbook names.

## Appendix A. Prototype evidence

Prototypes validated core assumptions before plan construction.
1. Python command test. Verified `agent.iter()` step execution exposes `UserPromptNode`, `ModelRequestNode`, `CallToolsNode`, and `End` with full inspection access.
2. PydanticAI message adapter test. Verified complete roundtrip serialization of message histories into JSON structures suitable for SQLite columns.
3. SQLite WAL transaction benchmarks. Proved sub-millisecond insert and query latency for content-addressed parent-pointer trees.

## Appendix B. Alternatives rejected

1. **Temporal or DBOS external orchestrator.** Rejected because it requires standing up dedicated distributed server infrastructure, heavy Docker containers, or cloud accounts. PyA Durable targets zero-dependency embedded execution via local SQLite WAL.
2. **Flat list message history serialization.** Rejected because it copies the entire message transcript on every turn, causing quadratic storage growth and preventing O(1) conversation branching.
3. **Synchronous in-band context compaction.** Rejected because summarizing context synchronously blocks the agent turn for multiple seconds during interactive coding sessions. Background compaction keeps the agent responsive.

## Appendix C. Risks

1. **SQLite concurrency lock contention.** High-frequency multi-subagent writes could encounter database lock timeouts. Mitigated by using WAL mode, short atomic transactions, and an in-process asyncio write queue. Lands in PR-01 and monitored across all lanes.
2. **Non-deterministic tool side effects.** Bash commands modifying external services or filesystem files may leave partial artifacts before a crash. Mitigated by defaulting bash tools to unsafe replay and creating isolated Git worktrees. Lands in PR-03 and PR-05.
3. **Model context compaction loss.** Summarization may drop subtle technical constraints needed in later turns. Mitigated by keeping complete uncompressed history searchable via the `search_history` tool. Lands in PR-06.

## Appendix D. Links and reading list

- Pi Durable specification and reference guide. Reference at `@earendil-works/pi-durable`.
- PydanticAI documentation on `agent.iter` and graph nodes. Reference at `https://pydantic.dev/docs/ai/`.
- Autopilot stack execution guide. Reference at `pstack/skills/poteto-mode/playbooks/autopilot-stack.md`.
- Swarm verification playbook. Reference at `pstack/skills/swarm/SKILL.md`.
