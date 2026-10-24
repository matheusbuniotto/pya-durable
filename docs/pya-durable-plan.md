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

- [ ] Create `src/pya_durable/replay.py`.
- [ ] Create `src/pya_durable/recovery.py`.
- [ ] Create `tests/test_recovery.py`.

**Build.**

- [ ] Implement `@durable_tool` decorator with replay policy parameter in `src/pya_durable/replay.py`.
- [ ] Implement `Harness.resume()` scanning running tasks and reconciling interrupted tool invocations in `src/pya_durable/recovery.py`.

**You see.**

- [ ] Crash simulation resumes and re-executes safe tools while sealing unsafe tools.
- [ ] Log outputs `Reconciliation finished, resumed task id from phase`.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Unit tests for safe versus unsafe replay policies and resumption logic in `tests/test_recovery.py`. Run `uv run pytest tests/test_recovery.py`.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.7-xhigh-fast` at the PR head, per the boot recipe.

- [ ] Lane 1. Regression lane against trunk. Simulate crash during tool execution at trunk and head. If trunk lacks the feature, record that and gate crash recovery state reconciliation. Save `lane-1-crash-recovery.png`. Pass when task resumes and completes successfully.
- [ ] Lane 2. Safe read tool replay after process termination. Save `lane-2-safe-tool.png`. Pass when tool reruns and stores result.
- [ ] Lane 3. Unsafe mutating tool interrupted state recording. Save `lane-3-unsafe-tool.png`. Pass when tool is not re-executed and marked interrupted.
- [ ] Lane 4. Request ID deduplication on retry submission. Save `lane-4-idempotent-submit.png`. Pass when second submission returns original task result.
- [ ] Lane 5. Multiple in-flight tasks recovered in parallel during boot. Save `lane-5-parallel-recovery.png`. Pass when all running tasks transition to valid state.
- [ ] Lane 6. Partial tool output preserved across crash. Save `lane-6-partial-output.png`. Pass when streamed partial text is retrievable.
- [ ] Lane 7. Tool timeout during replay handled gracefully. Save `lane-7-tool-timeout.png`. Pass when timeout marks tool invocation failed.
- [ ] Lane 8. Process kill during model request generation. Save `lane-8-kill-model-req.png`. Pass when model request regenerates cleanly on restart.
- [ ] Lane 9. Recovery with clean database containing zero incomplete tasks. Save `lane-9-noop-recovery.png`. Pass when resume executes as a clean no-op.
- [ ] Lane 10. Replay policy validation rejecting unknown policy string. Save `lane-10-invalid-policy.png`. Pass when invalid policy raises configuration error.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Time required to scan and reconcile one hundred interrupted tool calls at boot.
- [ ] Probe. Measure execution duration of `Harness.resume()` over database preloaded with one hundred started tool rows.
- [ ] Baseline. Measure empty database boot time of two milliseconds.
- [ ] Rule. Reconciling one hundred interrupted invocations must complete within fifty milliseconds.

**Review gate.** None. PR-03 is not review-gated.

**Merge.**

- [ ] Root clean verdict at exact head SHA.
- [ ] Bugbot triage done.
- [ ] Rebased onto parent branch after verdict, patch-id unchanged.
- [ ] Owner pushes branch and reports STACK-READY to root.

## Implement task hierarchy and subagent delegation (PR-04)

**Depends on.** PR-03.

**Files.**

- [ ] Create `src/pya_durable/task_tree.py`.
- [ ] Create `src/pya_durable/subagents.py`.
- [ ] Create `tests/test_tasks.py`.

**Build.**

- [ ] Implement `define_task` state machine runner with phase transitions in `src/pya_durable/task_tree.py`.
- [ ] Implement subagent conversation creation with task ownership in `src/pya_durable/subagents.py`.
- [ ] Implement recursive bottom-up task cancellation.

**You see.**

- [ ] Parent task creates child task and awaits result.
- [ ] Canceling parent task cascades abort to all child tasks and subagents.
- [ ] Log outputs `Aborted child task id due to parent cancellation`.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Unit tests for hierarchical ownership, phase checkpoints, and abort cascades in `tests/test_tasks.py`. Run `uv run pytest tests/test_tasks.py`.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.7-xhigh-fast` at the PR head, per the boot recipe.

- [ ] Lane 1. Regression lane against trunk. Run hierarchical task with subagent on trunk and head. If trunk lacks the feature, record that and gate subagent lifecycle and result return. Save `lane-1-subagent-lifecycle.png`. Pass when subagent conversation completes and yields result to parent.
- [ ] Lane 2. Multi-tier task ownership tree with depth of three. Save `lane-2-depth-three.png`. Pass when root task coordinates leaf children.
- [ ] Lane 3. Bottom-up abort cascade on parent cancel. Save `lane-3-abort-cascade.png`. Pass when all descendants transition to aborted status.
- [ ] Lane 4. Task failure with fail-fast policy stopping sibling tasks. Save `lane-4-fail-fast.png`. Pass when sibling tasks abort immediately.
- [ ] Lane 5. Subagent crash recovery resuming child conversation. Save `lane-5-subagent-recovery.png`. Pass when child agent restarts from latest turn.
- [ ] Lane 6. Subagent context isolation preventing tool access leak. Save `lane-6-context-isolation.png`. Pass when child agent cannot see parent tool credentials.
- [ ] Lane 7. Custom task phase transition checkpoint persistence. Save `lane-7-phase-checkpoint.png`. Pass when phase state restores accurately.
- [ ] Lane 8. Task compensation logic executing during abort. Save `lane-8-compensation.png`. Pass when compensation cleanup deletes temporary files.
- [ ] Lane 9. Background task running detached from parent turn. Save `lane-9-detached-task.png`. Pass when parent completes while background task continues.
- [ ] Lane 10. Cycle detection in task parent ownership pointers. Save `lane-10-cycle-detect.png`. Pass when recursive parent cycle raises validation error.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Cancellation cascade latency across a tree of fifty subtasks.
- [ ] Probe. Spawn tree of fifty tasks and invoke abort on root node.
- [ ] Baseline. Measure single task abort taking one millisecond.
- [ ] Rule. Aborting fifty descendant tasks must complete in under twenty-five milliseconds.

**Review gate.** None. PR-04 is not review-gated.

**Merge.**

- [ ] Root clean verdict at exact head SHA.
- [ ] Bugbot triage done.
- [ ] Rebased onto parent branch after verdict, patch-id unchanged.
- [ ] Owner pushes branch and reports STACK-READY to root.

## Support transcript branching and live steering (PR-05)

**Depends on.** PR-04.

**Files.**

- [ ] Create `src/pya_durable/branching.py`.
- [ ] Create `src/pya_durable/steering.py`.
- [ ] Create `src/pya_durable/worktrees.py`.
- [ ] Create `tests/test_branching.py`.

**Build.**

- [ ] Implement recursive CTE query for copy-on-write transcript history in `src/pya_durable/branching.py`.
- [ ] Implement `conversation.fork()` creating linked child conversation without copying rows.
- [ ] Implement `when_busy="steer"` injection queue between tool steps in `src/pya_durable/steering.py`.
- [ ] Implement Git worktree isolation per active conversation in `src/pya_durable/worktrees.py`.

**You see.**

- [ ] Forked conversation loads full history through parent entry pointers.
- [ ] Steering message steers agent after tool step without resetting turn.
- [ ] Log outputs `Injected steer message into active turn for conversation id`.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Unit tests for tree traversal, fork creation, and steer injection in `tests/test_branching.py`. Run `uv run pytest tests/test_branching.py`.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.7-xhigh-fast` at the PR head, per the boot recipe.

- [ ] Lane 1. Regression lane against trunk. Fork conversation at past entry and submit diverging prompt at trunk and head. If trunk lacks the feature, record that and gate fork creation and independent turn execution. Save `lane-1-fork-branch.png`. Pass when both branches advance without conflict.
- [ ] Lane 2. Forking from deep transcript history with twenty turns. Save `lane-2-deep-fork.png`. Pass when recursive CTE resolves correct chronological order.
- [ ] Lane 3. Mid-run steer injection during multi-step tool execution. Save `lane-3-steer-inject.png`. Pass when model adjusts behavior on next turn.
- [ ] Lane 4. Concurrent execution on two forks of same parent. Save `lane-4-concurrent-forks.png`. Pass when both forks commit distinct transcript entries.
- [ ] Lane 5. Git worktree creation for isolated branch exploration. Save `lane-5-worktree-create.png`. Pass when worktree directory checkouts target branch.
- [ ] Lane 6. Worktree cleanup after completed exploration. Save `lane-6-worktree-clean.png`. Pass when temporary worktree and branch prune cleanly.
- [ ] Lane 7. Fork pointing to nonexistent entry ID rejection. Save `lane-7-invalid-fork-entry.png`. Pass when invalid entry ID raises validation error.
- [ ] Lane 8. Steer queue buffer handling multiple sequential steers. Save `lane-8-multiple-steers.png`. Pass when steers process in FIFO order.
- [ ] Lane 9. Forking an already-forked conversation producing tree branch. Save `lane-9-nested-branch.png`. Pass when history traverses through both parent forks.
- [ ] Lane 10. Steer ignored cleanly if turn settles before steer arrives. Save `lane-10-late-steer.png`. Pass when late steer handles as next user input.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Query latency to reconstruct five hundred turns via recursive CTE.
- [ ] Probe. Measure execution duration of recursive parent-pointer query across five hundred linked transcript rows.
- [ ] Baseline. Measure flat table select of five hundred rows taking two milliseconds.
- [ ] Rule. Recursive tree traversal for five hundred entries must complete within ten milliseconds.

**Review gate.** The operator reviews before merge.

- [ ] Copy lane 1 screenshots into `docs/media/pr-05-review-fork-branch.png`.
- [ ] Record a 30 to 60 second video of the change on a lane VM. Save it as `docs/media/pr-05-review.mp4`.
- [ ] Post the screenshots and the video in chat. Stop at merge-ready. Wait for the operator's click.

**Merge.**

- [ ] Root clean verdict at exact head SHA.
- [ ] Bugbot triage done.
- [ ] Rebased onto parent branch after verdict, patch-id unchanged.
- [ ] Owner pushes branch and reports STACK-READY to root.

## Add durable documents and background compaction (PR-06)

**Depends on.** PR-05.

**Files.**

- [ ] Create `src/pya_durable/docs.py`.
- [ ] Create `src/pya_durable/compaction.py`.
- [ ] Create `src/pya_durable/streaming.py`.
- [ ] Create `tests/test_docs_and_compaction.py`.

**Build.**

- [ ] Implement `define_doc` decorator and atomic SQLite document storage in `src/pya_durable/docs.py`.
- [ ] Implement background compaction worker and context handoff in `src/pya_durable/compaction.py`.
- [ ] Implement `conversation.watch()` real-time state subscription in `src/pya_durable/streaming.py`.

**You see.**

- [ ] Document changes commit atomically with agent tool steps.
- [ ] Background compaction generates structured summary without pausing agent.
- [ ] Observer streams receive live state deltas.
- [ ] Log outputs `Compacted turns up to entry id into background checkpoint`.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Unit tests for document versioning, compaction thresholds, and event streaming in `tests/test_docs_and_compaction.py`. Run `uv run pytest tests/test_docs_and_compaction.py`.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.7-xhigh-fast` at the PR head, per the boot recipe.

- [ ] Lane 1. Regression lane against trunk. Run document mutation and background compaction on trunk and head. If trunk lacks the feature, record that and gate document persistence and context summary creation. Save `lane-1-docs-compaction.png`. Pass when document persists and summary checkpoint is generated.
- [ ] Lane 2. Document update rolled back atomically when tool call fails. Save `lane-2-doc-rollback.png`. Pass when document retains prior version.
- [ ] Lane 3. Background compaction triggering on high token count. Save `lane-3-auto-compaction.png`. Pass when background task spawns above threshold.
- [ ] Lane 4. Context handoff tool clearing active messages while saving note. Save `lane-4-handoff-tool.png`. Pass when active message buffer resets.
- [ ] Lane 5. Full-text search over compacted historical turns via tool. Save `lane-5-search-history.png`. Pass when query matches text in archived entry.
- [ ] Lane 6. Real-time state subscription streaming node deltas to client. Save `lane-6-state-stream.png`. Pass when client receives all intermediate events.
- [ ] Lane 7. Multiple simultaneous subscribers reading event stream. Save `lane-7-multi-subscriber.png`. Pass when all clients receive identical deltas.
- [ ] Lane 8. Document schema migration on version bump. Save `lane-8-schema-migration.png`. Pass when updated schema loads existing rows cleanly.
- [ ] Lane 9. Compaction boundary safety preventing open tool truncation. Save `lane-9-safe-compaction.png`. Pass when open tool call is not truncated.
- [ ] Lane 10. Streaming client disconnect handled without worker crash. Save `lane-10-client-disconnect.png`. Pass when agent continues running after client drops.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Background compaction runtime on fifty conversation turns.
- [ ] Probe. Measure elapsed execution time of background compaction task with mock summarizer.
- [ ] Baseline. Measure direct prompt summary taking ten milliseconds.
- [ ] Rule. Background compaction must complete in under thirty milliseconds without delaying foreground event loop.

**Review gate.** The operator reviews before merge.

- [ ] Copy lane 1 screenshots into `docs/media/pr-06-review-docs-compaction.png`.
- [ ] Record a 30 to 60 second video of the change on a lane VM. Save it as `docs/media/pr-06-review.mp4`.
- [ ] Post the screenshots and the video in chat. Stop at merge-ready. Wait for the operator's click.

**Merge.**

- [ ] Root clean verdict at exact head SHA.
- [ ] Bugbot triage done.
- [ ] Rebased onto parent branch after verdict, patch-id unchanged.
- [ ] Owner pushes branch and reports STACK-READY to root.

## Close the program

- [ ] Every box above is checked with its evidence.
- [ ] Reply to the operator with the report the execution playbook names.

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
