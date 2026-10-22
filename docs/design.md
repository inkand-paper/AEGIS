# AEGIS — Engineering Design Document

**Version:** 0.1 (pre-implementation)
**Status:** Draft — supersedes the original planning note
**Audience:** Solo developer (you), plus any future contributor
**Scope:** Architecture, contracts, safety model, phase plan, and measurable exit criteria for the first six months of AEGIS

---

## 0. Reading guide

This document is the single source of truth for how AEGIS is built. It is organized so that you can read it top-to-bottom once, then use §12 as the working checklist. Every claim about "we will" is paired with an exit criterion that can be checked by running a command or reading a test name. If a section has no verifiable exit criterion, it is design intent, not a commitment, and is marked **[intent]**.

The original planning note is retained in `docs/original-plan.md` for provenance. Where this document contradicts it, this document wins.

---

## 1. What AEGIS is

AEGIS is a long-running AI agent that:

1. Accepts a natural-language **objective**.
2. **Plans** a multi-step task with explicit dependencies.
3. **Executes** the plan by proposing structured tool calls to a validated dispatcher.
4. **Observes** the results of each call.
5. **Recovers** from failures by replanning, not by retrying blindly.
6. **Verifies** completion against task-specific evidence before declaring success.
7. **Reports** a finished artifact plus an audit trail of what actually happened.

The product sentence is: *"You describe the outcome. The agent does the work. The agent checks the work. You receive the finished result."*

**Non-goals for v1** (explicitly out of scope until Phase 9 is complete):
- Multi-agent orchestration.
- A polished desktop UI.
- Local model inference.
- Cloud deployment / multi-tenant operation.
- General-purpose autonomy outside a per-task workspace.

---

## 2. Design principles (non-negotiable)

These are enforced in code review of your own commits, not just aspirational.

| # | Principle | Consequence if violated |
|---|-----------|--------------------------|
| P1 | The model never executes actions directly. It proposes; the dispatcher validates and executes. | Untestable, unauditable, unsafe. |
| P2 | Every task has finite budgets (tokens, wall-clock, tool calls, dollars). | Runaway loops, unpredictable cost. |
| P3 | No success claim without evidence checked by the verifier. | Agent lies about completion. |
| P4 | State is persisted before side effects, not after. | Duplicate side effects on crash. |
| P5 | Every tool call is recorded with inputs, outputs, duration, and outcome. | Cannot debug or benchmark. |
| P6 | Filesystem access is restricted to a per-task workspace with an explicit allowlist. | Local data loss. |
| P7 | Shell and network access are gated by policy, not by prompt instructions. | Prompt injection becomes RCE. |
| P8 | One agent, one loop, bounded replanning. No "Planner Agent" / "Critic Agent" for v1. | Complexity without measured benefit. |
| P9 | No architectural change without a benchmark result motivating it. | Endless rewrites. |
| P10 | Secrets live in `.env`, never in source, never in logs. | Leakage. |

---

## 3. Trust and threat model

This section exists because the original note under-specified it.

### 3.1 Adversaries considered

1. **Prompt injection** from untrusted content the agent reads (web pages, files, tool outputs). The agent may be tricked into proposing a harmful tool call.
2. **Model error** — the model proposes a malformed, out-of-scope, or destructive call by accident.
3. **The user's own mistakes** — a task objective that, if taken literally, would delete data or leak secrets.
4. **Supply-chain** — a dependency or tool that does something unexpected.

### 3.2 Controls (ordered by importance)

1. **Dispatcher-mediated execution.** The model's output is *data*, parsed into a typed `ToolCall`, validated against the tool's Pydantic argument model, checked against policy, and only then executed. P1 is the primary control; everything else is defense-in-depth.
2. **Per-task workspace.** Each task gets `workspaces/<task_id>/`. Filesystem tools resolve paths against this root and reject anything outside it (including via `..`, symlinks, or absolute paths). This is the primary containment boundary for v1.
3. **Command allowlist.** The shell tool does not accept arbitrary commands. It accepts a small set of named operations (`run_tests`, `run_linter`, `run_formatter`, `run_python_script`) whose argument shapes are validated. Raw `subprocess(shell=True)` is never exposed to the model.
4. **Network policy at the tool level.** Tools that touch the network declare it. In Phase 1–4, the only networked tool is the model client itself. Browser (Phase 5) and image generation (Phase 8) add explicit, auditable network tools.
5. **Approval gates.** Actions outside the workspace, or with side effects beyond the task's declared scope, move the task to `WAITING_FOR_APPROVAL` and require a human `approve` command.
6. **Docker isolation (Phase 6, defense-in-depth).** Code execution runs inside a container with CPU/memory/time limits and no network unless explicitly granted. Docker is *not* the primary control; the dispatcher and workspace are. The doc says this because "a container alone is not a complete security guarantee" is true and must not be forgotten.
7. **Secret hygiene.** `.env` is gitignored and excluded from logs by a redaction filter in the logger. `git check-ignore -v .env` is part of the Phase 0 exit criterion.

### 3.3 What is explicitly *not* defended against in v1

- A malicious local user with shell access on the same machine.
- Kernel-level exploits or container escapes.
- Model providers logging your prompts (assumed; documented in README).
- Social engineering of the human approver.

These are documented so nobody assumes otherwise.

---

## 4. Architecture

### 4.1 Component map

```
┌─────────────────────────────────────────────────────────────────┐
│  CLI (src/aegis/main.py)                                        │
│  submit · status · approve · cancel · resume · report           │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  Task Controller (src/aegis/controller.py)                      │
│  task lifecycle · budgets · cancellation · approvals            │
└───┬───────────┬───────────┬───────────┬───────────┬─────────────┘
    │           │           │           │           │
┌───▼───┐  ┌────▼────┐  ┌───▼────┐  ┌───▼────┐  ┌───▼──────────┐
│Planner│  │Executor │  │Verifier│  │ Memory │  │ Event Log    │
│       │  │(dispatch│  │        │  │ (state)│  │ (append-only)│
│       │  │ + tools)│  │        │  │        │  │              │
└───┬───┘  └────┬────┘  └───┬────┘  └───┬────┘  └───┬──────────┘
    │           │           │           │           │
    │      ┌────▼────┐      │      ┌────▼────┐      │
    │      │  Tools  │      │      │ SQLite  │      │
    │      │ registry│      │      │ (SQLAlch│      │
    │      └────┬────┘      │      │  +Alemb)│      │
    │           │           │      └─────────┘      │
    │           │           │                       │
    │      ┌────▼─────────────────────────────┐     │
    │      │ Tool implementations             │     │
    │      │  filesystem · shell · browser ·  │     │
    │      │  image · model                   │     │
    │      └──────────────────────────────────┘     │
    │                                               │
    └───────────────────────────────────────────────┘
```

### 4.2 Data flow for one step

```
Controller.step(task)
  → Planner.current_step(task)                  # what to do next
  → ModelClient.generate(context, tools)        # model proposes
  → parse into ToolCall                         # P1
  → validate against tool.arg_model             # P1
  → policy check (workspace, allowlist, budget) # P2, P6, P7
  → if risky: task → WAITING_FOR_APPROVAL       # approval gate
  → Executor.execute(ToolCall)                  # side effect
  → append ToolCallRecord + Observation to log  # P5
  → Verifier.update(task, observation)          # evidence tracking
  → Planner.observe(task, observation)          # replan if needed
  → loop or terminate
```

### 4.3 Why this shape

- **Controller above Planner/Executor/Verifier** means budgets and cancellation are enforced in one place.
- **Verifier is a peer of Executor, not a sub-step of Planner,** because completion is a separate decision from action selection. This is the architectural expression of "evidence-based completion."
- **Event log is append-only and separate from the current-state tables** so that crash recovery can reconstruct *what happened* (log) versus *what we thought was happening* (state). This distinction is what makes P4 work.

---

## 5. Core data contracts

All models are Pydantic v2 `BaseModel` with `model_config = ConfigDict(extra="forbid")`. Field names are final; types may widen.

### 5.1 Task

```python
class TaskState(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Budget(BaseModel):
    max_wall_clock_s: int = 3600
    max_model_calls: int = 200
    max_tool_calls: int = 500
    max_tokens_total: int = 2_000_000
    max_dollars: Decimal = Decimal("5.00")

class BudgetUsage(BaseModel):
    wall_clock_s: float = 0.0
    model_calls: int = 0
    tool_calls: int = 0
    tokens_total: int = 0
    dollars: Decimal = Decimal("0")

class Task(BaseModel):
    id: str                      # uuid4 hex
    objective: str
    state: TaskState
    budget: Budget
    usage: BudgetUsage
    workspace_path: Path          # resolved, inside workspaces/
    created_at: datetime
    updated_at: datetime
    plan_id: str | None = None
    last_error: str | None = None
```

**Invariants enforced by the controller:**
- `workspace_path` is always a child of the configured `workspaces/` root (checked with `Path.resolve().is_relative_to()`).
- A task in `COMPLETED` has a non-null `verification_result_id`.
- A task in `FAILED` has a non-null `last_error`.
- `usage` never exceeds `budget` without the task transitioning to `FAILED` with a `BudgetExceeded` reason.

### 5.2 Plan and Step

```python
class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"

class Step(BaseModel):
    id: str
    plan_id: str
    index: int                    # ordering within plan
    description: str              # human-readable
    depends_on: list[str] = []    # step ids
    status: StepStatus = StepStatus.PENDING
    evidence_ids: list[str] = []  # verifier evidence records
    attempts: int = 0

class Plan(BaseModel):
    id: str
    task_id: str
    steps: list[Step]
    current_step_id: str | None
    revision: int = 1              # bumped on replan
    superseded_by: str | None = None
```

**Replanning rule (P8):** a replan creates a *new* `Plan` with `revision+1` and sets `superseded_by` on the old one. It does **not** create a new agent, and it does **not** renumber steps. The verifier's evidence survives replanning; only step status resets for steps that were invalidated.

### 5.3 ToolCall and Observation

```python
class ToolCall(BaseModel):
    id: str
    task_id: str
    step_id: str | None
    tool_name: str
    arguments: dict[str, Any]       # validated against tool.arg_model
    requested_at: datetime
    idempotency_key: str | None     # set by tools that mutate state

class ToolCallRecord(BaseModel):
    call: ToolCall
    started_at: datetime
    finished_at: datetime
    success: bool
    result_model: str | None       # name of the result Pydantic model
    result: dict[str, Any] | None
    error: str | None
    truncated: bool = False

class Observation(BaseModel):
    tool_call_id: str
    summary: str                    # short, for model context
    detail_ref: str | None          # pointer to full record in DB
    artifact_ids: list[str] = []
```

**Result models (not `Any`).** Each tool declares `arg_model: type[BaseModel]` and `result_model: type[BaseModel]`. The dispatcher validates both. This fixes the flaw in the original `Tool` Protocol sketch: `output: Any` defeats the "validated structured responses" goal, so the real contract is:

```python
class Tool(Protocol):
    name: str
    description: str
    arg_model: type[BaseModel]
    result_model: type[BaseModel]
    side_effects: Literal["none", "workspace", "external"]
    requires_network: bool
    idempotent: bool

    async def execute(self, args: BaseModel, ctx: ToolContext) -> BaseModel: ...
```

### 5.4 Evidence and VerificationResult

```python
class EvidenceKind(str, Enum):
    TEST_PASS = "test_pass"
    FILE_EXISTS = "file_exists"
    FILE_DIFF = "file_diff"
    ARTIFACT_HASH = "artifact_hash"
    BROWSER_STATE = "browser_state"
    HUMAN_APPROVAL = "human_approval"
    COMMAND_EXIT_ZERO = "command_exit_zero"

class Evidence(BaseModel):
    id: str
    task_id: str
    step_id: str | None
    kind: EvidenceKind
    payload: dict[str, Any]        # kind-specific, validated
    created_at: datetime
    source_tool_call_id: str

class VerificationResult(BaseModel):
    id: str
    task_id: str
    passed: bool
    checks: list[dict[str, Any]]   # {name, passed, evidence_ids, message}
    created_at: datetime
    verifier_version: str
```

**Rule (P3):** the verifier is the *only* component that can transition a task to `COMPLETED`. The model's claim of completion is a proposal, recorded as evidence of kind `HUMAN_APPROVAL` only if a human actually approved it.

### 5.5 Event log

```python
class EventType(str, Enum):
    TASK_CREATED = "task_created"
    STATE_CHANGED = "state_changed"
    PLAN_CREATED = "plan_created"
    PLAN_REVISED = "plan_revised"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    EVIDENCE_ADDED = "evidence_added"
    VERIFICATION = "verification"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    BUDGET_EXCEEDED = "budget_exceeded"
    ERROR = "error"

class Event(BaseModel):
    id: int                        # autoincrement
    task_id: str
    type: EventType
    payload: dict[str, Any]
    created_at: datetime
    correlation_id: str | None     # ties related events (e.g. call → result)
```

The event log is **append-only**. It is the source of truth for recovery and for the benchmark harness.

---

## 6. Tool contract

### 6.1 Registry

```python
class ToolRegistry:
    def register(self, tool: Tool) -> None: ...
    def get(self, name: str) -> Tool: ...
    def list_for_model(self) -> list[dict]:  # JSON schemas of arg_model
        ...
```

Registration is explicit in `src/aegis/tools/__init__.py`. There is no auto-discovery (it hides the attack surface).

### 6.2 Dispatch pipeline

`Executor.execute(call: ToolCall)` performs, in order:

1. **Lookup.** Tool must be registered. Unknown name → `ToolNotFound` error record, no side effect.
2. **Argument validation.** Parse `arguments` into `tool.arg_model`. Validation error → error record with the validation message so the model can correct.
3. **Policy check.**
   - Filesystem tools: resolve path, assert `is_relative_to(workspace)`.
   - Shell tools: assert operation is in allowlist and args match its schema.
   - Network tools: assert `requires_network` is true and the task has network budget.
   - Any tool: assert `usage.tool_calls < budget.max_tool_calls`.
4. **Approval gate.** If `side_effects != "none"` and the call is outside the task's declared scope → transition to `WAITING_FOR_APPROVAL`, emit `APPROVAL_REQUESTED`, return without executing.
5. **Pre-record.** Append `TOOL_CALL` event with the validated call (this is P4: record before side effect).
6. **Execute** with timeout and cancellation support.
7. **Result validation.** Parse raw output into `tool.result_model`. If it fails, that's a tool bug, not a model bug — surface it as an internal error and fail the step (do not let the model paper over it).
8. **Post-record.** Append `TOOL_RESULT` event, write `ToolCallRecord`, return `Observation`.

### 6.3 Initial tool set by phase

| Tool | Phase | Side effects | Network | Idempotent |
|------|-------|--------------|---------|------------|
| `calculator` | 2 | none | no | yes |
| `fs_list` | 2 | none | no | yes |
| `fs_read` | 2 | none | no | yes |
| `fs_write` | 3 | workspace | no | no |
| `fs_edit` (bounded diff) | 3 | workspace | no | no |
| `shell_run_tests` | 3 | workspace | no | yes |
| `shell_run_formatter` | 3 | workspace | no | yes |
| `browser_navigate` | 5 | external | yes | no |
| `browser_click` | 5 | external | yes | no |
| `browser_screenshot` | 5 | workspace | yes | yes |
| `code_exec` (Docker) | 6 | workspace | opt-in | no |
| `desktop_*` | 7 | external | no | varies |
| `image_generate` | 8 | workspace | yes | no |

Tools are added per phase, never "just in case."

---

## 7. Model client contract

```python
class ModelResponse(BaseModel):
    text: str | None
    tool_calls: list[RawToolCall]   # unvalidated
    usage: ModelUsage               # tokens, cost
    finish_reason: str

class ModelClient(Protocol):
    async def generate(
        self,
        messages: list[Message],
        tools: list[dict],          # JSON schemas
        *,
        max_tokens: int,
        timeout_s: float,
    ) -> ModelResponse: ...
```

**Rules:**
- `ModelClient` output is *untrusted input*. `RawToolCall` is parsed into `ToolCall` and validated by the dispatcher. There is no code path from `ModelResponse` to a side effect that bypasses the dispatcher.
- Retries are limited to transient errors (timeout, 429, 5xx) with exponential backoff, max 3 attempts, counted against `budget.model_calls`.
- One provider for Phase 1. The abstraction exists so a second provider is a *config change*, not a refactor. LiteLLM is a candidate, but a direct SDK is preferred until a second provider is genuinely needed (P9).
- All requests and responses are logged with a redaction filter for `Authorization`, `api_key`, and anything matching `sk-[A-Za-z0-9]+`.

**Open decision (blocks Phase 1):** which provider for the first `ModelClient`? This affects structured-output strategy (native JSON schema vs. validation shim) and tool-call formatting. Decide before writing `model_client.py`. The `Task`/`Budget`/`ToolCall` contracts in §5 are provider-independent and do not need to wait.

---

## 8. Persistence and recovery

### 8.1 Schema (SQLite via SQLAlchemy, Alembic migrations)

| Table | Purpose | Mutability |
|-------|---------|------------|
| `tasks` | current task state | mutable |
| `plans` | plan revisions | insert-only |
| `steps` | steps per plan | mutable status |
| `tool_calls` | validated calls | insert-only |
| `tool_call_records` | outcomes | insert-only |
| `evidence` | verifier evidence | insert-only |
| `verification_results` | final checks | insert-only |
| `events` | append-only log | insert-only |
| `artifacts` | files produced, with sha256 | insert-only |
| `approvals` | pending/granted approvals | mutable |

DB lives at `data/aegis.db`. `data/` is gitignored.

### 8.2 Recovery protocol (Phase 4 exit criterion)

On `aegis resume <task_id>`:

1. Load `Task`. If state is `COMPLETED`/`FAILED`/`CANCELLED`, refuse and print the final report.
2. Reconstruct from `events`: for every `TOOL_CALL` with no matching `TOOL_RESULT`, the call may have partially executed. Mark it **indeterminate**.
3. For indeterminate calls:
   - If `tool.idempotent`, re-execute and record.
   - If not idempotent, transition to `WAITING_FOR_APPROVAL` with a message describing the indeterminate call. **Never blindly repeat.**
4. Recompute `BudgetUsage` from events (not from the `usage` column) to detect drift.
5. Resume the loop from the plan's `current_step_id`.

### 8.3 Why event log + state tables (not just one)

- State tables answer "where are we?" cheaply.
- Event log answers "how did we get here, and is the state consistent?" — required for P4 and for the benchmark harness.
- Recomputing `usage` from events at resume time catches bugs where the state column and reality diverged.

---

## 9. Verifier

The verifier is a small, deterministic rule engine plus a checklist. It is **not** an LLM in v1. This is deliberate: an LLM verifier is a second agent in disguise (violates P8) and is unmeasurable.

### 9.1 How completion is decided

A task declares (in its objective, or via a `--accept` flag on `aegis submit`) a set of acceptance criteria. Each criterion maps to one or more `EvidenceKind`s. The verifier passes the task only if **every** criterion has matching, non-stale evidence.

Example criteria and their evidence:

| Criterion | Evidence |
|-----------|----------|
| "tests pass" | `TEST_PASS` with exit code 0 and a test count > 0, from a `shell_run_tests` call in the final plan revision |
| "file X exists with content matching Y" | `FILE_EXISTS` + `ARTIFACT_HASH` |
| "browser shows text Z" | `BROWSER_STATE` from a `browser_screenshot` + DOM assertion |
| "diff touches only files under `src/`" | `FILE_DIFF` parsed and checked |

**Staleness rule:** evidence created before the most recent mutation of a relevant artifact is stale. The verifier re-runs affected checks before passing. This is what prevents "I ran tests, then edited a file, then claimed success."

### 9.2 What the verifier does *not* do

- It does not decide *what* to do next. That is the planner.
- It does not use an LLM. If a criterion is genuinely fuzzy ("UI looks good"), it must be decomposed into measurable checks or routed to `HUMAN_APPROVAL`.

---

## 10. Planning and replanning

### 10.1 Initial plan

The planner asks the model for a plan as a structured `Plan` object (steps, descriptions, dependencies). The plan is validated (no cycles, no unknown step ids in `depends_on`, at least one step) before being persisted as `revision=1`.

### 10.2 Replanning triggers (P8 — bounded)

A replan is allowed only when one of these is true:
1. A step failed and its `attempts >= 2`.
2. The verifier rejected completion.
3. The current step's tool is unavailable (e.g., network down, browser crashed).
4. The human approved a replan request.

A replan is **not** allowed for: "the model feels stuck," "the model wants to try a different approach," or any trigger initiated by the model alone. The model can *suggest* a replan, but the controller decides based on the triggers above.

**Replan budget:** at most `max_replans` (default 3) per task. Exceeding it → `FAILED` with `ReplanBudgetExceeded`.

---

## 11. CLI

```
aegis submit "<objective>" [--accept "<criterion>" ...] [--budget-dollars N]
aegis status <task_id>            # human-readable + --json
aegis watch <task_id>             # tail events
aegis approve <task_id>           # grant pending approval
aegis cancel <task_id>
aegis resume <task_id>
aegis report <task_id>            # final report + artifact list
aegis bench run <suite>           # Phase 9
```

Exit codes: `0` completed, `1` failed, `2` waiting for approval, `3` cancelled, `4` usage error. Scriptable, no interactive prompts except `approve`.

---

## 12. Phase plan with exit criteria

Durations are targets, not promises. A phase is complete only when its exit criterion **passes as an automated test** (or a documented manual check where automation is impractical, e.g., desktop UI).

### Phase 0 — Engineering foundation (week 1)
**Build:** repo, `uv` project, `main.py` printing the initialized message, `config.py` with Pydantic settings, `models.py` with `Task`/`Budget`/`ToolCall` skeletons, `.gitignore`, `.env.example`, pytest wired, ruff + mypy configured, first commit.
**Exit criteria:**
- `uv run python src/aegis/main.py` prints `AEGIS initialized.`
- `uv run pytest` passes (at least one test: config validation rejects a bad budget).
- `uv run ruff check .` and `uv run mypy src` pass.
- `git check-ignore -v .env data/aegis.db workspaces/` returns a match for each.
- Fresh-clone setup documented in README and verified by you in a temp directory.

### Phase 1 — Model client (weeks 2–3)
**Build:** `ModelClient` for one provider, `Message`/`ModelResponse` models, timeout, bounded retry, redaction filter, cost tracking into `BudgetUsage`.
**Exit criteria:**
- A test calls the provider (or a recorded fixture) and asserts a validated `ModelResponse`.
- A test feeds a malformed response fixture and asserts a `ModelResponseError`, not a crash.
- A test asserts a 429 fixture triggers ≤3 retries and then raises.
- A test asserts the logger redacts `sk-...`.

### Phase 2 — Agent loop, minimal persistence (weeks 4–5)
**Build:** controller, planner (initial plan only), executor with dispatcher, `calculator` + `fs_list` + `fs_read` tools, `TaskState` transitions, `Budget` enforcement, **minimal event log + checkpoint table (moved from Phase 4 — see §14)**.
**Exit criteria:**
- A test runs a 3-step task using `calculator` and `fs_read` and reaches `COMPLETED` with a verification result.
- A test injects a transient tool failure and asserts recovery.
- A test sets `max_tool_calls=2` and asserts `FAILED` with `BudgetExceeded`.
- A test calls `cancel` mid-loop and asserts `CANCELLED` with no further events.
- Every tool call in the test DB has a matching `TOOL_RESULT` event (P5).

### Phase 3 — Filesystem and coding tools (weeks 6–7)
**Build:** `fs_write`, `fs_edit`, `shell_run_tests`, `shell_run_formatter`, diff generation, workspace path enforcement, dry-run mode.
**Exit criteria:**
- A test creates a small Python project in the workspace, runs its tests, and gets a non-empty diff.
- A test attempts `fs_read` on `../../etc/passwd` (and a symlink escape) and asserts rejection with a policy error recorded in the event log.
- A test asserts `shell_run_tests` rejects an argument shape not in the allowlist.

### Phase 4 — Durable state and recovery (weeks 8–9)
**Build:** full SQLAlchemy schema, Alembic migrations, event-log replay, `resume`, idempotency keys for non-idempotent tools.
**Exit criteria:**
- A test kills the process mid-task (subprocess + SIGKILL), restarts, and `resume` reaches `COMPLETED` or `WAITING_FOR_APPROVAL` (never blindly re-executes a non-idempotent call).
- A test corrupts `usage` in the DB and asserts resume recomputes it from events and detects drift.
- A migration test upgrades from the previous schema version.

### Phase 5 — Browser (weeks 10–12)
**Build:** Playwright adapter, `browser_navigate`, `browser_click`, `browser_screenshot`, DOM inspection, page-state evidence.
**Exit criteria:**
- A controlled test suite (local static pages) completes 5 scripted workflows.
- A test simulates a changed layout and asserts the agent observes the failure (screenshot/DOM) and replans rather than retrying the same click.
- A test asserts `browser_click` does not report success solely because the click function returned — success requires a post-condition observation.

### Phase 6 — Isolated execution (weeks 13–15)
**Build:** Docker-backed `code_exec`, resource limits, network policy, approval for risky commands.
**Exit criteria:**
- Known-good and known-bad code samples run under limits; bad samples are rejected or killed at the limit.
- A test asserts `code_exec` cannot escape the workspace mount.
- A test asserts a network call fails when the task has no network budget.

### Phase 7 — Desktop (weeks 16–18)
**Build:** Windows UI automation adapter (pywinauto or accessibility APIs), screenshots, approval for sensitive actions.
**Exit criteria:**
- One repeatable desktop workflow completes with a visible action log.
- A test simulates an unexpected dialog and asserts recovery or `WAITING_FOR_APPROVAL`.

### Phase 8 — Image generation + specialist tools (weeks 19–21)
**Build:** provider-independent image interface, artifact saving with hash, `image_generate` tool.
**Exit criteria:**
- A test generates an image, saves it to the workspace, and verifies format + hash.
- A small end-to-end task uses research + code + image and produces a verified report.

### Phase 9 — Verification, evaluation, long-running work (weeks 22–26)
**Build:** benchmark harness (30–50 tasks), metrics, model comparison, checkpoint/progress summaries, bounded replanning polish.
**Exit criteria:**
- Benchmark runs reproducibly on a frozen suite with hidden holdouts.
- Metrics reported: success rate, verification pass rate, recovery rate, intervention count, cost per successful task, safety violations.
- A baseline comparison against one existing agent (e.g., OpenHands) on the same tasks is documented, including where AEGIS loses.

---

## 13. Benchmark and metrics (Phase 9, but designed now)

### 13.1 Task suite shape

- 30–50 tasks across: file manipulation, coding, testing, browser workflows, interrupted execution.
- Each task has: a written objective, machine-checkable acceptance criteria, a seed for deterministic setup, and a max budget.
- **Held-out set:** at least 10 tasks never used during development. This is the anti-overfitting control.

### 13.2 Metrics

| Metric | Definition | Target for v0.1 |
|--------|------------|-----------------|
| Task success rate | verifier passed / attempted | measured, not targeted |
| Verification pass rate | verifier passes / completion claims | ≥ 0.95 (i.e., almost no false claims) |
| Recovery rate | resumed tasks reaching terminal state / interrupted tasks | ≥ 0.80 |
| Intervention count | human approvals per task | ≤ 1 median |
| Cost per successful task | sum of all attempt costs / successes | measured |
| Safety violations | policy rejections that indicate a bypass attempt | 0 in the frozen suite |

Targets are **measured first, then tightened**. P9: no architecture change without a benchmark showing it helps.

---

## 14. Deviations from the original planning note

This section exists so the changes are explicit and reviewable.

1. **Minimal event log + checkpoint moved from Phase 4 to Phase 2.** Reason: retrofitting idempotency and checkpointing after the executor exists is a rewrite. Full recovery/resumption still lands in Phase 4.
2. **`Tool.result_model` is mandatory** (no `Any`). Reason: `output: Any` defeats structured validation (P1).
3. **Verifier is deterministic, not an LLM, in v1.** Reason: an LLM verifier is a second agent (P8) and is unmeasurable.
4. **Safety model made explicit (§3), with Docker demoted to defense-in-depth.** Reason: "a container alone is not a complete security guarantee" was true but the note didn't say what *was* the primary control. Now it does: dispatcher + workspace + allowlist.
5. **Replanning has explicit triggers and a budget.** Reason: "failure-driven replanning" was a differentiator but had no bound, which is how agents loop forever.
6. **Budget is a first-class field on `Task` and enforced by the controller.** Reason: "limit model expenditure" was a footnote; cost-per-success is a headline metric.
7. **Model provider decision is an explicit blocker for Phase 1.** Reason: the note said "start with one provider" but didn't make the choice a gate, and the choice affects structured-output and tool-call code.
8. **(Phase 4) Persistence is stdlib `sqlite3` with numbered migrations (`PRAGMA user_version`), not SQLAlchemy + Alembic, and the schema is `events` + `checkpoints` + `leases`, not the ten tables of §8.1.** Reason: one writer, and every other table in §8.1 can be derived from the event log plus the checkpoint, so the ORM and the extra tables would add two dependencies and a second source of truth without enabling anything yet. Revisit when the Phase 9 benchmark harness needs relational queries over many tasks. The recovery contract of §8.2 is implemented as written.
9. **(Phase 3) Side-effect class `execute` was added** between `workspace` and `external`. Reason: running the project's tests runs code the model wrote, which is more dangerous than writing a file, so declaring `workspace` never pre-approves it.
10. **(Phase 3) Approval continues a paused task by saving the paused step's conversation** (`PendingApproval`), because `aegis approve` runs in a separate process. Crash recovery (Phase 4) rebuilds the interrupted step from events instead.
11. **(Phase 4) One worker per task** is enforced with a lease and heartbeat, so two processes never drive the same task. Not in the original design; required for `resume` to be safe.
12. **(Phase 4) Idempotency keys** suppress an exact repeat of the most recent change when nothing else has changed since, returning the recorded result instead of running it again.
13. **(Phase 5) Replanning is built in Phase 5, not later**, because its exit criterion needs it. Triggers 1 (a step failed after its attempts) and 2 (the verifier rejected completion) are implemented. Trigger 3 is not a separate trigger: an unavailable tool (`ToolUnavailable`, outcome `unavailable`) fails its step, and that failure triggers the replan. Trigger 4 (human-approved replan) is NOT built. The replan keeps finished steps, asks the model only for the remaining steps, and is capped by `Budget.max_replans` (default 3; 0 turns it off). PLAN_REVISED is an event, crash recovery uses the latest plan, and `replay` checks the replan count against the budget.
14. **(Phase 5) The browser is text and DOM only.** There is no `browser_screenshot` (the free-tier model has no vision, and a screenshot would be unread evidence) and no tool that types (`browser_type`): the agent can read, follow links and press buttons, so it cannot log in or fill a form. `browser_click` takes an element id from the last page listing plus that element's exact text, and MUST state a post-condition (`expect_text` or `expect_url_contains`); without it the call is rejected, and if the page does not show it the call fails. Page-state evidence is `BROWSER_STATE`; new acceptance criteria are `page_contains:<text>` (any page the browser showed) and `url_contains:<text>` (the last page).
15. **(Phase 5) `external` is opened per host, not per call.** `--allow-host example.com` (or `*.example.com`, optionally `:port`) lets a task use the browser on that host only. Every request the page makes goes through the allowlist, and redirects are checked hop by hop (Chromium would otherwise follow a redirect to any host by itself: a test found this). Local and private addresses are refused unless named exactly. A task with no host gets no network tools at all. Known limits: a public name made to resolve to a private address (DNS rebinding) is not caught until the Phase 6 container; and allowing a host lets the agent click anything on its pages.
16. **(Phase 5) A browser lives for one run of a task.** After an approval pause or a `resume` the page is gone and the agent navigates again. The repeat guard is in memory too: after a `resume`, one identical repeat of an earlier failed call is allowed.
17. **(Phase 5) A call to a tool whose failure repeats** (`repeatable_after_failure = False`, currently `browser_click`) is refused with outcome `repeat_blocked` when the identical call already failed in the same step. The refusal is still a recorded TOOL_CALL and TOOL_RESULT (P5).
18. **(Phase 6) The sandbox is per task and opt-in** (`submit --sandbox`). With it, `shell_run_tests` runs in the container and `code_exec` becomes available. Without it, `code_exec` is not offered and is refused if called, so it can never run on the host. `shell_run_formatter` stays on the host (it runs ruff, not model code).
19. **(Phase 6) `code_exec` has no network at all.** Section 6 lists its network as "opt-in". Docker cannot enforce a host allowlist without a proxy, so a per-host grant like `--allow-host` would be dishonest for a container. Every sandbox container runs with `--network none`. Opting in to a network for containers is not built.
20. **(Phase 6) `code_exec` runs a workspace file, not a code string.** The file is written with `fs_write` first, so what runs is in the workspace, the event log and the diff. It reports which workspace files the run changed, and a changing run is FILE_DIFF evidence, so `tests_pass` goes stale after it. The sandbox never pulls images, containers are named and labelled (`aegis.sandbox=1`), and `aegis sandbox cleanup` removes strays after a crash. A killed worker can leave a container running until it ends by itself. Limits that are NOT enforced: total disk used by the workspace, and escapes of the kernel or runtime (section 3.3).

---

## 15. Open decisions (blockers)

| # | Decision | Blocks | Default if undecided |
|---|----------|--------|----------------------|
| D1 | First model provider (OpenAI / Anthropic / other) and monthly budget | Phase 1 | none — must decide |
| D2 | Approval UX: CLI `approve` only, or a minimal local web page | Phase 3 | CLI only |
| D3 | Docker availability on your Windows setup (WSL2 backend) | Phase 6 | defer, note in README |
| D4 | Benchmark task authoring: you write 30–50 tasks, or we generate and you curate | Phase 9 | you curate |
| D5 | Project rename (AEGIS is a codename) | cosmetic | keep AEGIS |

---

## 16. Repository layout (final)

```
AEGIS/
├── src/aegis/
│   ├── __init__.py
│   ├── main.py              # CLI entry
│   ├── config.py            # Pydantic settings, paths, budgets defaults
│   ├── models.py            # all Pydantic models from §5
│   ├── controller.py        # task lifecycle, budgets, approvals
│   ├── planner.py           # initial plan + bounded replan
│   ├── executor.py          # dispatch pipeline (§6.2)
│   ├── verifier.py          # deterministic checks (§9)
│   ├── memory.py            # SQLAlchemy session, event log, resume
│   ├── model_client.py      # ModelClient (§7)
│   ├── logging.py           # structured logging + redaction
│   └── tools/
│       ├── __init__.py      # explicit registry
│       ├── base.py          # Tool protocol, ToolContext
│       ├── calculator.py
│       ├── filesystem.py
│       ├── shell.py
│       ├── browser.py       # Phase 5
│       ├── code_exec.py     # Phase 6
│       ├── desktop.py       # Phase 7
│       └── image.py         # Phase 8
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/            # recorded model responses, sample workspaces
├── docs/
│   ├── design.md            # this document
│   ├── original-plan.md     # provenance
│   ├── threat-model.md      # §3 extracted for security review
│   └── benchmark.md         # Phase 9 task suite spec
├── data/                    # gitignored: aegis.db
├── workspaces/              # gitignored: per-task dirs
├── .env                     # gitignored
├── .env.example
├── .gitignore
├── pyproject.toml
├── alembic.ini              # Phase 4
└── README.md
```

---

## 17. First 30 days (working checklist)

Matches §12 Phase 0–2. Each item is a commit, not a vibe.

**Days 1–3 — Foundation**
- [ ] `python --version` ≥ 3.12, `uv --version`, `git --version`, `code --version` all succeed
- [ ] `mkdir AEGIS && cd AEGIS && git init && uv init --python 3.12`
- [ ] `uv add pydantic httpx rich && uv add --dev pytest ruff mypy`
- [ ] `.gitignore` written; `git check-ignore -v .env data/aegis.db workspaces/` matches all three
- [ ] `.env.example` written; `.env` created locally
- [ ] `src/aegis/main.py` prints `AEGIS initialized.` and `uv run python src/aegis/main.py` verifies it
- [ ] `src/aegis/config.py` + `src/aegis/models.py` skeletons from §5
- [ ] `tests/unit/test_config.py` rejects a negative budget; `uv run pytest` passes
- [ ] `uv run ruff check .` and `uv run mypy src` pass
- [ ] First commit: `chore: initialize AEGIS project`

**Days 4–7 — Model client**
- [ ] **D1 decided** (provider + budget)
- [ ] `model_client.py` implemented with timeout + bounded retry
- [ ] Redaction filter tested
- [ ] Malformed-response fixture test passes
- [ ] 429 fixture test passes
- [ ] `BudgetUsage.dollars` updated from a real (or recorded) call

**Week 2 — Tools**
- [ ] `tools/base.py` with `Tool` protocol and `ToolContext`
- [ ] `calculator` tool with arg/result models
- [ ] `fs_list` and `fs_read` with workspace enforcement
- [ ] Dispatcher rejects unknown tool, bad args, out-of-workspace path — all as tests
- [ ] Every tool call writes a `TOOL_CALL` event and a `TOOL_RESULT` event

**Week 3 — Agent loop**
- [ ] Controller loop with `TaskState` transitions
- [ ] Minimal event log + checkpoint table (the Phase 2 deviation)
- [ ] Max-step, max-tool-call, wall-clock, dollar limits enforced
- [ ] Failed tool call test + cancellation test
- [ ] `usage` recomputed from events in a test

**Week 4 — First deliverable**
- [ ] End-to-end: submit an objective, create a small project in the workspace, run its tests, get a verified report
- [ ] Verifier passes only with fresh `TEST_PASS` evidence (staleness rule tested)
- [ ] `aegis report <id>` prints artifacts + known limitations
- [ ] `docs/benchmark.md` drafted with at least 10 candidate tasks (not yet run)

---

## 18. What "done" means for v0.1

AEGIS v0.1 is done when, on a fresh clone and a fresh `.env`:

1. `aegis submit "<objective>" --accept "tests pass"` runs to a terminal state.
2. The final report includes a `VerificationResult` and a list of artifacts with hashes.
3. The event log replays cleanly (`aegis report --replay` exits 0).
4. Killing the process mid-task and running `aegis resume <id>` either completes or moves to `WAITING_FOR_APPROVAL` — never silently duplicates a side effect.
5. The benchmark suite in `docs/benchmark.md` runs and produces the six metrics in §13.2.
6. `ruff`, `mypy`, and `pytest` are green in CI (even if CI is just a local pre-commit hook at this stage).

Anything beyond that is v0.2+.

---

## 19. Change log

| Version | Date | Change |
|---------|------|--------|
| 0.1 | — | Initial design doc. Supersedes the original planning note. Incorporates the review points: Phase 2 persistence, mandatory result models, deterministic verifier, explicit threat model, bounded replanning, first-class budgets, provider decision as a blocker. |
| 0.2 | 2026-10 | Implementation notes for Phases 2 to 4: deviations 8 to 12 in section 14. No change to the contracts in sections 5 to 9. |
| 0.3 | 2026-10 | Implementation notes for Phase 5: deviations 13 to 17 in section 14 (replanning built now, text-only browser, per-host `external`, browser lifetime, repeat guard). |
| 0.4 | 2026-10 | Implementation notes for Phase 6: deviations 18 to 20 in section 14 (per-task `--sandbox`, no container network, `code_exec` runs files). |

---

**One thing this document deliberately does not do:** it does not claim AEGIS will beat existing agents. §13 makes the comparison measurable, and §18 makes "done" checkable. That is the honest version of the ambition.