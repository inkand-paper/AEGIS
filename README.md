# AEGIS

A long-running AI agent. You describe an outcome, AEGIS plans it, does the work through a
validated tool dispatcher, checks the work against evidence, and reports what actually happened.

The full design, safety model and phase plan are in [docs/design.md](docs/design.md).

## Status

| Phase | What | State |
|-------|------|-------|
| 0 | Project foundation, config, models | done |
| 1 | Provider-agnostic model client, retry, redaction | done |
| 2 | Tools, dispatcher, event log, agent loop, verifier | done |
| 3 | Write and edit tools, allowlisted shell, approval flow | done |

## Setup

Requires Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/inkand-paper/AEGIS.git
cd AEGIS
uv sync
cp .env.example .env        # Windows PowerShell: copy .env.example .env
```

Edit `.env` and set your provider and key. Groq has a free tier and is the default:

```ini
AEGIS_PROVIDER=groq
AEGIS_API_KEY=your_key_here
AEGIS_MODEL=openai/gpt-oss-120b
```

`uv run python scripts/list_models.py` lists the models your key can use. To switch provider,
change only `AEGIS_PROVIDER` and `AEGIS_API_KEY` (supported: groq, gemini, openai, anthropic,
openrouter, ollama, or `custom` with `AEGIS_BASE_URL` and `AEGIS_MODEL`).

Free tiers may use your prompts to improve their models. Never send secrets or private code.

## Usage

```bash
# Run a task to a terminal state. --accept adds checks the verifier must pass.
uv run aegis submit "What is 17 * 23? Use the calculator." --accept result_contains:391

# Give the agent files to work on (copied into the task workspace)
uv run aegis submit "Summarise notes.txt" --attach notes.txt --accept file_read:notes.txt

# Inspect a task later
uv run aegis status <task_id>
```

Acceptance criteria: `tool_used:<tool>`, `result_contains:<text>`, `file_read:<path>`,
`answer_contains:<text>`, `file_written:<path>`, `tests_pass`. A task only completes if the
verifier finds matching tool evidence. `tests_pass` needs a `shell_run_tests` run that exited 0
with at least one passing test, and it goes stale: if any file changed after that run, the task
must run the tests again.

### Writing files, running tests, and approvals

The agent can create and edit files (`fs_write`, `fs_edit`), run the project's tests
(`shell_run_tests`) and format code (`shell_run_formatter`). Each tool call has a side effect
class, and a call whose class the task did not declare up front waits for you:

| Class | Examples | How to pre-approve |
|-------|----------|--------------------|
| none | reading, listing, calculator, dry runs, `check_only` | never needs approval |
| workspace | creating or editing files, reformatting | `--allow-writes` |
| execute | running tests (runs code the model wrote) | `--allow-exec` |

```bash
uv run aegis submit "Fix the bug in calc.py so the tests pass" --attach calc.py --attach test_calc.py --accept tests_pass
# exit code 2: the task is waiting. Look at what it wants, and at the workspace files.
uv run aegis status <task_id>
uv run aegis approve <task_id>   # shows the call, asks y/N, then carries the task on
uv run aegis cancel <task_id>    # denies it instead
```

`approve` shows the exact arguments (file content is shown line by line) and the workspace path.
Another approval may follow later in the same task. Use `--yes` only in scripts you trust.
While waiting, the clock does not run: waiting time does not count against the wall-clock budget.

Exit codes: `0` completed, `1` failed, `2` waiting for approval, `3` cancelled, `4` usage error.

Each task works inside its own folder, `workspaces/<task_id>/`. State is kept in `data/aegis.db`.
Both are gitignored.

## Development

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```

Some filesystem tests create symlinks and are skipped where the OS does not allow it (on Windows,
enable Developer Mode).

## Safety model

The model never executes anything. It proposes tool calls, and the dispatcher validates arguments,
enforces the workspace boundary and policy, records the call, runs it with a timeout, and validates
the result. See section 3 of the design document.

What is enforced for file tools: paths are resolved inside the task workspace, `..` and absolute
paths and symlinks that lead outside are refused, writes are atomic, and every call (including
refused ones) is in the append-only event log.

What is NOT isolated yet, so read this before using `--allow-exec`: `shell_run_tests` runs pytest on
code the model wrote, as your user, on your computer. The command line is fixed, there is no shell,
the environment is scrubbed of your API keys, output is capped and a hung run is killed. But that
code can still read your files and use the network. Review the workspace before you approve, and do
not run it on a machine or a folder you cannot afford to lose. A container sandbox arrives with
Phase 6. The test and format runners use the pytest and ruff from your dev dependencies (`uv sync`).
