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
| 3 | Write and edit tools, allowlisted shell | next |

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
`answer_contains:<text>`. A task only completes if the verifier finds matching tool evidence.

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
