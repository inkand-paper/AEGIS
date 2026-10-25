# AEGIS

[![CI](https://github.com/inkand-paper/AEGIS/actions/workflows/ci.yml/badge.svg)](https://github.com/inkand-paper/AEGIS/actions/workflows/ci.yml)

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
| 4 | Durable state, crash recovery (`resume`), event-log replay | done |
| 5 | Replanning, network allowlist, browser (Playwright) | done |
| 6 | Docker sandbox for running code (`--sandbox`, `code_exec`) | done |
| 7 | Windows desktop automation (`--allow-app`, `actions` log) | built, real backend unverified |
| 8 | Image generation (`image_generate`, `image_saved` check) | built, hosted provider unverified |
| 9 | Benchmark harness: 33 tasks, independent oracle, six metrics (`aegis bench`) | built, not yet run with a real model |

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
`answer_contains:<text>`, `file_written:<path>`, `tests_pass`, `page_contains:<text>`,
`url_contains:<text>`, `desktop_contains:<text>`, `image_saved:<path>`. A task only completes if the
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

### If the process dies: resume, report, replay

Every step is written to the append-only event log before it happens (`data/aegis.db`). If a task
is cut off (power cut, killed terminal, crash), carry it on with:

```bash
uv run aegis resume <task_id>
uv run aegis report <task_id> --replay   # final report, files with hashes, and a log consistency check
```

`resume` does not start over. It rebuilds where the task was from the log and carries on from the
interrupted step. A tool call that was cut off is closed on the record as interrupted. A safe-to-repeat
call (a read, a test run) runs again. A call that may have half-run, such as a file write, is NEVER
repeated on its own: the task waits for your `aegis approve`, so a side effect is never duplicated
silently. Usage (calls, tokens, dollars) is recomputed from the log, and any difference from what was
stored is reported and corrected.

Only one process may work on a task at a time. A task whose worker was killed becomes resumable once
its lease runs out, about 30 seconds after the last heartbeat. Before that, `resume` tells you who
holds it. The database is upgraded automatically when you update AEGIS (a copy is kept next to it
first), and a database written by a newer AEGIS is refused rather than damaged.

`report --replay` replays the log and exits 1 if anything breaks the rules (an illegal state change,
a call without a result, evidence from a failed call, COMPLETED without a passing verification).

## Replanning

When a step fails after its attempts, or the verifier rejects the result, AEGIS does not just stop.
It asks the model for a new plan for the REMAINING work (finished steps are kept, not redone) and
tells it what went wrong and not to repeat that approach. This is bounded: at most 3 replans by
default (`--max-replans N`, `0` turns it off). After that the task fails with `ReplanBudgetExceeded`.
Every replan is a `PLAN_REVISED` event, so `aegis report <id> --replay` shows it.

## Browser

The agent can read and click through web pages, but only on hosts you name. A task starts with NO
network access:

```bash
uv run playwright install chromium     # once, downloads the browser (about 150 MB)
uv run aegis submit "Open http://localhost:8000 and tell me what the product list shows" \
    --allow-host localhost:8000 --accept page_contains:Item
```

Tools: `browser_navigate` (open a URL), `browser_read` (read the page again) and `browser_click`.
Pages are given to the model as text plus a numbered list of links and buttons. There is no
screenshot, and there is NO tool that types, so the agent cannot log in or fill in a form.

- `--allow-host example.com` allows that exact host; `*.example.com` allows its subdomains (not
  `example.com` itself); add `:port` to restrict the port. Repeat the flag for more hosts.
- Every request the page makes is checked, not just the first one: images, scripts and redirects
  to other hosts are blocked, and the blocked hosts are reported to the agent. Local and private
  addresses (`localhost`, `127.0.0.1`, `192.168.x.x`, `169.254.x.x`) are refused unless you name
  them exactly. New windows, downloads and WebSockets are blocked.
- A click must say what the page should show afterwards. If the page does not show it, the click
  FAILS and the agent is told what the page shows instead. An identical failed click is refused if
  repeated, so the agent has to choose something else (or the plan is revised).
- `page_contains:<text>` (a page the browser really showed) and `url_contains:<text>` (the last
  page) are acceptance criteria, so a task cannot complete on a page that was never shown.
- Read this before allowing a host: the agent can click ANYTHING on that host's pages that a
  visitor could click (buy, delete, send). Allow only hosts where that is acceptable.
- The browser exists for one run. After an approval pause or `resume`, the agent opens the page
  again. If Chromium is not installed the browser tools report that, and the browser tests are
  skipped. `AEGIS_BROWSER_EXECUTABLE` can point to another Chromium.

## Sandbox (Docker)

`shell_run_tests` normally runs the model's code on your computer, as you. With `--sandbox` the
code runs in a Docker container instead, and the agent also gets `code_exec`, which runs a Python
file from the workspace in the container:

```powershell
uv run aegis sandbox build      # once: builds the image (downloads python:3.13-slim)
uv run aegis sandbox check      # is Docker running and the image present?
uv run aegis submit "Write fib.py that prints the first 10 Fibonacci numbers and run it" `
    --sandbox --allow-writes --allow-exec --accept result_contains:34
```

Docker Desktop (WSL2 backend on Windows) must be running. Inside the container:

- no network at all (not even DNS), so downloads, API calls and `pip install` all fail;
- only the task workspace is visible, at `/workspace`; the rest of the file system is read-only
  and nothing of your computer is mounted;
- no Linux capabilities, no privilege gain, none of your environment variables (no API keys);
- limits: 256 MB memory (no swap), 1 CPU, 64 processes, 256 open files, 50 MB per file, 30 s per
  run (`AEGIS_SANDBOX_*` in `.env`). A run that goes over is killed and the result says why.

`code_exec` only runs a file that is already in the workspace (write it with `fs_write` first),
and it never runs on your computer: without `--sandbox` the tool is not offered. It still counts
as "execute", so it needs `--allow-exec` or your approval. If a run is cut off by a crash,
`aegis sandbox cleanup` removes leftover containers.

What this does NOT promise: a container is defense in depth, not a guarantee. A bug in Docker or
the kernel could let code out. The workspace is writable, so a script can fill the disk with many
files, and it can read and change everything in the workspace, so never put secrets there.
`--sandbox` does not make `--allow-host` browsing safer: the browser runs on your computer, not in
the container.

## Desktop (Windows)

With `--allow-app` the agent can see and use the windows of the programs you name. Windows only,
and it needs the screen of your computer to be unlocked while it works. A task starts with NO
desktop access.

```powershell
uv run python scripts/desktop_smoke.py     # check the Windows backend by hand, no model needed
uv run aegis submit "Open Notepad, type 'Meeting at noon' and save it as note.txt" `
    --allow-app notepad.exe --accept desktop_contains:note.txt
uv run aegis actions <task_id> --desktop   # the readable log of every desktop action
```

Tools: `desktop_windows`, `desktop_launch`, `desktop_inspect`, `desktop_click`, `desktop_type`
and `desktop_screenshot` (saved as a PNG with its hash under `screenshots/` in the workspace).

- Only windows of the allowed programs are visible. Terminals, script hosts, browsers, Explorer,
  installers and system tools (`cmd`, `powershell`, `python`, `chrome`, `explorer`, `regedit` ...)
  cannot be allowed at all: they would bypass the other controls. The list is best effort, so allow
  only programs you would let a stranger use.
- Controls are used through their accessibility actions (press, select, toggle, set text), never
  by moving the mouse to a position, so a window covering another one cannot take a click meant for
  something else. A control is named by id AND exact name, and a mismatch is refused.
- Every click must say what should be true afterwards (text shown, a window title present or gone).
  If not, the call FAILS and lists any unexpected dialog and its buttons, so the agent can
  dismiss it or report it instead of clicking on.
- A click on a control whose name looks sensitive (Delete, Yes, Send, Pay, Install, Replace, Sign
  in, Run ...) ALWAYS waits for your `aegis approve`, even with `--allow-writes` and `--allow-exec`.
  The same goes for anything you could not undo. The agent never types into password-like fields.
- `desktop_contains:<text>` is an acceptance criterion: text a program window really showed.
- The programs the agent starts stay open when the task ends. Close them yourself.
- Packaged (Store) apps appear to Windows under `ApplicationFrameHost.exe`, so they cannot be
  allowed by their own name. Use classic programs.
- `desktop_screenshot` shows everything visible in that window, and the free-tier model cannot read
  images anyway. It is there for you, as evidence.

NOT verified yet: the Windows backend (`src/aegis/desktop_windows.py`) has only been checked on
Linux, with a simulated desktop. The first real run is `scripts/desktop_smoke.py`.

## Images

`image_generate` makes one picture from a prompt and saves it in the task workspace. Which service
makes it is a setting, not code (`.env`):

```powershell
# default: offline test provider. No network, no account, no cost.
uv run aegis submit "Make a 256x256 cover image called cover.png" --allow-writes `
    --accept image_saved:cover.png

# a hosted provider (needs your own key; the task must also name the host)
#   AEGIS_IMAGE_PROVIDER=openai          (or custom, with AEGIS_IMAGE_BASE_URL and AEGIS_IMAGE_MODEL)
#   AEGIS_IMAGE_API_KEY=...
uv run aegis submit "Draw a lighthouse at dusk, lighthouse.png" --allow-host api.openai.com `
    --accept image_saved:lighthouse.png
```

- The default provider (`procedural`) draws a deterministic ABSTRACT picture from the prompt. It
  does not show what the prompt says. It exists so the pipeline can be run and tested for free,
  and the tool tells the model so, so a report should not claim the image depicts anything.
- Hosted providers speak the OpenAI `/images/generations` protocol and must return the picture
  inline (base64). A reply that only holds a URL is refused, so no second host is ever contacted.
  A hosted call is `external`: it needs `--allow-host <the provider host>` (which also pre-approves
  it), and your prompt is sent to that service. One attempt only, never a silent retry, because a
  generation can cost money.
- The bytes are inspected before anything is saved: PNG (every chunk checksum), JPEG, GIF and
  WebP must be well formed, at most 4096 per side, 16 million pixels and 10 MB. The file name must
  match the real format. An existing file is only replaced with `overwrite`.
- `image_saved:<path>` passes only if the file on disk RIGHT NOW is still a valid image with the
  SHA-256 recorded when it was made. Replace or damage it afterwards and the task fails.
- The check is structural. It proves the file is a real image of the stated size and unchanged.
  It cannot tell whether the picture is any good or matches the prompt.

NOT verified yet: the hosted provider (`OpenAIImagesProvider`) is tested against a simulated
server only. The first real call needs your key.

## Benchmark

`aegis bench` measures AEGIS against a frozen suite of 33 tasks (13 held out) with an oracle that
is independent of the agent's own verifier. Full description: [docs/benchmark.md](docs/benchmark.md).

```powershell
uv run aegis bench validate bench/suites/v1             # prove the tasks are fair (no model needed)
uv run aegis bench run bench/suites/v1 --only fm-sum-csv
uv run aegis bench run bench/suites/v1                  # the dev split; slow on a free tier
uv run aegis bench show data/bench/<run>/results.json
```

It reports task success, verification pass rate (false completion claims), recovery rate,
interventions, tokens and dollars per success, and safety violations. NOT done yet: a run with a
real model, and the baseline comparison with another agent (`bench export` and `bench grade`
prepare it).

## Development

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```

Browser tests start a real headless Chromium against local test pages (run `uv run playwright
install chromium` first, otherwise they are skipped). Some filesystem tests create symlinks and are skipped where the OS does not allow it (on Windows,
enable Developer Mode).

## Safety model

The model never executes anything. It proposes tool calls, and the dispatcher validates arguments,
enforces the workspace boundary and policy, records the call, runs it with a timeout, and validates
the result. See section 3 of the design document.

What is enforced for file tools: paths are resolved inside the task workspace, `..` and absolute
paths and symlinks that lead outside are refused, writes are atomic, and every call (including
refused ones) is in the append-only event log.

What is NOT isolated unless you pass `--sandbox`, so read this before using `--allow-exec`:
`shell_run_tests` runs pytest on code the model wrote, as your user, on your computer. The command line is fixed, there is no shell,
the environment is scrubbed of your API keys, output is capped and a hung run is killed. But that
code can still read your files and use the network. Review the workspace before you approve, and do
not run it on a machine or a folder you cannot afford to lose, or use `--sandbox`. The test and
format runners use the pytest and ruff from your dev dependencies (`uv sync`).
