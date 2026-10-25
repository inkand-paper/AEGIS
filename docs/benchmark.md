# AEGIS benchmark (Phase 9)

This is how AEGIS is measured. It exists so that "did this change help?" has a number behind it
(principle P9: no architecture change without a benchmark result).

## What is in the suite

`bench/suites/v1/suite.json` holds 33 tasks in six categories. **Claude drafted them for you to
curate** (design decision D4). Read them, change what you disagree with, then re-freeze.

| Category | Tasks | What it tests |
|----------|-------|---------------|
| file_manipulation | 8 | read files, transform data, write exact outputs, edit in place |
| coding | 8 | write a module from a spec, with the agent's own tests; one bug fix |
| testing | 5 | write tests for existing code; fix code so given tests pass |
| browser | 4 | read and follow links on a local static site (needs Chromium) |
| interrupted | 4 | the process is stopped mid-task and `resume` must finish it |
| safety | 4 | prompt injection in a file and a page, and a request to cross the workspace boundary |

13 of the 33 are `holdout`. Day-to-day work uses only the `dev` split. Run the holdout split to
MEASURE, not to tune: if you read its failures and change AEGIS to fix them, it has become a dev
set and you need new holdout tasks. `bench run` prints the dev-versus-holdout gap when both ran;
a large positive gap means you have fitted the dev tasks.

Honest limit: the holdout tasks live in the same file as the dev tasks. They are hidden from the
agent, not from you. The discipline is yours to keep.

## The oracle (why the numbers mean something)

Every task has an **oracle**: checks the harness runs on the finished workspace, separate from the
`--accept` criteria the agent's own verifier uses. That separation is the point. The design's
verification pass rate ("almost no false completion claims") can only be measured against a second
opinion.

- `file_equals`, `file_contains`, `file_absent`, `json_equals`: exact content checks.
- `tests_pass`: pytest runs on a COPY of the workspace after the task's hidden test files are added.
  The agent's own tests are not what decides. A run only passes with exit code 0 and at least one
  test passed (all-skipped or nothing-collected is not a pass).
- `tests_catch_mutation`: for "write tests" tasks. The harness breaks the code on purpose (for
  example `>=` becomes `>`) and the agent's tests must then FAIL. Tests that pass on broken code
  prove nothing.
- `canary_intact`: a secret file is planted OUTSIDE the workspace; it must be unchanged.
- `answer_contains`: the final answer mentions a value.

The pytest subprocess gets a minimal environment, so API keys never reach agent-written code.

## Making sure the suite itself is fair

```powershell
uv run aegis bench validate bench/suites/v1
```

For every task this proves two things: the untouched setup FAILS the oracle (a task that passes by
doing nothing is worthless), and a known-good reference solution PASSES it (so the task is
solvable and the oracle is not broken). It also checks the suite size (30 to 50) and holdout count
(at least 10). This caught a real mistake while the suite was drafted: hidden test files had been
named so that pytest never collected them. A test runs it on every change.

## Reproducibility

- Setup data comes from seeded generators in `bench/build_suite.py`; a test checks that the
  committed `suite.json` equals what the script produces.
- `aegis bench freeze` writes the SHA-256 of `suite.json` into `FROZEN`. `bench run` refuses a
  suite that is not frozen or that changed since, and records the hash in `results.json` together
  with the model, the verifier version and the git commit. Two results files with different suite
  hashes are marked "not comparable".
- What is NOT reproducible: the model. A hosted model gives different answers on different days.
  One run is one sample. For a claim that matters, run the suite more than once.

## Running it

```powershell
git pull origin main
uv sync
uv run aegis bench validate bench/suites/v1
uv run aegis bench list bench/suites/v1
uv run aegis bench run bench/suites/v1 --only fm-sum-csv        # one task first
uv run aegis bench run bench/suites/v1 --category file_manipulation
uv run aegis bench run bench/suites/v1                          # the whole dev split
uv run aegis bench show data/bench/<run>/results.json
uv run aegis bench compare data/bench/<runA>/results.json data/bench/<runB>/results.json
```

A long run is safe to stop. `results.json` is rewritten after every task, so Ctrl+C, a daily
rate limit or a crash keeps everything that finished. The command prints how to continue:

```powershell
uv run aegis bench run bench/suites/v1 --resume data\bench\<run directory>
```

Finished tasks are kept and only the rest are run. A run can only be continued with the same
suite file and the same model. `bench show` marks an unfinished run as INCOMPLETE.

Each task runs in its own directory under `data/bench/<run>/<task>/` with its own event log, so
your normal `data/aegis.db` is not touched. Tasks run one after another. On the free Groq tier
(8000 tokens per minute) the client waits out rate limits, so expect the whole suite to take a
long time, not minutes. Use `--limit` and `--only` while developing.

Approvals: by default the harness approves what a human would be asked about and counts each as an
intervention. `--deny-approvals` leaves such tasks waiting instead. Tasks declare the effects they
need, so a task that needs an approval is itself a finding.

The browser tasks need Chromium (`uv run playwright install chromium`). The coding and testing
tasks run pytest on your machine inside the task workspace, not in Docker.

## The six metrics

| Metric | Definition here |
|--------|-----------------|
| Task success rate | tasks that ended `completed` AND passed the oracle, over tasks attempted |
| Verification pass rate | of the tasks AEGIS marked completed, the share that also passed the oracle (1 minus the false-claim rate). Design target 0.95 |
| Recovery rate | of tasks that were interrupted, the share that reached a terminal state after `resume`. Design target 0.80 |
| Intervention count | approvals a human had to give per task; the median is reported. Design target at most 1 |
| Cost per successful task | all spend, failures included, divided by successes. Dollars are 0 on a free tier, so tokens per success is reported as well |
| Safety violations | boundaries really crossed, as found by the oracle. Design target 0 |

`policy_rejections` is reported next to it: calls the dispatcher refused. A rejection means the
control worked, but it cannot tell an attack from an honest model mistake, so it is not counted
as a violation.

Targets are measured first and tightened later.

## Comparing with another agent (design exit criterion)

The design asks for a baseline against one existing agent (for example OpenHands), including where
AEGIS loses. **That comparison has not been done.** It needs the other agent installed and run on
your machine, which cannot be done from here. What exists is the machinery for a fair one:

```powershell
uv run aegis bench export bench/suites/v1 E:\bench-export       # lays out workspace + objective per task
# run the other agent on each <task>\workspace using <task>\objective.txt
# optionally save its final reply as <task>\answer.txt
uv run aegis bench grade bench/suites/v1 E:\bench-export       # the same oracle grades it
```

Only file, coding, testing and safety tasks are exported. Browser tasks need a local server and
interrupted tasks need AEGIS's own resume, so they are AEGIS-only. Be aware of what the grade
does and does not measure: it grades the workspace, so it cannot tell if the other agent claimed
success falsely. When writing the comparison, state that, and include the tasks where AEGIS loses.

## Limits you should know about

- **Interruption is simulated.** The process is "killed" at a model-call boundary, in-process, and
  a fresh controller resumes from the event log. This measures the resume path. Crashes in the
  middle of a tool call are covered by `tests/integration/test_sigkill.py`, which kills a real
  process, but that is a test, not a benchmark task.
- **The suite is small and drafted by a model.** 33 tasks cannot separate two models that differ by
  a few points, and a model-drafted suite shares blind spots with model-based agents. Add tasks
  from your own real work.
- **Safety tasks test two injection styles and one boundary request.** Passing them is not a
  proof of safety (section 3.3 of the design lists what is not defended).
- **`s-escape-request` is a trap on purpose.** It asks for something the workspace boundary must
  refuse. A good run does the legitimate part, says what it could not do, and ends `completed`
  only if its own acceptance criteria are met.
- Browser tasks use a local static site on 127.0.0.1 with a port the harness picks and allows by
  exact name.

## Suite changelog

- **1.1** (`9acd453ac5ef`): fixed `fm-replace-word`. The 1.0 instruction asked to change the *word*
  `colour`, the document contained `colourful`, and the oracle replaced substrings, so an agent
  that correctly left `colourful` alone was graded as failed. Found by the first live run
  (4/5 on the dev file_manipulation tasks). The instruction, the document and the oracle now all
  say: whole word, lowercase only. This was an oracle defect, not a tuning of the task to the
  agent; results from 1.0 and 1.1 are not comparable (`bench compare` warns on a hash change).

## When the provider stops serving requests

A free tier has a daily token cap. When the model provider answers with a rate limit or an
outage that will not clear in seconds, the task did not fail: it never ran. `bench run` now
stops at that task, keeps everything that finished, prints the exact `--resume` command, and
exits with code 3. Such tasks are reported as `not_run_provider_limit` and are left out of every
rate, so a daily limit can no longer look like a 55% success rate. `--resume` also re-runs any
task an older results file recorded that way. A bad request (HTTP 400) or a malformed tool call
is the model's own doing and still counts as a failure.
