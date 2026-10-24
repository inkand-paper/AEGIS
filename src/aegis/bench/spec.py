"""Suite and task specification, and the freeze hash that pins a suite."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Category = Literal["file_manipulation", "coding", "testing", "browser", "interrupted", "safety"]
Split = Literal["dev", "holdout"]
CheckKind = Literal[
    "file_exists",
    "file_absent",
    "file_equals",
    "file_contains",
    "file_not_contains",
    "json_equals",
    "tests_pass",
    "tests_catch_mutation",
    "answer_contains",
    "canary_intact",
]
FROZEN_NAME = "FROZEN"


class Check(BaseModel):
    """One oracle check on the finished workspace.

    file_exists / file_absent        `path`
    file_equals                      `path`, `value` (trailing whitespace ignored)
    file_contains / _not_contains    `path`, `value` (case-insensitive when `ignore_case`)
    json_equals                      `path`, `value` is JSON text; compared after parsing
    tests_pass                       runs pytest on a COPY of the workspace after adding the
                                     hidden `files` (the agent's own tests are not trusted)
    tests_catch_mutation             the agent's own tests must FAIL once `path` has `old`
                                     replaced by `new` (so the tests really test something)
    answer_contains                  `value` is in the agent's final answer (case-insensitive)
    canary_intact                    the file planted OUTSIDE the workspace is unchanged
    """

    model_config = ConfigDict(extra="forbid")

    kind: CheckKind
    path: str | None = None
    value: str | None = None
    ignore_case: bool = False
    files: dict[str, str] = Field(default_factory=dict)
    old: str | None = None
    new: str | None = None

    @model_validator(mode="after")
    def _complete(self) -> "Check":
        needs_path = {
            "file_exists",
            "file_absent",
            "file_equals",
            "file_contains",
            "file_not_contains",
            "json_equals",
            "tests_catch_mutation",
        }
        needs_value = {
            "file_equals",
            "file_contains",
            "file_not_contains",
            "json_equals",
            "answer_contains",
        }
        if self.kind in needs_path and not self.path:
            raise ValueError(f"{self.kind} needs a path")
        if self.kind in needs_value and self.value is None:
            raise ValueError(f"{self.kind} needs a value")
        if self.kind == "tests_catch_mutation" and (self.old is None or self.new is None):
            raise ValueError("tests_catch_mutation needs old and new")
        if self.kind == "json_equals":
            try:
                json.loads(self.value or "")
            except ValueError as exc:
                raise ValueError(f"json_equals value is not JSON: {exc}") from exc
        return self


class Setup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: dict[str, str] = Field(default_factory=dict)  # put in the workspace before the run
    site: dict[str, str] = Field(default_factory=dict)  # served over HTTP: {site_url} in text
    canary: str | None = None  # secret text planted outside the workspace


class Reference(BaseModel):
    """A known-good solution. `bench validate` applies it to prove the oracle can pass, and
    proves the untouched setup FAILS the oracle (so no task passes by doing nothing)."""

    model_config = ConfigDict(extra="forbid")

    files: dict[str, str] = Field(default_factory=dict)
    answer: str = ""


class BenchBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_model_calls: int = Field(default=60, ge=1)
    max_tool_calls: int = Field(default=60, ge=1)
    max_replans: int = Field(default=2, ge=0)


class BenchTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    category: Category
    split: Split = "dev"
    objective: str  # may use {site_url}
    setup: Setup = Field(default_factory=Setup)
    accept: list[str] = Field(default_factory=list)  # the agent-side `--accept` criteria
    allow_writes: bool = True
    allow_exec: bool = False
    allow_hosts: list[str] = Field(default_factory=list)  # may use {site_host}
    budget: BenchBudget = Field(default_factory=BenchBudget)
    interrupt_after_model_calls: int | None = Field(default=None, ge=1)
    oracle: list[Check] = Field(min_length=1)
    reference: Reference
    seed: int | None = None  # the seed `build_suite.py` used to make this task's data

    @model_validator(mode="after")
    def _consistent(self) -> "BenchTask":
        if self.category == "interrupted" and self.interrupt_after_model_calls is None:
            raise ValueError("an interrupted task must say when to interrupt")
        if self.category != "interrupted" and self.interrupt_after_model_calls is not None:
            raise ValueError("only an interrupted task may set interrupt_after_model_calls")
        if self.category == "browser" and not self.setup.site:
            raise ValueError("a browser task needs a site")
        if any(c.kind == "canary_intact" for c in self.oracle) and self.setup.canary is None:
            raise ValueError("canary_intact needs setup.canary")
        return self


class Suite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    description: str = ""
    tasks: list[BenchTask] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> "Suite":
        ids = [t.id for t in self.tasks]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate task ids: {dupes}")
        return self


def suite_path(location: Path) -> Path:
    return location / "suite.json" if location.is_dir() else location


def suite_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_suite(location: Path) -> tuple[Suite, str]:
    """Return the suite and the SHA-256 of its file."""
    path = suite_path(location)
    return Suite.model_validate_json(path.read_text(encoding="utf-8")), suite_digest(path)


def frozen_digest(location: Path) -> str | None:
    marker = suite_path(location).parent / FROZEN_NAME
    if not marker.is_file():
        return None
    return marker.read_text(encoding="utf-8").split()[0]


def freeze(location: Path) -> str:
    path = suite_path(location)
    load_suite(location)  # refuse to freeze an invalid suite
    digest = suite_digest(path)
    (path.parent / FROZEN_NAME).write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def select(
    suite: Suite,
    *,
    split: str = "dev",
    only: list[str] | None = None,
    category: str | None = None,
    limit: int | None = None,
) -> list[BenchTask]:
    tasks = [t for t in suite.tasks if split == "all" or t.split == split]
    if only:
        unknown = set(only) - {t.id for t in suite.tasks}
        if unknown:
            raise ValueError(f"unknown task ids: {sorted(unknown)}")
        tasks = [t for t in suite.tasks if t.id in only]
    if category:
        tasks = [t for t in tasks if t.category == category]
    return tasks[:limit] if limit else tasks
