"""Docker sandbox for running model-written code (design doc section 3, principle 6; Phase 6).

Be clear about what this is. The primary controls stay the dispatcher, the workspace boundary and
the approval gate. The container is defense in depth: if the model got code approved, that code
still runs with

  * no network at all (`--network none`: only a loopback device exists),
  * a read-only root file system, and a small no-exec scratch /tmp,
  * ONLY the task workspace mounted (read-write) at /workspace, nothing else of the host,
  * no Linux capabilities, no privilege gain, a non-default user,
  * a memory cap (no swap), a CPU cap, a process-count cap, an open-files cap, a file-size cap,
  * a wall-clock limit, after which the container is killed by name.

What it does NOT do: a container is not a complete security guarantee. A kernel or runtime
escape is out of scope (design doc section 3.3), the workspace is writable so a script can fill
the disk with many files, and a script can read and change everything inside the workspace.
Do not put secrets in a workspace.

The sandbox talks to Docker only through the `docker` command line. It never pulls images
(`--pull never`), so a missing image is a clear error instead of a surprise download.
"""

import asyncio
import contextlib
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "aegis-sandbox:latest"
LABEL = "aegis.sandbox=1"
MAX_OUTPUT_CHARS = 6000
CONTAINER_WORKDIR = "/workspace"

# The docker CLI needs a few variables to find its daemon and config. Nothing else is passed on:
# no API keys, no tokens. (The container itself starts with an empty environment.)
_DOCKER_ENV_KEEP = (
    "PATH", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "SYSTEMROOT", "SYSTEMDRIVE",
    "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "TMPDIR", "XDG_RUNTIME_DIR",
    "XDG_CONFIG_HOME", "LANG", "LC_ALL",
)  # fmt: skip


class SandboxUnavailable(Exception):
    """Docker or the sandbox image cannot be used. The message says what to do."""


@dataclass(frozen=True)
class SandboxLimits:
    memory_mb: int = 256
    cpus: float = 1.0
    pids: int = 64
    timeout_s: float = 30.0
    max_file_mb: int = 50  # largest single file a script can write
    tmp_mb: int = 64
    open_files: int = 256

    def __post_init__(self) -> None:
        if min(self.memory_mb, self.pids, self.max_file_mb, self.tmp_mb, self.open_files) < 1:
            raise ValueError("sandbox limits must be positive")
        if self.cpus <= 0 or self.timeout_s <= 0:
            raise ValueError("sandbox limits must be positive")


@dataclass
class SandboxResult:
    exit_code: int
    output: str
    truncated: bool
    timed_out: bool
    oom_killed: bool
    duration_s: float

    def note(self) -> str | None:
        """One line saying WHY a run was stopped, for the model and the log."""
        if self.timed_out:
            return "killed: the time limit was reached"
        if self.oom_killed:
            return "killed: the memory limit was reached"
        return None


async def read_tail(stream: asyncio.StreamReader, limit: int) -> tuple[bytes, int]:
    """Keep only the LAST `limit` bytes (a test summary is at the end). Returns (tail, total)."""
    kept = bytearray()
    total = 0
    while chunk := await stream.read(65536):
        total += len(chunk)
        kept += chunk
        if len(kept) > limit:
            del kept[: len(kept) - limit]
    return bytes(kept), total


def docker_environment() -> dict[str, str]:
    return {name: os.environ[name] for name in _DOCKER_ENV_KEEP if os.environ.get(name)}


def container_user() -> str:
    """Run as the host user on POSIX, so files written to the workspace belong to you."""
    if hasattr(os, "getuid") and sys.platform != "win32":
        return f"{os.getuid()}:{os.getgid()}"
    return "1000:1000"  # Docker Desktop bind mounts are writable for any user


def build_run_args(
    name: str,
    image: str,
    argv: list[str],
    workspace: Path,
    limits: SandboxLimits,
    *,
    user: str | None = None,
    docker: str = "docker",
) -> list[str]:
    """The exact `docker run` command line. Pure, so a test can inspect every flag."""
    return [
        docker, "run",
        "--name", name,
        "--label", LABEL,
        "--pull", "never",
        "--network", "none",
        "--read-only",
        "--tmpfs", f"/tmp:rw,noexec,nosuid,nodev,size={limits.tmp_mb}m",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", str(limits.pids),
        "--memory", f"{limits.memory_mb}m",
        "--memory-swap", f"{limits.memory_mb}m",
        "--cpus", str(limits.cpus),
        "--ulimit", f"nofile={limits.open_files}:{limits.open_files}",
        "--ulimit", f"fsize={limits.max_file_mb * 1024 * 1024}",
        "--user", user or container_user(),
        "--volume", f"{workspace}:{CONTAINER_WORKDIR}:rw",
        "--workdir", CONTAINER_WORKDIR,
        "--env", "HOME=/tmp",
        "--env", f"PYTHONPATH={CONTAINER_WORKDIR}",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", "PYTHONUNBUFFERED=1",
        "--env", "PYTHONIOENCODING=utf-8",
        "--env", "NO_COLOR=1",
        image,
        *argv,
    ]  # fmt: skip


class Sandbox:
    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        limits: SandboxLimits | None = None,
        *,
        docker: str = "docker",
    ) -> None:
        self.image = image
        self.limits = limits or SandboxLimits()
        self._docker = docker

    async def docker_call(self, *args: str, timeout_s: float = 30.0) -> tuple[int, str]:
        try:
            process = await asyncio.create_subprocess_exec(
                self._docker,
                *args,
                env=docker_environment(),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError:
            raise SandboxUnavailable(
                "Docker is not installed or not on PATH. Install Docker Desktop (WSL2 backend "
                "on Windows), then run: uv run aegis sandbox check"
            ) from None
        try:
            raw, _ = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            raise SandboxUnavailable("Docker did not answer in time") from None
        return process.returncode or 0, raw.decode("utf-8", errors="replace").strip()

    async def check(self) -> str:
        """Raise SandboxUnavailable with advice unless Docker works and the image exists."""
        code, out = await self.docker_call("version", "--format", "{{.Server.Version}}")
        if code != 0:
            raise SandboxUnavailable(
                f"Docker is installed but its engine is not running ({_first_line(out)}). "
                "Start Docker Desktop and try again."
            )
        code, _ = await self.docker_call("image", "inspect", self.image)
        if code != 0:
            raise SandboxUnavailable(
                f"the sandbox image {self.image!r} does not exist. Build it once with: "
                "uv run aegis sandbox build"
            )
        return out

    async def run(
        self, argv: list[str], workspace: Path, *, timeout_s: float | None = None
    ) -> SandboxResult:
        """Run `argv` in a fresh container with the workspace mounted. Always cleans up."""
        root = workspace.resolve()
        limit = timeout_s if timeout_s is not None else self.limits.timeout_s
        name = f"aegis-{uuid.uuid4().hex[:12]}"
        args = build_run_args(name, self.image, argv, root, self.limits, docker=self._docker)
        started = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                env=docker_environment(),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError:
            raise SandboxUnavailable("Docker is not installed or not on PATH") from None
        assert process.stdout is not None
        timed_out = False
        try:
            try:
                async with asyncio.timeout(limit):
                    (tail, total), code = await asyncio.gather(
                        read_tail(process.stdout, MAX_OUTPUT_CHARS * 4), process.wait()
                    )
            except TimeoutError:
                timed_out = True
                await self._kill(name)
                (tail, total), code = await asyncio.gather(
                    read_tail(process.stdout, MAX_OUTPUT_CHARS * 4), process.wait()
                )
            oom = False if timed_out else await self._oom_killed(name)
        except asyncio.CancelledError:
            await asyncio.shield(self._kill(name))
            raise
        finally:
            await asyncio.shield(self._remove(name))
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
        text = tail.decode("utf-8", errors="replace")
        if code == 125 and not timed_out:  # docker itself failed before the program started
            raise SandboxUnavailable(_first_line(text) or "docker could not start the container")
        return SandboxResult(
            exit_code=code,
            output=text[-MAX_OUTPUT_CHARS:],
            truncated=total > len(tail) or len(text) > MAX_OUTPUT_CHARS,
            timed_out=timed_out,
            oom_killed=oom,
            duration_s=round(time.monotonic() - started, 2),
        )

    async def _kill(self, name: str) -> None:
        with contextlib.suppress(Exception):
            await self.docker_call("kill", name, timeout_s=15)

    async def _remove(self, name: str) -> None:
        with contextlib.suppress(Exception):
            await self.docker_call("rm", "--force", "--volumes", name, timeout_s=30)

    async def _oom_killed(self, name: str) -> bool:
        try:
            _, out = await self.docker_call(
                "inspect", "--format", "{{.State.OOMKilled}}", name, timeout_s=15
            )
        except SandboxUnavailable:
            return False
        return out.strip() == "true"

    async def cleanup(self) -> int:
        """Remove sandbox containers left behind (for example by a killed process)."""
        _, out = await self.docker_call("ps", "-aq", "--filter", f"label={LABEL}")
        ids = out.split()
        if ids:
            await self.docker_call("rm", "--force", "--volumes", *ids)
        return len(ids)


def _first_line(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1][:300] if lines else ""
