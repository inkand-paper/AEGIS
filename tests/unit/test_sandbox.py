"""Phase 6: the Docker sandbox.

The pure tests always run. The container tests need Docker and the sandbox image
(`uv run aegis sandbox build`, or AEGIS_SANDBOX_IMAGE for another image) and are skipped without.
"""

import asyncio
import os
import socket
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fakes import ScriptedModel, call_tools, plan_json, say

from aegis.controller import Controller, _evidence_kind
from aegis.executor import Executor
from aegis.memory import EventLog
from aegis.models import EventType, EvidenceKind, ToolCall, ToolOutcome
from aegis.sandbox import (
    LABEL,
    Sandbox,
    SandboxLimits,
    SandboxUnavailable,
    build_run_args,
    docker_environment,
)
from aegis.tools import default_registry
from aegis.tools.base import ToolContext, ToolError, ToolUnavailable
from aegis.tools.codeexec import CodeExecArgs, CodeExecTool, snapshot

IMAGE = os.environ.get("AEGIS_SANDBOX_IMAGE", "aegis-sandbox:latest")


def _docker_ready() -> bool:
    try:
        done = subprocess.run(
            ["docker", "image", "inspect", IMAGE], capture_output=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


needs_docker = pytest.mark.skipif(
    not _docker_ready(), reason=f"Docker or the image {IMAGE!r} is not available"
)

# ------------------------------------------------------------------ the command line (pure)


def args_for(limits: SandboxLimits | None = None, workspace: Path = Path("/w")) -> list[str]:
    return build_run_args(
        "n", "img:1", ["python", "x.py"], workspace, limits or SandboxLimits(), user="1:1"
    )


def values_of(args: list[str], flag: str) -> list[str]:
    return [args[i + 1] for i, a in enumerate(args) if a == flag]


def test_the_container_command_has_every_isolation_flag() -> None:
    args = args_for(SandboxLimits(memory_mb=128, pids=32, cpus=0.5))
    assert values_of(args, "--network") == ["none"]
    assert "--read-only" in args
    assert values_of(args, "--cap-drop") == ["ALL"]
    assert values_of(args, "--security-opt") == ["no-new-privileges"]
    assert values_of(args, "--pull") == ["never"]
    assert values_of(args, "--memory") == ["128m"]
    assert values_of(args, "--memory-swap") == ["128m"]  # no swap beyond the cap
    assert values_of(args, "--pids-limit") == ["32"]
    assert values_of(args, "--cpus") == ["0.5"]
    assert values_of(args, "--user") == ["1:1"]
    assert "noexec" in values_of(args, "--tmpfs")[0]
    assert any(v.startswith("fsize=") for v in values_of(args, "--ulimit"))
    assert LABEL in values_of(args, "--label")


def test_only_the_workspace_is_mounted_and_nothing_dangerous_is_passed() -> None:
    args = args_for(workspace=Path("/data/ws1"))
    assert values_of(args, "--volume") == ["/data/ws1:/workspace:rw"]
    for forbidden in ("--privileged", "--pid", "--ipc", "--uts", "--device", "--cap-add"):
        assert forbidden not in args  # whole flags, so --pids-limit does not count
    assert not any("docker.sock" in a for a in args)
    assert values_of(args, "--network") == ["none"]
    assert "-v" not in args and "--mount" not in args
    assert args[-3:] == ["img:1", "python", "x.py"]  # the image comes right before the program


def test_the_docker_cli_does_not_receive_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEGIS_API_KEY", "sk-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    monkeypatch.setenv("PATH", "/usr/bin")
    env = docker_environment()
    assert "AEGIS_API_KEY" not in env and "GITHUB_TOKEN" not in env
    assert env["PATH"] == "/usr/bin"
    assert not any("secret" in v for v in values_of(args_for(), "--env"))


def test_limits_must_be_positive() -> None:
    for bad in ({"memory_mb": 0}, {"pids": 0}, {"cpus": 0}, {"timeout_s": 0}):
        with pytest.raises(ValueError):
            SandboxLimits(**bad)


def test_a_missing_docker_binary_says_what_to_do() -> None:
    box = Sandbox(docker="definitely-not-a-docker-binary")
    with pytest.raises(SandboxUnavailable, match="not installed"):
        asyncio.run(box.check())
    with pytest.raises(SandboxUnavailable, match="not installed"):
        asyncio.run(box.run(["python", "x.py"], Path(".")))


# ------------------------------------------------------------------ code_exec and the dispatcher


def call(tool: str, arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(
        id="c1",
        task_id="t1",
        step_id="s1",
        tool_name=tool,
        arguments=arguments,
        requested_at=datetime.now(UTC),
    )


def ctx_for(tmp_path: Path, *, sandbox: bool) -> ToolContext:
    return ToolContext(
        task_id="t1", workspace=tmp_path, allowed_effects=frozenset({"execute"}), sandbox=sandbox
    )


def test_code_exec_is_refused_for_a_task_without_sandbox(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print(1)")
    executor = Executor(default_registry(Sandbox(IMAGE)), EventLog(tmp_path / "e.db"))
    result = asyncio.run(
        executor.execute(call("code_exec", {"path": "a.py"}), ctx_for(tmp_path, sandbox=False))
    )
    assert result.record.outcome is ToolOutcome.POLICY_VIOLATION
    assert "--sandbox" in (result.record.error or "")


def test_code_exec_never_falls_back_to_running_on_the_host(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("open('ran-on-host', 'w').write('x')")
    tool = CodeExecTool(sandbox=None)
    with pytest.raises(ToolUnavailable):
        asyncio.run(tool.execute(CodeExecArgs(path="a.py"), ctx_for(tmp_path, sandbox=True)))
    assert not (tmp_path / "ran-on-host").exists()


def test_code_exec_arguments_are_validated(tmp_path: Path) -> None:
    for bad in ("-x.py", "a.txt", "a.py\n--privileged", "a.py\n"):
        with pytest.raises(ValueError):
            CodeExecArgs(path=bad)
    registry_tool = default_registry(Sandbox(IMAGE)).get("code_exec")
    executor = Executor(default_registry(Sandbox(IMAGE)), EventLog(tmp_path / "e.db"))
    for escape in ("../a.py", "/etc/passwd.py"):
        result = asyncio.run(
            executor.execute(call("code_exec", {"path": escape}), ctx_for(tmp_path, sandbox=True))
        )
        assert result.record.outcome in {
            ToolOutcome.POLICY_VIOLATION,
            ToolOutcome.INVALID_ARGUMENTS,
        }
    assert registry_tool.requires_sandbox


def test_a_run_that_changes_files_is_file_change_evidence() -> None:
    assert _evidence_kind("code_exec", {"changed": True}) is EvidenceKind.FILE_DIFF
    assert _evidence_kind("code_exec", {"changed": False}) is EvidenceKind.TOOL_RESULT


def test_snapshot_sees_created_changed_and_removed_files(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("a")
    (tmp_path / "edit.txt").write_text("a")
    (tmp_path / "gone.txt").write_text("a")
    before = snapshot(tmp_path)
    (tmp_path / "edit.txt").write_text("b")
    (tmp_path / "gone.txt").unlink()
    (tmp_path / "new.txt").write_text("n")
    after = snapshot(tmp_path)
    changed = sorted(n for n in before.keys() | after.keys() if before.get(n) != after.get(n))
    assert changed == ["edit.txt", "gone.txt", "new.txt"]


# ------------------------------------------------------------------ real containers


def box(**limits: Any) -> Sandbox:
    return Sandbox(IMAGE, SandboxLimits(**limits))


def run_script(tmp_path: Path, code: str, **limits: Any) -> Any:
    (tmp_path / "s.py").write_text(code)
    return asyncio.run(box(**limits).run(["python", "s.py"], tmp_path))


def leftover_containers() -> list[str]:
    out = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"label={LABEL}"], capture_output=True, text=True
    )
    return out.stdout.split()


@needs_docker
def test_known_good_code_runs_and_its_output_comes_back(tmp_path: Path) -> None:
    result = run_script(tmp_path, "print(sum(range(10)))")
    assert (result.exit_code, result.output.strip()) == (0, "45")
    assert not result.timed_out and not result.oom_killed
    assert leftover_containers() == []  # the container was removed


@needs_docker
def test_the_exit_code_and_errors_are_reported(tmp_path: Path) -> None:
    result = run_script(tmp_path, "import sys\nprint('bad', file=sys.stderr)\nsys.exit(3)")
    assert result.exit_code == 3
    assert "bad" in result.output


@needs_docker
def test_a_script_can_write_inside_the_workspace_and_the_tool_reports_it(tmp_path: Path) -> None:
    (tmp_path / "s.py").write_text("open('out.txt', 'w').write('hi')")
    tool = CodeExecTool(box())
    result = asyncio.run(tool.execute(CodeExecArgs(path="s.py"), ctx_for(tmp_path, sandbox=True)))
    assert result.exit_code == 0
    assert result.changed and result.files_changed == ["out.txt"]
    assert (tmp_path / "out.txt").read_text() == "hi"


@needs_docker
def test_an_endless_loop_is_killed_at_the_time_limit(tmp_path: Path) -> None:
    result = run_script(tmp_path, "while True:\n    pass", timeout_s=3)
    assert result.timed_out
    assert result.note() == "killed: the time limit was reached"
    assert result.duration_s < 15
    assert leftover_containers() == []


@needs_docker
def test_memory_hogs_are_killed_at_the_memory_limit(tmp_path: Path) -> None:
    result = run_script(
        tmp_path, "x = bytearray(900 * 1024 * 1024)\nprint('allocated')", memory_mb=64
    )
    assert "allocated" not in result.output
    assert result.exit_code != 0
    assert result.oom_killed or result.exit_code == 137


@needs_docker
def test_a_fork_bomb_is_stopped_by_the_process_limit(tmp_path: Path) -> None:
    code = """
import os, time
children = 0
try:
    while children < 500:
        if os.fork() == 0:
            time.sleep(5)
            os._exit(0)
        children += 1
    print('forked all')
except OSError as exc:
    print('stopped after', children, type(exc).__name__)
"""
    result = run_script(tmp_path, code, pids=16, timeout_s=20)
    assert "forked all" not in result.output
    assert "stopped after" in result.output
    assert leftover_containers() == []


@needs_docker
def test_huge_output_is_cut_not_buffered(tmp_path: Path) -> None:
    result = run_script(tmp_path, "print('x' * 20_000_000)")
    assert result.truncated
    assert len(result.output) <= 6000


@needs_docker
def test_a_single_file_cannot_grow_past_the_file_size_limit(tmp_path: Path) -> None:
    code = "open('big.bin', 'wb').write(b'0' * (8 * 1024 * 1024))\nprint('written')"
    result = run_script(tmp_path, code, max_file_mb=1)
    assert "written" not in result.output
    assert (tmp_path / "big.bin").stat().st_size <= 1024 * 1024


# ---- the workspace boundary (Phase 6 exit criterion)


@needs_docker
def test_code_cannot_read_the_host_or_write_outside_the_workspace(tmp_path: Path) -> None:
    secret_dir = tmp_path.parent / f"{tmp_path.name}-secret"
    secret_dir.mkdir()
    secret = secret_dir / "secret.txt"
    secret.write_text("TOP-SECRET")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "link").symlink_to(secret)  # a symlink that points at a real host file
    code = f"""
import os
def attempt(label, fn):
    try:
        print(label, 'OK', fn())
    except Exception as exc:
        print(label, 'BLOCKED', type(exc).__name__)

attempt('read-host-path', lambda: open({str(secret)!r}).read())
attempt('read-via-symlink', lambda: open('link').read())
attempt('read-via-dotdot', lambda: open('../{secret_dir.name}/secret.txt').read())
attempt('write-etc', lambda: open('/etc/evil', 'w').write('x'))
attempt('write-root', lambda: open('/evil', 'w').write('x'))
attempt('write-opt', lambda: open('/opt/evil', 'w').write('x'))
attempt('docker-socket', lambda: os.stat('/var/run/docker.sock'))
attempt('write-tmp-exec', lambda: os.chmod(open('/tmp/x', 'w').name, 0o755))
attempt('write-workspace', lambda: open('/workspace/ok.txt', 'w').write('fine'))
"""
    (workspace / "s.py").write_text(code)
    result = asyncio.run(box().run(["python", "s.py"], workspace))
    lines = dict(line.split(" ", 1) for line in result.output.splitlines() if " " in line)
    assert "TOP-SECRET" not in result.output
    for label in (
        "read-host-path",
        "read-via-symlink",
        "read-via-dotdot",
        "write-etc",
        "write-root",
        "write-opt",
        "docker-socket",
    ):
        assert lines[label].startswith("BLOCKED"), (label, result.output)
    assert lines["write-workspace"].startswith("OK")  # the one place it may write
    assert (workspace / "ok.txt").read_text() == "fine"
    assert not Path("/etc/evil").exists()


@needs_docker
def test_the_container_has_no_capabilities_and_no_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AEGIS_API_KEY", "sk-should-not-leak")
    code = """
import os
print('ENV', sorted(os.environ))
for line in open('/proc/self/status'):
    if line.startswith(('CapEff', 'NoNewPrivs')):
        print(line.strip())
"""
    result = run_script(tmp_path, code)
    assert "AEGIS_API_KEY" not in result.output and "sk-should-not-leak" not in result.output
    assert "CapEff:\t0000000000000000" in result.output
    assert "NoNewPrivs:\t1" in result.output


# ---- the network (Phase 6 exit criterion)


@needs_docker
def test_a_network_call_fails_inside_the_sandbox(tmp_path: Path) -> None:
    hits: list[str] = []
    server = socket.socket()
    server.bind(("0.0.0.0", 0))
    server.listen(5)
    port = server.getsockname()[1]

    def accept() -> None:
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            hits.append("connected")
            conn.close()

    threading.Thread(target=accept, daemon=True).start()
    host_ips = {"127.0.0.1", "172.17.0.1", socket.gethostbyname(socket.gethostname())}
    code = f"""
import socket
for ip in {sorted(host_ips)!r}:
    try:
        socket.create_connection((ip, {port}), timeout=3).close()
        print('CONNECTED', ip)
    except OSError as exc:
        print('FAILED', ip, type(exc).__name__)
try:
    socket.getaddrinfo('example.com', 80)
    print('DNS RESOLVED')
except OSError:
    print('DNS FAILED')
"""
    result = run_script(tmp_path, code)
    server.close()
    assert "CONNECTED" not in result.output and "DNS RESOLVED" not in result.output
    assert result.output.count("FAILED") >= len(host_ips) + 1
    assert hits == []  # the host listener never saw a connection


# ---- cleanup


@needs_docker
def test_cancelling_a_run_removes_the_container(tmp_path: Path) -> None:
    (tmp_path / "s.py").write_text("import time\ntime.sleep(60)")

    async def go() -> None:
        task = asyncio.create_task(box(timeout_s=60).run(["python", "s.py"], tmp_path))
        await asyncio.sleep(3)
        assert len(leftover_containers()) == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    assert leftover_containers() == []


@needs_docker
def test_a_missing_image_is_a_clear_error_and_nothing_is_pulled(tmp_path: Path) -> None:
    ghost = Sandbox("aegis-image-that-does-not-exist:0")
    with pytest.raises(SandboxUnavailable, match="aegis sandbox build"):
        asyncio.run(ghost.check())
    (tmp_path / "s.py").write_text("print(1)")
    with pytest.raises(SandboxUnavailable):
        asyncio.run(ghost.run(["python", "s.py"], tmp_path))


@needs_docker
def test_cleanup_removes_leftover_containers(tmp_path: Path) -> None:
    subprocess.run(
        ["docker", "create", "--label", LABEL, "--name", "aegis-leftover-test", IMAGE, "true"],
        capture_output=True,
        check=True,
    )
    removed = asyncio.run(box().cleanup())
    assert removed >= 1
    assert leftover_containers() == []


# ------------------------------------------------------------------ the whole agent


@needs_docker
def test_the_agent_writes_runs_and_verifies_code_in_the_sandbox(tmp_path: Path) -> None:
    events = EventLog(tmp_path / "aegis.db")
    script = "def add(a, b):\n    return a + b\n\nprint('sum', add(2, 3))\n"
    model = ScriptedModel(
        say(plan_json("Write the script", "Run it in the sandbox")),
        call_tools(("fs_write", {"path": "add.py", "content": script})),
        say("Written."),
        call_tools(("code_exec", {"path": "add.py"})),
        say("It printed sum 5."),
    )
    controller = Controller(
        model,
        default_registry(Sandbox(IMAGE)),
        events,
        workspaces_dir=tmp_path / "workspaces",
    )
    task = controller.create_task(
        "Add 2 and 3 with a script",
        acceptance=["file_written:add.py", "result_contains:sum 5"],
        allowed_effects=["workspace", "execute"],
        sandbox=True,
    )
    finished = asyncio.run(controller.run(task.id))
    assert finished.state.value == "completed", finished.last_error
    assert finished.sandbox is True
    outcomes = [e.payload["outcome"] for e in events.events(task.id, EventType.TOOL_RESULT)]
    assert outcomes == ["ok", "ok"]


@needs_docker
def test_tests_run_in_the_sandbox_when_the_task_asks_for_it(tmp_path: Path) -> None:
    events = EventLog(tmp_path / "aegis.db")
    code = "def add(a, b):\n    return a + b\n"
    test = "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    model = ScriptedModel(
        say(plan_json("Write code and tests", "Run the tests")),
        call_tools(
            ("fs_write", {"path": "calc.py", "content": code}),
            ("fs_write", {"path": "test_calc.py", "content": test}),
        ),
        say("Written."),
        call_tools(("shell_run_tests", {})),
        say("The tests pass."),
    )
    controller = Controller(
        model, default_registry(Sandbox(IMAGE)), events, workspaces_dir=tmp_path / "workspaces"
    )
    task = controller.create_task(
        "Add",
        acceptance=["tests_pass"],
        allowed_effects=["workspace", "execute"],
        sandbox=True,
    )
    finished = asyncio.run(controller.run(task.id))
    assert finished.state.value == "completed", finished.last_error
    result = events.events(task.id, EventType.TOOL_RESULT)[-1].payload["result"]
    assert result["passed"] == 1 and result["exit_code"] == 0


def test_tool_error_type_is_exported_for_callers() -> None:
    assert issubclass(ToolUnavailable, ToolError)
