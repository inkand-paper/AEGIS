"""Phase 8: image providers, the image check, the image_generate tool and the image_saved criterion.

Nothing here reaches the network: the HTTP provider is tested against httpx's mock transport.
"""

import asyncio
import base64
import hashlib
import struct
import zlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from fakes import ScriptedModel, call_tools, plan_json, say

from aegis.controller import Controller
from aegis.executor import Executor
from aegis.images import (
    GeneratedImage,
    ImageError,
    OpenAIImagesProvider,
    ProceduralProvider,
    build_image_provider,
    inspect_image,
    require_extension,
)
from aegis.memory import EventLog
from aegis.models import (
    EventType,
    Evidence,
    EvidenceKind,
    Plan,
    Step,
    StepStatus,
    TaskState,
    ToolCall,
    ToolOutcome,
)
from aegis.tools import default_registry
from aegis.tools.base import PolicyError, ToolContext, ToolError
from aegis.tools.image import ImageGenerateTool
from aegis.verifier import Verifier, parse_criterion

# ------------------------------------------------------------------ small image fixtures


def make_png(width: int = 8, height: int = 8) -> bytes:
    return asyncio.run(
        ProceduralProvider().generate("fixture", width=width, height=height, seed=1)
    ).data


def png_with_size(width: int, height: int) -> bytes:
    """A structurally valid PNG that DECLARES a size, with a tiny body (a decompression bomb)."""

    def chunk(tag: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(tag + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00"))
        + chunk(b"IEND", b"")
    )


def make_jpeg(width: int = 4, height: int = 3) -> bytes:
    frame = struct.pack(">BHHB", 8, height, width, 1) + b"\x01\x11\x00"
    return (
        b"\xff\xd8"
        + b"\xff\xc0"
        + struct.pack(">H", len(frame) + 2)
        + frame
        + b"\xff\xda\x00\x02"
        + b"\x00"
        + b"\xff\xd9"
    )


def make_gif(width: int = 5, height: int = 2) -> bytes:
    return b"GIF89a" + struct.pack("<HH", width, height) + b"\x00\x00\x00" + b"\x3b"


def make_webp_lossless(width: int = 6, height: int = 4) -> bytes:
    bits = (width - 1) | ((height - 1) << 14)
    chunk_body = b"\x2f" + struct.pack("<I", bits) + b"\x00" * 4
    payload = b"WEBP" + b"VP8L" + struct.pack("<I", len(chunk_body)) + chunk_body
    return b"RIFF" + struct.pack("<I", len(payload)) + payload


# ------------------------------------------------------------------ inspect_image


def test_a_real_png_is_described_with_its_hash() -> None:
    data = make_png(16, 8)
    info = inspect_image(data)
    assert (info.format, info.width, info.height) == ("png", 16, 8)
    assert info.size_bytes == len(data)
    assert info.sha256 == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (make_jpeg(4, 3), ("jpeg", 4, 3)),
        (make_gif(5, 2), ("gif", 5, 2)),
        (make_webp_lossless(6, 4), ("webp", 6, 4)),
    ],
)
def test_other_formats_are_recognised_by_their_bytes(
    data: bytes, expected: tuple[str, int, int]
) -> None:
    info = inspect_image(data)
    assert (info.format, info.width, info.height) == expected


def test_a_flipped_byte_inside_a_png_is_caught_by_its_crc() -> None:
    data = bytearray(make_png())
    data[len(data) // 2] ^= 0xFF
    with pytest.raises(ImageError, match="corrupt"):
        inspect_image(bytes(data))


def test_a_cut_off_png_is_refused() -> None:
    with pytest.raises(ImageError, match="cut off"):
        inspect_image(make_png()[:-20])


def test_bytes_hidden_after_the_end_of_a_png_are_refused() -> None:
    with pytest.raises(ImageError, match="extra bytes"):
        inspect_image(make_png() + b"<script>")


@pytest.mark.parametrize(
    "data", [b"", b"hello", b"<html></html>", b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff"]
)
def test_things_that_are_not_images_are_refused(data: bytes) -> None:
    with pytest.raises(ImageError):
        inspect_image(data)


def test_a_cut_off_jpeg_gif_and_webp_are_refused() -> None:
    with pytest.raises(ImageError, match="cut off"):
        inspect_image(make_jpeg()[:-2])
    with pytest.raises(ImageError, match="cut off"):
        inspect_image(make_gif()[:-1])
    with pytest.raises(ImageError, match="does not match"):
        inspect_image(make_webp_lossless() + b"\x00\x00")


def test_a_tiny_file_that_declares_a_huge_picture_is_refused() -> None:
    with pytest.raises(ImageError, match="each side must be"):
        inspect_image(png_with_size(100_000, 10))
    with pytest.raises(ImageError, match="more than"):
        inspect_image(png_with_size(4096, 4096))


def test_a_file_over_the_size_limit_is_refused() -> None:
    with pytest.raises(ImageError, match="larger than"):
        inspect_image(b"\x89PNG\r\n\x1a\n" + b"\x00" * (10 * 1024 * 1024))


def test_the_extension_decides_the_expected_format() -> None:
    assert require_extension("a/b.PNG") == "png"
    assert require_extension("x.jpeg") == "jpeg"
    for bad in ["noext", "pic.exe", "pic.png.txt", "pic."]:
        with pytest.raises(ValueError, match="must end in"):
            require_extension(bad)


# ------------------------------------------------------------------ the offline provider


def test_the_offline_provider_is_deterministic_and_prompt_dependent() -> None:
    provider = ProceduralProvider()

    def gen(prompt: str, seed: int | None = None, size: int = 32) -> bytes:
        return asyncio.run(provider.generate(prompt, width=size, height=size, seed=seed)).data

    assert gen("a cat") == gen("a cat")
    assert gen("a cat") != gen("a dog")
    assert gen("a cat", seed=1) != gen("a cat", seed=2)
    assert gen("a cat", size=32) != gen("a cat", size=48)
    assert inspect_image(gen("a cat")).width == 32


def test_the_offline_provider_says_it_does_not_depict_the_prompt() -> None:
    provider = ProceduralProvider()
    assert provider.depicts_prompt is False and provider.endpoint is None
    tool = ImageGenerateTool(provider)
    assert "does NOT depict the prompt" in tool.description


def test_the_offline_provider_refuses_huge_sizes() -> None:
    with pytest.raises(ImageError, match="at most"):
        asyncio.run(ProceduralProvider().generate("x", width=2000, height=64, seed=None))


# ------------------------------------------------------------------ the HTTP provider


def patch_http(
    monkeypatch: pytest.MonkeyPatch, handler: Any, seen: list[httpx.Request] | None = None
) -> None:
    real = httpx.AsyncClient

    def record(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return handler(request)  # type: ignore[no-any-return]

    def factory(**kwargs: Any) -> httpx.AsyncClient:
        return real(transport=httpx.MockTransport(record), **kwargs)

    monkeypatch.setattr("aegis.images.httpx.AsyncClient", factory)


def ok_reply(data: bytes) -> httpx.Response:
    return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(data).decode()}]})


def remote(model: str = "gpt-image-1", key: str | None = "sk-secret-123") -> OpenAIImagesProvider:
    return OpenAIImagesProvider(base_url="https://img.example.com/v1/", api_key=key, model=model)


def test_the_http_provider_sends_the_openai_request_and_decodes_the_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[httpx.Request] = []
    png = make_png()
    patch_http(monkeypatch, lambda r: ok_reply(png), seen)
    made = asyncio.run(remote().generate("a red fox", width=512, height=256, seed=7))
    assert made.data == png and made.provider == "openai-images"
    request = seen[0]
    assert str(request.url) == "https://img.example.com/v1/images/generations"
    assert request.headers["authorization"] == "Bearer sk-secret-123"
    body = __import__("json").loads(request.content)
    assert body["prompt"] == "a red fox" and body["size"] == "512x256" and body["seed"] == 7
    assert "response_format" not in body  # gpt-image models reject it


def test_dall_e_models_ask_for_base64_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[httpx.Request] = []
    reply = ok_reply(make_png())
    patch_http(monkeypatch, lambda r: reply, seen)
    asyncio.run(remote("dall-e-3").generate("x", width=64, height=64, seed=None))
    assert __import__("json").loads(seen[0].content)["response_format"] == "b64_json"


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (httpx.Response(500, text="boom"), "HTTP 500"),
        (httpx.Response(200, json={"data": [{"url": "https://x/y.png"}]}), "no inline image"),
        (httpx.Response(200, json={"oops": 1}), "unexpected shape"),
        (httpx.Response(200, text="not json"), "unexpected shape"),
        (httpx.Response(200, json={"data": [{"b64_json": "!!!not base64"}]}), "invalid base64"),
    ],
)
def test_bad_provider_replies_become_clear_errors(
    monkeypatch: pytest.MonkeyPatch, reply: httpx.Response, message: str
) -> None:
    patch_http(monkeypatch, lambda r: reply)
    with pytest.raises(ImageError, match=message):
        asyncio.run(remote().generate("x", width=64, height=64, seed=None))


def test_the_api_key_never_appears_in_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_http(monkeypatch, lambda r: httpx.Response(401, text="bad key sk-secret-123 given"))
    with pytest.raises(ImageError) as info:
        asyncio.run(remote().generate("x", width=64, height=64, seed=None))
    assert "sk-secret-123" not in str(info.value) and "***" in str(info.value)


def test_a_network_failure_is_one_clear_error_and_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ConnectError("down")

    patch_http(monkeypatch, handler)
    with pytest.raises(ImageError, match="could not reach"):
        asyncio.run(remote().generate("x", width=64, height=64, seed=None))
    assert len(attempts) == 1  # a generation can cost money: never repeated silently


def test_settings_choose_the_provider() -> None:
    assert isinstance(
        build_image_provider("", api_key=None, base_url=None, model=None), ProceduralProvider
    )
    openai = build_image_provider("openai", api_key="k", base_url=None, model=None)
    assert openai.endpoint == "https://api.openai.com/v1/images/generations"
    custom = build_image_provider("custom", api_key=None, base_url="http://h:1/v1", model="m")
    assert custom.endpoint == "http://h:1/v1/images/generations"
    with pytest.raises(ValueError, match="BASE_URL"):
        build_image_provider("custom", api_key=None, base_url=None, model="m")
    with pytest.raises(ValueError, match="MODEL"):
        build_image_provider("custom", api_key=None, base_url="http://h", model=None)
    with pytest.raises(ValueError, match="unknown image provider"):
        build_image_provider("midjourney", api_key=None, base_url=None, model=None)


# ------------------------------------------------------------------ the tool


class FixedProvider:
    """Returns whatever bytes it was given, as if a provider had sent them."""

    name = "fixed"
    model: str | None = None
    depicts_prompt = True

    def __init__(self, data: bytes, endpoint: str | None = None) -> None:
        self.data = data
        self.endpoint = endpoint
        self.calls = 0

    async def generate(
        self, prompt: str, *, width: int, height: int, seed: int | None
    ) -> GeneratedImage:
        self.calls += 1
        return GeneratedImage(self.data, self.name)


def ctx(
    tmp_path: Path, hosts: set[str] | None = None, effects: set[str] | None = None
) -> ToolContext:
    return ToolContext(
        task_id="t1",
        workspace=tmp_path,
        allowed_effects=frozenset(effects if effects is not None else {"workspace"}),
        allowed_hosts=frozenset(hosts or set()),
    )


def run_image(tool: ImageGenerateTool, args: dict[str, Any], c: ToolContext) -> Any:
    return asyncio.run(tool.execute(tool.arg_model.model_validate(args), c))


def call(tool: str, arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(
        id="c1",
        task_id="t1",
        step_id="s1",
        tool_name=tool,
        arguments=arguments,
        requested_at=datetime.now(UTC),
    )


def test_the_tool_saves_a_checked_image_with_its_hash(tmp_path: Path) -> None:
    tool = ImageGenerateTool(ProceduralProvider())
    result = run_image(
        tool, {"prompt": "a lighthouse", "path": "art/cover.png", "width": 96, "height": 64},
        ctx(tmp_path),
    )  # fmt: skip
    saved = (tmp_path / "art" / "cover.png").read_bytes()
    assert result.path == "art/cover.png" and result.format == "png"
    assert (result.width, result.height) == (96, 64)
    assert result.sha256 == hashlib.sha256(saved).hexdigest()
    assert result.size_bytes == len(saved) and result.depicts_prompt is False
    assert not list(tmp_path.rglob(".aegis-*"))  # no temp file left behind


def test_the_same_request_gives_the_same_hash(tmp_path: Path) -> None:
    tool = ImageGenerateTool(ProceduralProvider())
    args = {"prompt": "dunes", "seed": 3, "width": 64, "height": 64}
    first = run_image(tool, {**args, "path": "a.png"}, ctx(tmp_path))
    second = run_image(tool, {**args, "path": "b.png"}, ctx(tmp_path))
    assert first.sha256 == second.sha256


def test_an_existing_file_is_kept_unless_overwrite_is_set(tmp_path: Path) -> None:
    tool = ImageGenerateTool(ProceduralProvider())
    run_image(tool, {"prompt": "one", "path": "a.png", "width": 64, "height": 64}, ctx(tmp_path))
    before = (tmp_path / "a.png").read_bytes()
    with pytest.raises(ToolError, match="already exists"):
        run_image(
            tool, {"prompt": "two", "path": "a.png", "width": 64, "height": 64}, ctx(tmp_path)
        )
    assert (tmp_path / "a.png").read_bytes() == before
    run_image(
        tool,
        {"prompt": "two", "path": "a.png", "width": 64, "height": 64, "overwrite": True},
        ctx(tmp_path),
    )
    assert (tmp_path / "a.png").read_bytes() != before


def test_the_file_name_must_match_what_the_provider_returned(tmp_path: Path) -> None:
    tool = ImageGenerateTool(FixedProvider(make_png()))
    with pytest.raises(ToolError, match="returned a png image"):
        run_image(tool, {"prompt": "x", "path": "photo.jpg"}, ctx(tmp_path))
    with pytest.raises(ToolError, match="must end in"):
        run_image(tool, {"prompt": "x", "path": "photo.txt"}, ctx(tmp_path))
    assert not (tmp_path / "photo.jpg").exists()


@pytest.mark.parametrize(
    "bad", [b"<html>not an image</html>", make_png()[:-9], make_png() + b"junk", b""]
)
def test_bytes_that_are_not_a_valid_image_are_never_saved(tmp_path: Path, bad: bytes) -> None:
    tool = ImageGenerateTool(FixedProvider(bad))
    with pytest.raises(ToolError):
        run_image(tool, {"prompt": "x", "path": "out.png"}, ctx(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_a_directory_cannot_be_the_target(tmp_path: Path) -> None:
    (tmp_path / "pics.png").mkdir()
    tool = ImageGenerateTool(ProceduralProvider())
    with pytest.raises(ToolError, match="already exists|directory"):
        run_image(tool, {"prompt": "x", "path": "pics.png", "overwrite": True}, ctx(tmp_path))


def test_paths_outside_the_workspace_are_refused_by_the_dispatcher(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    executor = Executor(default_registry(), EventLog(tmp_path / "e.db"))
    result = asyncio.run(
        executor.execute(
            call("image_generate", {"prompt": "x", "path": "../escape.png"}), ctx(workspace)
        )
    )
    assert result.record.outcome is ToolOutcome.POLICY_VIOLATION
    assert not (tmp_path / "escape.png").exists()


def test_the_offline_tool_needs_workspace_permission_and_no_network(tmp_path: Path) -> None:
    executor = Executor(default_registry(), EventLog(tmp_path / "e.db"))
    args = {"prompt": "x", "path": "a.png", "width": 64, "height": 64}
    asked = asyncio.run(
        executor.execute(call("image_generate", args), ctx(tmp_path, effects=set()))
    )
    assert asked.record.outcome is ToolOutcome.APPROVAL_REQUIRED
    assert not (tmp_path / "a.png").exists()
    done = asyncio.run(executor.execute(call("image_generate", args), ctx(tmp_path)))
    assert done.record.outcome is ToolOutcome.OK
    assert done.record.result is not None and done.record.result["sha256"]


def test_a_provider_that_uses_the_network_is_refused_without_an_allowed_host(
    tmp_path: Path,
) -> None:
    provider = FixedProvider(make_png(), endpoint="https://img.example.com/v1/images/generations")
    registry = default_registry(images=provider)
    executor = Executor(registry, EventLog(tmp_path / "e.db"))
    result = asyncio.run(
        executor.execute(
            call("image_generate", {"prompt": "x", "path": "a.png"}),
            ctx(tmp_path, effects={"workspace", "external"}),
        )
    )
    assert result.record.outcome is ToolOutcome.POLICY_VIOLATION
    assert "no host was allowed" in (result.record.error or "")
    assert provider.calls == 0
    names = [t["name"] for t in registry.list_for_model(network=False)]
    assert "image_generate" not in names  # not even offered to the model
    assert "image_generate" in [t["name"] for t in registry.list_for_model(network=True)]


def test_a_provider_host_that_was_not_allowed_is_refused_even_if_others_are(
    tmp_path: Path,
) -> None:
    provider = FixedProvider(make_png(), endpoint="https://img.example.com/v1/images/generations")
    tool = ImageGenerateTool(provider)
    with pytest.raises(PolicyError, match="image provider is not reachable"):
        run_image(tool, {"prompt": "x", "path": "a.png"}, ctx(tmp_path, hosts={"other.com"}))
    assert provider.calls == 0 and not (tmp_path / "a.png").exists()
    result = run_image(
        tool, {"prompt": "x", "path": "a.png"}, ctx(tmp_path, hosts={"img.example.com"})
    )
    assert result.provider == "fixed" and provider.calls == 1


def test_declaring_the_host_is_what_pre_approves_the_remote_provider(tmp_path: Path) -> None:
    provider = FixedProvider(make_png(), endpoint="https://img.example.com/v1/images/generations")
    assert ImageGenerateTool(provider).side_effects == "external"
    assert ImageGenerateTool(ProceduralProvider()).side_effects == "workspace"


# ------------------------------------------------------------------ the image_saved criterion


def hash_evidence(path: str, data: bytes, index: int = 0) -> Evidence:
    info = inspect_image(data)
    return Evidence(
        id=f"img{index}",
        task_id="t",
        step_id="s1",
        kind=EvidenceKind.ARTIFACT_HASH,
        payload={
            "tool_name": "image_generate",
            "arguments": {"path": path},
            "result": {"path": path, "sha256": info.sha256},
            "outcome": "ok",
        },
        created_at=datetime.now(UTC),
        source_tool_call_id=f"c{index}",
    )


def verify_images(criteria: list[str], evidence: list[Evidence], workspace: Path | None) -> Any:
    plan = Plan(
        id="p",
        task_id="t",
        steps=[Step(id="s1", plan_id="p", index=1, description="d", status=StepStatus.DONE)],
    )
    return Verifier().verify(
        task_id="t",
        plan=plan,
        evidence=evidence,
        criteria=[parse_criterion(c) for c in criteria],
        final_answer="done",
        unmatched_tool_calls=0,
        workspace=workspace,
    )


def check(result: Any) -> dict[str, Any]:
    return next(c for c in result.checks if c["name"].startswith("criterion:image_saved"))


def test_image_saved_passes_when_the_file_on_disk_matches_the_record(tmp_path: Path) -> None:
    data = make_png()
    (tmp_path / "cover.png").write_bytes(data)
    result = verify_images(["image_saved:cover.png"], [hash_evidence("cover.png", data)], tmp_path)
    assert result.passed, result.checks
    assert "png 8x8" in check(result)["message"]


def test_image_saved_fails_if_the_file_was_replaced_afterwards(tmp_path: Path) -> None:
    data = make_png()
    (tmp_path / "cover.png").write_bytes(make_png(16, 16))  # a valid image, but not the one saved
    result = verify_images(["image_saved:cover.png"], [hash_evidence("cover.png", data)], tmp_path)
    assert not result.passed
    assert "changed after" in check(result)["message"]


def test_image_saved_fails_if_the_file_is_gone_or_damaged(tmp_path: Path) -> None:
    data = make_png()
    evidence = [hash_evidence("cover.png", data)]
    gone = verify_images(["image_saved:cover.png"], evidence, tmp_path)
    assert not gone.passed and "not in the workspace" in check(gone)["message"]
    (tmp_path / "cover.png").write_bytes(data[:-10])
    damaged = verify_images(["image_saved:cover.png"], evidence, tmp_path)
    assert not damaged.passed and "not a valid image" in check(damaged)["message"]


def test_image_saved_fails_when_nothing_was_generated_or_no_workspace_is_given(
    tmp_path: Path,
) -> None:
    data = make_png()
    (tmp_path / "cover.png").write_bytes(data)  # a file exists, but image_generate never made it
    never = verify_images(["image_saved:cover.png"], [], tmp_path)
    assert not never.passed and "never saved" in check(never)["message"]
    blind = verify_images(["image_saved:cover.png"], [hash_evidence("cover.png", data)], None)
    assert not blind.passed


def test_image_saved_uses_the_last_save_of_a_path(tmp_path: Path) -> None:
    first, second = make_png(8, 8), make_png(16, 16)
    (tmp_path / "cover.png").write_bytes(second)
    evidence = [hash_evidence("cover.png", first, 0), hash_evidence("cover.png", second, 1)]
    assert verify_images(["image_saved:cover.png"], evidence, tmp_path).passed
    (tmp_path / "cover.png").write_bytes(first)  # the OLD image is back on disk: not the last one
    assert not verify_images(["image_saved:cover.png"], evidence, tmp_path).passed


def test_image_saved_cannot_be_pointed_outside_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    data = make_png()
    (tmp_path / "outside.png").write_bytes(data)
    result = verify_images(
        ["image_saved:../outside.png"], [hash_evidence("../outside.png", data)], workspace
    )
    assert not result.passed and "not in the workspace" in check(result)["message"]


def test_image_saved_needs_a_path() -> None:
    with pytest.raises(ValueError, match="needs a value"):
        parse_criterion("image_saved")


# ------------------------------------------------------------------ whole tasks


def agent(tmp_path: Path, model: ScriptedModel, **kwargs: Any) -> tuple[Controller, EventLog]:
    events = EventLog(tmp_path / "aegis.db")
    controller = Controller(
        model, default_registry(**kwargs), events, workspaces_dir=tmp_path / "workspaces"
    )
    return controller, events


def test_a_task_that_makes_an_image_is_verified_against_the_file(tmp_path: Path) -> None:
    model = ScriptedModel(
        say(plan_json("Make the cover image")),
        call_tools(("image_generate", {"prompt": "a lighthouse", "path": "cover.png"})),
        say("Saved cover.png (an abstract test picture, not a lighthouse)."),
    )
    controller, events = agent(tmp_path, model)
    task = controller.create_task(
        "Make a cover image",
        acceptance=["image_saved:cover.png", "tool_used:image_generate"],
        allowed_effects=["workspace"],
    )
    finished = asyncio.run(controller.run(task.id))
    assert finished.state is TaskState.COMPLETED, finished.last_error
    kinds = [e.kind for e in _evidence(events, task.id)]
    assert EvidenceKind.ARTIFACT_HASH in kinds


def _evidence(events: EventLog, task_id: str) -> list[Evidence]:
    return [
        Evidence.model_validate(e.payload) for e in events.events(task_id, EventType.EVIDENCE_ADDED)
    ]


def test_fs_write_cannot_overwrite_a_generated_image_with_text(tmp_path: Path) -> None:
    """A text write onto the picture is refused by fs_write, so the recorded hash still holds."""
    replace = (
        "fs_write",
        {"path": "cover.png", "content": "not the generated image", "overwrite": True},
    )
    model = ScriptedModel(
        say(plan_json("Make the image and tidy up")),
        call_tools(("image_generate", {"prompt": "x", "path": "cover.png"})),
        call_tools(replace),
        say("Done."),
    )
    controller, events = agent(tmp_path, model)
    task = controller.create_task(
        "Make a cover image",
        acceptance=["image_saved:cover.png"],
        allowed_effects=["workspace"],
    )
    finished = asyncio.run(controller.run(task.id))
    results = [e.payload for e in events.events(task.id, EventType.TOOL_RESULT)]
    assert [r["outcome"] for r in results] == ["ok", "tool_error"]
    assert "not a UTF-8 text file" in results[1]["error"]
    assert inspect_image((Path(finished.workspace_path) / "cover.png").read_bytes()).format == "png"
    assert finished.state is TaskState.COMPLETED


def test_research_code_and_image_end_to_end_with_a_verified_report(tmp_path: Path) -> None:
    """Phase 8 exit criterion: research + code + image -> a verified report.

    Research is reading a notes file, code is a small tested module, the image is generated and
    hashed, and the report references it. Every part is checked by the verifier, not claimed. The
    tests run last on purpose: a test run only counts if nothing changed after it.
    """
    notes = tmp_path / "seed"
    notes.mkdir()
    module = "def total(prices):\n    return sum(prices)\n"
    tests = "from shop import total\n\n\ndef test_total():\n    assert total([2, 3]) == 5\n"
    model = ScriptedModel(
        say(
            plan_json(
                "Read the notes",
                "Make the chart image",
                "Write the report",
                "Write and test the code",
            )
        ),
        call_tools(
            ("fs_write", {"path": "notes.txt", "content": "Item 1 costs 2; Item 2 costs 3."})
        ),
        call_tools(("fs_read", {"path": "notes.txt"})),
        say("Two items, 2 and 3."),
        call_tools(
            (
                "image_generate",
                {
                    "prompt": "bar chart of two prices",
                    "path": "chart.png",
                    "seed": 5,
                    "width": 128,
                    "height": 96,
                },
            )
        ),
        say("Image saved."),
        call_tools(
            (
                "fs_write",
                {"path": "report.md", "content": "# Report\n\nTotal: 5\n\n![chart](chart.png)\n"},
            )
        ),
        say("Report written: total 5."),
        call_tools(
            ("fs_write", {"path": "shop.py", "content": module}),
            ("fs_write", {"path": "test_shop.py", "content": tests}),
        ),
        call_tools(("shell_run_tests", {})),
        say("Tests pass; report says total 5."),
    )
    controller, events = agent(tmp_path, model)
    task = controller.create_task(
        "Research prices, compute the total, chart it and report",
        acceptance=[
            "file_read:notes.txt",
            "tests_pass",
            "image_saved:chart.png",
            "file_written:report.md",
            "answer_contains:total 5",
        ],
        allowed_effects=["workspace", "execute"],
    )
    finished = asyncio.run(controller.run(task.id))
    assert finished.state is TaskState.COMPLETED, finished.last_error
    workspace = tmp_path / "workspaces" / task.id
    info = inspect_image((workspace / "chart.png").read_bytes())
    assert (info.width, info.height) == (128, 96)
    assert "chart.png" in (workspace / "report.md").read_text()
