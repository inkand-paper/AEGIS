"""image_generate (design doc section 6.3, Phase 8).

The model asks for a picture; this tool asks the configured provider, then looks at the bytes
before anything is saved: they must be a well-formed image, and the format must match the file
name. The file is written atomically into the workspace and its SHA-256 goes into the result,
which the controller records as ARTIFACT_HASH evidence.

Network: a provider that contacts a server needs that server's host in `--allow-host`, and the
prompt is sent to it. The offline provider needs nothing.
"""

import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from aegis.images import (
    ImageError,
    ImageFormat,
    ImageProvider,
    inspect_image,
    require_extension,
)
from aegis.tools.base import PolicyError, SideEffects, Tool, ToolContext, ToolError
from aegis.tools.browser import check_url
from aegis.tools.filewrite import _write_atomically

logger = logging.getLogger(__name__)


class ImageGenerateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=1000, description="What the picture should show.")
    path: str = Field(
        description="Where to save it in the workspace, ending in .png, .jpg, .gif or .webp."
    )
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    seed: int | None = Field(default=None, description="Same prompt and seed repeat a result.")
    overwrite: bool = Field(default=False, description="Allow replacing an existing file.")


class ImageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    format: ImageFormat
    width: int
    height: int
    size_bytes: int
    sha256: str
    provider: str
    model: str | None
    depicts_prompt: bool  # False: the picture does not show the prompt (offline test provider)
    prompt: str


class ImageGenerateTool(Tool[ImageGenerateArgs, ImageResult]):
    name = "image_generate"
    arg_model = ImageGenerateArgs
    result_model = ImageResult
    idempotent = False
    path_fields = ("path",)
    timeout_s = 150.0

    def __init__(self, provider: ImageProvider) -> None:
        self._provider = provider
        remote = provider.endpoint is not None
        # Instance values: they depend on the provider chosen in settings.
        self.requires_network = remote
        self.side_effects: SideEffects = "external" if remote else "workspace"
        self.description = (
            "Generate one image from a text prompt and save it in the task workspace. "
            "The file name decides the format (use .png unless told otherwise). "
        ) + (
            "The pictures it makes show what the prompt describes."
            if provider.depicts_prompt
            else (
                "This is the OFFLINE TEST provider: it makes a deterministic abstract picture "
                "that does NOT depict the prompt. Do not describe it as showing the prompt."
            )
        )

    async def execute(self, args: ImageGenerateArgs, ctx: ToolContext) -> ImageResult:
        try:
            wanted = require_extension(args.path)
        except ValueError as exc:
            raise ToolError(str(exc)) from None
        target = ctx.resolve(args.path)
        if target == ctx.workspace.resolve() or target.is_dir():
            raise ToolError(f"{args.path} is a directory")
        if target.exists() and not args.overwrite:
            raise ToolError(f"{args.path} already exists: choose another name or set overwrite")
        if self._provider.endpoint is not None:
            try:
                check_url(self._provider.endpoint, ctx.allowed_hosts)
            except PolicyError as exc:
                raise PolicyError(f"the image provider is not reachable: {exc}") from None
        try:
            made = await self._provider.generate(
                args.prompt, width=args.width, height=args.height, seed=args.seed
            )
            info = inspect_image(made.data)
        except ImageError as exc:
            raise ToolError(str(exc)) from None
        if info.format != wanted:
            raise ToolError(
                f"the provider returned a {info.format} image but {args.path} ends in a "
                f"{wanted} extension: use a file name ending in .{info.format}"
            )
        _write_atomically(target, made.data)
        shown = Path(target).relative_to(ctx.workspace.resolve()).as_posix()
        logger.info("image saved: %s %s %dx%d", shown, info.format, info.width, info.height)
        return ImageResult(
            path=shown,
            format=info.format,
            width=info.width,
            height=info.height,
            size_bytes=info.size_bytes,
            sha256=info.sha256,
            provider=made.provider,
            model=made.model,
            depicts_prompt=self._provider.depicts_prompt,
            prompt=args.prompt,
        )
