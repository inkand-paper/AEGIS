"""Image generation (design doc section 6.3, Phase 8): the provider interface and the image check.

Two separate jobs live here:

1. `ImageProvider`: something that turns a prompt into image bytes. The agent code never knows
   which one it is talking to, so changing provider is a setting, not a code change.
     * `ProceduralProvider`  offline and free. It draws a deterministic abstract picture from the
                              prompt. It does NOT depict the prompt: it exists so the whole
                              pipeline (tool, workspace, hashing, verifier) can be run and tested
                              without a network or an account. The tool tells the model so.
     * `OpenAIImagesProvider` any server that speaks the OpenAI `/images/generations` protocol
                              (OpenAI itself, and many compatible servers). This one reaches the
                              network, so the task must allow its host (`--allow-host`).

2. `inspect_image`: looks at the BYTES and says what they really are. A provider's claim ("this is
   a PNG") is never trusted. The check is structural, written with the standard library only:
   PNG is checked chunk by chunk with its CRCs, JPEG, GIF and WebP down to their headers and
   end markers. It cannot tell you a picture is good, only that the file is a well-formed image
   of the stated size, and it gives the SHA-256 that goes into the evidence.
"""

import base64
import hashlib
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_SIDE = 4096
MAX_PIXELS = 16_000_000  # a small file can still declare a huge picture

ImageFormat = Literal["png", "jpeg", "gif", "webp"]
EXTENSIONS: dict[str, ImageFormat] = {
    ".png": "png",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".gif": "gif",
    ".webp": "webp",
}


class ImageError(Exception):
    """Generation failed or the bytes are not a valid image. The message is shown to the model."""


# ---------------------------------------------------------------------------- the image check


class ImageInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format: ImageFormat
    width: int
    height: int
    size_bytes: int
    sha256: str


def _png(data: bytes) -> tuple[int, int]:
    position = 8
    size: tuple[int, int] | None = None
    seen_data = False
    while True:
        if position + 12 > len(data):
            raise ImageError("the PNG is cut off before its end marker")
        (length,) = struct.unpack(">I", data[position : position + 4])
        tag = data[position + 4 : position + 8]
        end = position + 8 + length + 4
        if end > len(data):
            raise ImageError("the PNG is cut off inside a chunk")
        body = data[position + 8 : end - 4]
        (crc,) = struct.unpack(">I", data[end - 4 : end])
        if zlib.crc32(tag + body) & 0xFFFFFFFF != crc:
            raise ImageError(f"the PNG has a corrupt {tag.decode('latin-1')!r} chunk")
        if size is None:
            if tag != b"IHDR" or length != 13:
                raise ImageError("the PNG does not start with a valid header chunk")
            size = struct.unpack(">II", body[:8])
        elif tag == b"IDAT":
            seen_data = True
        elif tag == b"IEND":
            if end != len(data):
                raise ImageError("the PNG has extra bytes after its end marker")
            break
        position = end
    if not seen_data:
        raise ImageError("the PNG contains no image data")
    return size


_JPEG_SOF = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}


def _jpeg(data: bytes) -> tuple[int, int]:
    if not data.rstrip(b"\x00").endswith(b"\xff\xd9"):
        raise ImageError("the JPEG is cut off before its end marker")
    position = 2
    while position + 4 <= len(data):
        if data[position] != 0xFF:
            raise ImageError("the JPEG has a broken segment structure")
        marker = data[position + 1]
        if marker == 0xFF:  # fill byte
            position += 1
            continue
        if marker == 0xDA:
            break  # start of scan reached without a frame header
        (length,) = struct.unpack(">H", data[position + 2 : position + 4])
        if marker in _JPEG_SOF:
            if length < 8 or position + 9 > len(data):
                raise ImageError("the JPEG frame header is cut off")
            height, width = struct.unpack(">HH", data[position + 5 : position + 9])
            return width, height
        position += 2 + length
    raise ImageError("the JPEG has no frame header")


def _gif(data: bytes) -> tuple[int, int]:
    if len(data) < 14 or not data.endswith(b"\x3b"):
        raise ImageError("the GIF is cut off before its trailer")
    width, height = struct.unpack("<HH", data[6:10])
    return width, height


def _webp(data: bytes) -> tuple[int, int]:
    if len(data) < 21:
        raise ImageError("the WebP is too short")
    (riff_size,) = struct.unpack("<I", data[4:8])
    if riff_size + 8 != len(data):
        raise ImageError("the WebP length field does not match the file (cut off or padded)")
    kind = data[12:16]
    need = {b"VP8 ": 30, b"VP8L": 25, b"VP8X": 30}.get(kind)
    if need is None:
        raise ImageError("the WebP has an unknown format chunk")
    if len(data) < need:
        raise ImageError("the WebP header is cut off")
    if kind == b"VP8 ":
        width, height = struct.unpack("<HH", data[26:30])
        return width & 0x3FFF, height & 0x3FFF
    if kind == b"VP8L":
        (bits,) = struct.unpack("<I", data[21:25])
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    return (  # VP8X
        1 + int.from_bytes(data[24:27], "little"),
        1 + int.from_bytes(data[27:30], "little"),
    )


def inspect_image(data: bytes) -> ImageInfo:
    """What the bytes really are, or ImageError. Never trusts a file name or a stated type."""
    if not data:
        raise ImageError("the image is empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageError(f"the image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
    fmt: ImageFormat
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        fmt, (width, height) = "png", _png(data)
    elif data.startswith(b"\xff\xd8\xff"):
        fmt, (width, height) = "jpeg", _jpeg(data)
    elif data[:6] in (b"GIF87a", b"GIF89a"):
        fmt, (width, height) = "gif", _gif(data)
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        fmt, (width, height) = "webp", _webp(data)
    else:
        raise ImageError("the bytes are not a PNG, JPEG, GIF or WebP image")
    if not (0 < width <= MAX_SIDE and 0 < height <= MAX_SIDE):
        raise ImageError(f"the image is {width}x{height}; each side must be 1 to {MAX_SIDE}")
    if width * height > MAX_PIXELS:
        raise ImageError(f"the image is {width}x{height}, more than {MAX_PIXELS} pixels")
    return ImageInfo(
        format=fmt,
        width=width,
        height=height,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


# ---------------------------------------------------------------------------- the providers


@dataclass(frozen=True)
class GeneratedImage:
    data: bytes
    provider: str
    model: str | None = None


class ImageProvider(Protocol):
    """Anything that can make an image from a prompt."""

    name: str
    model: str | None
    # The URL this provider contacts, or None when it works offline. The tool uses it for the
    # network policy: a provider with an endpoint is only usable on an allowed host.
    endpoint: str | None
    # True when the picture is meant to show what the prompt describes. False for the offline
    # test provider, and the tool tells the model so, so it does not claim otherwise.
    depicts_prompt: bool

    async def generate(
        self, prompt: str, *, width: int, height: int, seed: int | None
    ) -> GeneratedImage: ...


def _png_bytes(width: int, height: int, rows: list[bytes]) -> bytes:
    def chunk(tag: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(tag + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", crc)

    raw = b"".join(b"\x00" + row for row in rows)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


class ProceduralProvider:
    """Offline test provider: the same prompt, seed and size always give the same PNG."""

    name: str = "procedural"
    model: str | None = None
    endpoint: str | None = None
    depicts_prompt: bool = False
    MAX_SIDE = 1024

    async def generate(
        self, prompt: str, *, width: int, height: int, seed: int | None
    ) -> GeneratedImage:
        if width > self.MAX_SIDE or height > self.MAX_SIDE:
            raise ImageError(
                f"the procedural provider makes at most {self.MAX_SIDE}x{self.MAX_SIDE}"
            )
        digest = hashlib.sha256(f"{prompt}|{seed}|{width}x{height}".encode()).digest()
        first, second = digest[0:3], digest[3:6]
        cell = 8 + digest[6] % 40
        span = max(width + height - 2, 1)
        rows: list[bytes] = []
        for y in range(height):
            row = bytearray()
            for x in range(width):
                mix = (x + y) / span
                shade = 24 if ((x // cell) + (y // cell)) % 2 else 0
                for channel in range(3):
                    value = first[channel] + (second[channel] - first[channel]) * mix
                    row.append(min(255, max(0, int(value) + shade)))
            rows.append(bytes(row))
        return GeneratedImage(_png_bytes(width, height, rows), self.name)


class OpenAIImagesProvider:
    """Any server with the OpenAI image protocol: POST {base_url}/images/generations.

    It asks for the picture as base64 inside the JSON reply, so no second download happens and no
    other host is contacted. A reply that only carries a URL is refused. One attempt only: a
    generation can cost money, so it is never retried behind the agent's back.
    """

    depicts_prompt: bool = True

    def __init__(
        self, *, base_url: str, api_key: str | None, model: str, name: str = "openai-images"
    ) -> None:
        self.name: str = name
        self.model: str | None = model
        self.endpoint: str | None = base_url.rstrip("/") + "/images/generations"
        self._api_key = api_key

    async def generate(
        self, prompt: str, *, width: int, height: int, seed: int | None
    ) -> GeneratedImage:
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "n": 1,
            "size": f"{width}x{height}",
        }
        if self.model and self.model.startswith("dall-e"):
            body["response_format"] = "b64_json"  # gpt-image models reject this field
        if seed is not None:
            body["seed"] = seed
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        assert self.endpoint is not None
        try:
            async with httpx.AsyncClient(timeout=120.0, follow_redirects=False) as client:
                response = await client.post(self.endpoint, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise ImageError(f"could not reach the image provider: {type(exc).__name__}") from None
        if response.status_code != 200:
            snippet = response.text[:200].replace(self._api_key or "\x00", "***")
            raise ImageError(f"the image provider answered HTTP {response.status_code}: {snippet}")
        try:
            item = response.json()["data"][0]
            encoded = item.get("b64_json")
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise ImageError("the image provider's reply has an unexpected shape") from None
        if not isinstance(encoded, str):
            raise ImageError("the image provider returned no inline image (b64_json)")
        try:
            data = base64.b64decode(encoded, validate=True)
        except ValueError:
            raise ImageError("the image provider returned invalid base64") from None
        return GeneratedImage(data, self.name, self.model)


PRESETS: dict[str, tuple[str, str]] = {
    # name: (base_url, default model). Unverified against the live service: see the README.
    "openai": ("https://api.openai.com/v1", "gpt-image-1"),
}


def build_image_provider(
    kind: str, *, api_key: str | None, base_url: str | None, model: str | None
) -> ImageProvider:
    """Choose the provider from settings. `procedural` is the default and needs nothing."""
    kind = (kind or "procedural").strip().lower()
    if kind == "procedural":
        return ProceduralProvider()
    if kind in PRESETS or kind == "custom":
        preset_url, preset_model = PRESETS.get(kind, ("", ""))
        url = base_url or preset_url
        chosen_model = model or preset_model
        if not url:
            raise ValueError("AEGIS_IMAGE_BASE_URL is required for AEGIS_IMAGE_PROVIDER=custom")
        if not chosen_model:
            raise ValueError("AEGIS_IMAGE_MODEL is required for AEGIS_IMAGE_PROVIDER=custom")
        return OpenAIImagesProvider(base_url=url, api_key=api_key, model=chosen_model, name=kind)
    known = ", ".join(["procedural", *PRESETS, "custom"])
    raise ValueError(f"unknown image provider {kind!r}. Known: {known}")


def require_extension(path: str) -> ImageFormat:
    """The format a file name promises, or ValueError listing what is accepted."""
    suffix = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
    try:
        return EXTENSIONS[suffix]
    except KeyError:
        raise ValueError(
            f"the file name must end in one of: {', '.join(sorted(EXTENSIONS))}"
        ) from None
