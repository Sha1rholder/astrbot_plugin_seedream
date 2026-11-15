"""Validate Seedream requests, download PNG outputs and maintain archives."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import math
import os
import shutil
import stat
import tempfile
import threading
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import jsonschema
import yaml
from PIL import Image, UnidentifiedImageError

REQUEST_SCHEMA: dict[str, Any] = json.loads(
    Path(__file__).with_name("request.schema.json").read_text(encoding="utf-8")
)
INPUT_SCHEMA: dict[str, Any] = REQUEST_SCHEMA["$defs"]["request"]
DEFAULT_ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
MAX_REFERENCE_BYTES = 30_000_000
STORAGE_LOCK = threading.Lock()


class ImageGenerationError(Exception):
    """An image error safe to return to the agent."""


class SeedreamAPI:
    """Own Seedream transport, reference validation and archive storage."""

    input_schema = INPUT_SCHEMA

    def __init__(self, config: Mapping[str, Any], plugin_dir: Path) -> None:
        """Keep configuration and metadata for lazy transport initialization.

        Args:
            config: Plugin configuration from AstrBot.
            plugin_dir: Directory containing plugin metadata.
        """
        self.config = config
        self.plugin_dir = plugin_dir
        self.client: httpx.AsyncClient | None = None

    def require_api_key(self) -> str:
        """Resolve plugin credentials before falling back to the environment.

        Returns:
            The configured API key.

        Raises:
            ImageGenerationError: If no usable credential is available.
        """
        key = (
            str(self.config.get("api_key") or "").strip()
            or os.getenv("VOLCENGINE_API_KEY", "").strip()
        )
        if not key:
            raise ImageGenerationError(
                "Image generation unavailable: no usable VOLCENGINE_API_KEY is configured."
            )
        return key

    def prepare(
        self, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], list[int | str]]:
        """Validate agent arguments and build the fixed Seedream payload.

        Args:
            arguments: Original tool arguments, without inferred defaults.

        Returns:
            The upstream body and ordered reference selectors.

        Raises:
            ImageGenerationError: If the arguments violate the tool contract.
        """
        try:
            jsonschema.validate(arguments, INPUT_SCHEMA)
        except jsonschema.ValidationError as exc:
            location = ".".join(map(str, exc.absolute_path)) or "arguments"
            raise ImageGenerationError(
                f"Invalid image arguments: {location} violates the {exc.validator} constraint. Read the usage instructions."
            ) from None
        if not arguments["prompt"].strip():
            raise ImageGenerationError("The image prompt must not be blank.")
        if not arguments["description"].strip():
            raise ImageGenerationError("The image description must not be blank.")
        return {
            "model": "doubao-seedream-5-5-261028",
            "prompt": arguments["prompt"].strip(),
            "layer_decomposition": False,
            "output_format": "png",
            "response_format": "url",
            "watermark": False,
            "background": arguments.get("background") or "auto",
        }, [
            item if isinstance(item, str) else int(item)
            for item in arguments.get("reference_images", [])
        ]

    @staticmethod
    def reference_payload(references: list[str]) -> list[str]:
        """Validate local references and preserve HTTPS references for the provider.

        Args:
            references: Raw Base64 image strings or HTTPS image URLs.

        Returns:
            Ordered MIME-qualified data URLs and unchanged HTTPS URLs.

        Raises:
            ImageGenerationError: If an image is corrupt or outside API limits.
        """
        images = []
        for position, encoded in enumerate(references):
            if encoded.startswith("https://"):
                try:
                    url = httpx.URL(encoded)
                    if (
                        not url.host
                        or url.userinfo
                        or url.fragment
                        or any(character.isspace() for character in encoded)
                    ):
                        raise ValueError
                except (ValueError, httpx.InvalidURL):
                    raise ImageGenerationError(
                        f"Invalid reference_images[{position}]: use an accessible HTTPS image URL without credentials or a fragment."
                    ) from None
                # Preserve signed queries exactly; the provider fetches and validates URLs.
                images.append(encoded)
                continue
            try:
                if len(encoded) > (MAX_REFERENCE_BYTES + 2) // 3 * 4:
                    raise ValueError
                raw = base64.b64decode(encoded, validate=True)
                if len(raw) > MAX_REFERENCE_BYTES:
                    raise ValueError
                with Image.open(io.BytesIO(raw)) as image:
                    width, height = image.size
                    if (
                        image.format not in ("PNG", "JPEG", "WEBP")
                        or min(width, height) <= 14
                        or width * height > 36_000_000
                        or max(width, height) > 16 * min(width, height)
                    ):
                        raise ValueError
                    mime = Image.MIME[image.format]
                    image.verify()
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
            except (
                ValueError,
                OSError,
                binascii.Error,
                UnidentifiedImageError,
                Image.DecompressionBombError,
            ):
                raise ImageGenerationError(
                    f"Invalid reference_images[{position}]. References must be valid PNG, JPEG or WebP images, "
                    "at most 30 MB, with each dimension above 14 pixels, at most 36 million pixels "
                    "and an aspect ratio between 1:16 and 16:1. No API request was made."
                ) from None
            images.append(f"data:{mime};base64,{encoded}")
        return images

    async def generate(
        self,
        payload: dict[str, Any],
        references: list[str],
        output_dir: Path,
        arguments: dict[str, Any],
        request_dir: Path | None = None,
        *,
        on_request_sent: Callable[[], None] | None = None,
        retention_floor: int | None = None,
    ) -> list[Path]:
        """Submit once, download the PNG and archive a lossless WebP.

        Args:
            payload: Validated Seedream options.
            references: Selected Base64 reference images or HTTPS URLs.
            output_dir: Root containing numbered archives.
            arguments: Original tool arguments to archive.
            request_dir: Previously reserved archive, if supplied.
            on_request_sent: Callback after POST body transmission completes.
            retention_floor: Oldest request still generating or awaiting delivery.

        Returns:
            A list containing the saved WebP path.

        Raises:
            ImageGenerationError: For invalid configuration, references or output.
            httpx.RequestError: If submission or download fails.
            OSError: If archive storage fails.
        """
        self.prepare(arguments)
        limit = self.storage_limit()
        key = self.require_api_key()
        try:
            endpoint = httpx.URL(
                str(self.config.get("base_url") or DEFAULT_ENDPOINT).strip()
            )
            if (
                endpoint.scheme not in ("http", "https")
                or not endpoint.host
                or endpoint.query
                or endpoint.fragment
                or endpoint.userinfo
            ):
                raise ValueError
        except (ValueError, httpx.InvalidURL):
            raise ImageGenerationError(
                "base_url must be a complete HTTP(S) image generation endpoint, "
                f"such as {DEFAULT_ENDPOINT}."
            ) from None
        try:
            timeout = float(self.config.get("timeout_seconds", 300))
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError
        except (TypeError, ValueError):
            raise ImageGenerationError(
                "timeout_seconds must be a positive, finite number of seconds."
            ) from None
        if self.client is None:
            # Credentials belong only to POST, never to the download or its redirects.
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout, connect=30), follow_redirects=False
            )
        if request_dir is None:
            request_dir = await asyncio.to_thread(
                self.create_request, output_dir, arguments
            )
        images = await asyncio.to_thread(self.reference_payload, references)
        body = dict(payload)
        if images:
            body["image"] = images
        await asyncio.to_thread(
            self.prune_requests,
            output_dir,
            limit,
            int(request_dir.name) if retention_floor is None else retention_floor,
        )
        sending_api_request = False

        async def trace(name: str, info: dict[str, Any]) -> None:
            """Acknowledge the POST body, excluding proxy CONNECT transmission.

            Args:
                name: HTTPcore transport event name.
                info: Event metadata, including the request at body start.
            """
            nonlocal sending_api_request
            if name in (
                "http11.send_request_body.started",
                "http2.send_request_body.started",
            ):
                sending_api_request = info["request"].method == b"POST"
            elif (
                name
                in (
                    "http11.send_request_body.complete",
                    "http2.send_request_body.complete",
                )
                and sending_api_request
                and on_request_sent is not None
            ):
                on_request_sent()

        response = await self.client.post(
            endpoint,
            json=body,
            headers={"Authorization": f"Bearer {key}"},
            extensions={"trace": trace},
        )
        if not response.is_success:
            explanation = {
                400: "The service rejected the parameters or content. Check the prompt and model access.",
                401: "Invalid API key. Check the plugin configuration.",
                403: "Model access is denied. Check account permissions.",
                404: "Model or endpoint is unavailable. Check base_url and model access.",
                429: "Rate limit or quota exceeded. Check account limits and balance.",
            }.get(response.status_code, "Image service returned an error.")
            raise ImageGenerationError(
                f"Image generation failed (HTTP {response.status_code}): {explanation} Do not automatically retry."
            )
        # A response proves submission even when a custom transport has no traces.
        if on_request_sent is not None:
            on_request_sent()
        try:
            result = response.json()
        except ValueError:
            raise ImageGenerationError(
                "Image service returned invalid JSON. Do not automatically retry."
            ) from None
        # Persist the provider response before downloading so URLs survive failures.
        response_path = request_dir / "response.yaml"
        with tempfile.TemporaryDirectory(
            prefix=".response-", dir=request_dir
        ) as temporary:
            staging = Path(temporary) / "response.yaml"
            await asyncio.to_thread(
                staging.write_text,
                yaml.safe_dump(result, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            await asyncio.to_thread(staging.replace, response_path)
        # The durable response is the source of truth for image downloads.
        result = yaml.safe_load(
            await asyncio.to_thread(response_path.read_text, encoding="utf-8")
        )
        data = result.get("data") if isinstance(result, dict) else None
        if (
            not isinstance(data, list)
            or len(data) != 1
            or not isinstance(data[0], dict)
            or result.get("error")
            or data[0].get("error")
        ):
            raise ImageGenerationError(
                "Image service did not return exactly one successful image. Do not automatically retry."
            )
        try:
            value = data[0].get("url")
            if not isinstance(value, str) or not value.strip():
                raise ValueError
            download_url = httpx.URL(value)
            if (
                download_url.scheme not in ("http", "https")
                or not download_url.host
                or download_url.userinfo
                or download_url.fragment
            ):
                raise ValueError
        except (ValueError, httpx.InvalidURL):
            raise ImageGenerationError(
                "Image service returned an invalid image URL. Do not automatically retry."
            ) from None
        try:
            downloaded = await self.client.get(download_url, follow_redirects=True)
        except httpx.RequestError:
            raise ImageGenerationError(
                "The generated image could not be downloaded. Generation may already be billed. Do not automatically retry."
            ) from None
        if not downloaded.is_success:
            raise ImageGenerationError(
                f"Generated image download failed (HTTP {downloaded.status_code}). Do not automatically retry."
            )
        path = await asyncio.to_thread(self.save_image, downloaded.content, request_dir)
        return [path]

    @staticmethod
    def save_image(raw: bytes, request_dir: Path) -> Path:
        """Convert downloaded bytes and atomically publish a lossless WebP.

        Args:
            raw: Downloaded image bytes.
            request_dir: Existing request archive directory.

        Returns:
            The published WebP path.

        Raises:
            OSError: If image decoding, conversion or storage fails.
        """
        with (
            STORAGE_LOCK,
            tempfile.TemporaryDirectory(prefix=".image-", dir=request_dir) as temporary,
        ):
            staging = Path(temporary) / "0.webp"
            with Image.open(io.BytesIO(raw)) as image:
                image.save(staging, "WEBP", lossless=True, exact=True)
            destination = request_dir / "0.webp"
            staging.replace(destination)
            # Remove an existing PNG only after its replacement is published.
            (request_dir / "0.png").unlink(missing_ok=True)
        return destination

    def storage_limit(self) -> int:
        """Return the archive limit in decimal bytes.

        Returns:
            Maximum archive storage in bytes.

        Raises:
            ImageGenerationError: If the configured limit is not a positive integer.
        """
        limit = self.config.get("max_storage_gb", 10)
        if type(limit) is not int or limit <= 0:
            raise ImageGenerationError("max_storage_gb must be a positive integer.")
        return limit * 1_000_000_000

    @staticmethod
    def next_request_id(output_dir: Path) -> int:
        """Derive the next ID from existing numeric archive directories.

        Args:
            output_dir: Root of numbered request archives.

        Returns:
            The next unused numeric ID, starting at zero.
        """
        if not output_dir.exists():
            return 0
        existing = (
            int(path.name)
            for path in output_dir.iterdir()
            if path.name.isascii()
            and path.name.isdecimal()
            and not path.is_symlink()
            and path.is_dir()
        )
        return max(existing, default=-1) + 1

    def create_request(
        self, output_dir: Path, arguments: dict[str, Any], *, user_id: str | None = None
    ) -> Path:
        """Atomically reserve a numbered directory with original arguments.

        Args:
            output_dir: Root of numbered request archives.
            arguments: Original tool arguments, preserving omitted and null fields.
            user_id: Original sender ID, if called from a message event.

        Returns:
            The reserved request directory.

        Raises:
            ImageGenerationError: If arguments, storage settings or metadata are invalid.
            OSError: If the archive cannot be written.
        """
        arguments = deepcopy(arguments)
        self.prepare(arguments)
        self.storage_limit()
        try:
            metadata = yaml.safe_load(
                (self.plugin_dir / "metadata.yaml").read_text(encoding="utf-8")
            )
            version = metadata["version"]
            if not isinstance(version, str) or not version:
                raise ValueError
        except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError):
            raise ImageGenerationError(
                "Cannot read the plugin version from metadata.yaml. No API request was made."
            ) from None
        archive = {
            "plugin_version": version,
            "request": arguments,
        }
        if user_id is not None:
            archive["user_id"] = user_id
        with STORAGE_LOCK:
            output_dir.mkdir(parents=True, exist_ok=True)
            directory = output_dir / str(self.next_request_id(output_dir))
            if directory.exists() or directory.is_symlink():
                raise FileExistsError(f"Request directory already exists: {directory}")
            with tempfile.TemporaryDirectory(
                prefix=".pending-", dir=output_dir
            ) as temporary:
                staging = Path(temporary)
                (staging / "request.yaml").write_text(
                    f"# yaml-language-server: $schema={REQUEST_SCHEMA['$id']}\n"
                    + yaml.safe_dump(
                        archive,
                        allow_unicode=True,
                        sort_keys=False,
                    ),
                    encoding="utf-8",
                )
                staging.replace(directory)
            return directory

    @staticmethod
    def storage_size(directory: Path) -> int:
        """Sum regular file sizes recursively without following symbolic links.

        Args:
            directory: Archive directory to inspect.

        Returns:
            Total regular file bytes.

        Raises:
            OSError: If directory contents cannot be inspected.
        """
        total = 0
        with os.scandir(directory) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    total += SeedreamAPI.storage_size(Path(entry.path))
                elif stat.S_ISREG(info.st_mode):
                    total += info.st_size
        return total

    @staticmethod
    def prune_requests(output_dir: Path, limit: int, current_id: int) -> None:
        """Remove oldest completed archives when the storage limit is exceeded.

        Args:
            output_dir: Root of numbered request archives.
            limit: Maximum storage size in bytes.
            current_id: Oldest protected request, including undelivered images.

        Raises:
            ImageGenerationError: If protected or unrelated files prevent cleanup.
            OSError: If an archive cannot be inspected or deleted.
        """
        with STORAGE_LOCK:
            total = SeedreamAPI.storage_size(output_dir)
            if total <= limit:
                return
            directories = sorted(
                (
                    directory
                    for directory in output_dir.iterdir()
                    if directory.name.isascii()
                    and directory.name.isdecimal()
                    and str(int(directory.name)) == directory.name
                    and int(directory.name) < current_id
                    and not directory.is_symlink()
                    and directory.is_dir()
                ),
                key=lambda directory: int(directory.name),
            )
            for directory in directories:
                size = SeedreamAPI.storage_size(directory)
                shutil.rmtree(directory)
                total -= size
                if total < limit:
                    return
            raise ImageGenerationError(
                "Image archive storage cannot be reduced below max_storage_gb without "
                "deleting current or queued requests or unrelated files. Increase the limit "
                "or free space in the image archive directory. No API request was made."
            )

    async def close(self) -> None:
        """Close an existing HTTP client and clear it so repeated cleanup is safe."""
        if self.client is not None:
            await self.client.aclose()
            self.client = None
