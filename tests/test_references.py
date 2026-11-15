"""Exercise local references through both API request and archive formats."""

import asyncio
import base64
import json
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
import yaml
from data.plugins.astrbot_plugin_seedream.src.generate import (
    MAX_REFERENCE_BYTES,
)

pytestmark = pytest.mark.asyncio

ARGUMENTS = {"prompt": "Use the reference images", "description": "Edited image"}


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "directory",
        "relative",
        "url",
        "corrupt",
        "gif",
        "large",
        "unreadable",
    ],
)
async def test_invalid_local_reference_never_submits(
    service, event, tmp_path, webp, monkeypatch, kind, mock_http, image_bytes
):
    """Reject invalid sources before HTTP submission without exposing host errors."""
    path = tmp_path / "reference.png"
    value = str(path)
    if kind == "directory":
        path.mkdir()
    elif kind == "relative":
        value = "reference.png"
    elif kind == "url":
        value = "http://example.invalid/reference.png"
    elif kind == "corrupt":
        path.write_bytes(b"not an image")
    elif kind == "gif":
        path.write_bytes(image_bytes("GIF"))
    elif kind == "large":
        with path.open("wb") as source:
            source.truncate(MAX_REFERENCE_BYTES + 1)
    elif kind == "unreadable":
        path.write_bytes(webp)
        original_open = Path.open

        def open_file(self, *args, **kwargs):
            if self == path:
                raise PermissionError("private host details")
            return original_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", open_file)

    handler = Mock(side_effect=AssertionError("Unexpected API request"))
    api = service.api
    mock_http(api, handler)
    result = await service.generate(event, {**ARGUMENTS, "reference_images": [value]})
    assert "reference_images[0]" in result
    assert "Image generation request submitted" not in result
    assert "private host details" not in result
    assert str(path) not in result
    handler.assert_not_called()
    assert not service._tasks
    service.context.send_message.assert_not_awaited()


@pytest.mark.parametrize(
    "suffix,format_name,mime",
    [
        (".jpg", "JPEG", "image/jpeg"),
        (".JpEg", "JPEG", "image/jpeg"),
        (".png", "PNG", "image/png"),
        (".webp", "WEBP", "image/webp"),
        (".png", "WEBP", "image/webp"),
    ],
)
async def test_mixed_reference_order_and_snapshot(
    service,
    event,
    tmp_path,
    png,
    webp,
    image_bytes,
    suffix,
    format_name,
    mime,
    monkeypatch,
    mock_http,
    finish_turn,
):
    from astrbot.api.message_components import Image, Reply

    path = tmp_path / f"参考 图{suffix}"
    raw = image_bytes(format_name)
    path.write_bytes(raw)
    event.get_messages.return_value = [
        Reply(id="old", chain=[Image.fromBytes(png)]),
        Image.fromBytes(webp),
    ]
    loaded = asyncio.Event()
    loop = asyncio.get_running_loop()
    original_loader = service._local_reference

    def load(value):
        result = original_loader(value)
        loop.call_soon_threadsafe(loaded.set)
        return result

    monkeypatch.setattr(service, "_local_reference", load)
    requests = []

    def respond(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, content=png)
        body = json.loads(request.content)
        assert body["image"] == [
            f"data:{mime};base64,{base64.b64encode(raw).decode()}",
            f"data:image/png;base64,{base64.b64encode(png).decode()}",
            f"data:image/webp;base64,{base64.b64encode(webp).decode()}",
        ]
        assert str(path) not in request.content.decode()
        return httpx.Response(
            200, json={"data": [{"url": "https://cdn.invalid/image"}]}
        )

    mock_http(service.api, respond)
    arguments = {**ARGUMENTS, "reference_images": [str(path), 1, 0]}
    async with service._generation_lock:
        task = asyncio.create_task(service.generate(event, arguments))
        await asyncio.wait_for(loaded.wait(), 5)
        path.unlink()
    assert "Image generation request submitted" in await asyncio.wait_for(task, 5)
    await finish_turn()
    assert len(requests) == 2
    assert (
        yaml.safe_load((service.output_dir / "0/request.yaml").read_text())["request"]
        == arguments
    )
    assert (service.output_dir / "0/0.png").read_bytes() == png


async def test_https_references_preserve_signatures_and_mixed_order(
    service, event, tmp_path, png, webp, mock_http, finish_turn
):
    """Forward signed URLs unchanged without fetching them on the bot host."""
    from astrbot.api.message_components import Image

    url = "https://cdn.example.com/output?x-signature=a%2Fb%2Bc&x-expires=123"
    path = tmp_path / "local.png"
    path.write_bytes(png)
    event.get_messages.return_value = [Image.fromBytes(webp)]
    requests = []

    def respond(request):
        requests.append(request)
        if request.method == "GET":
            assert str(request.url) == "https://cdn.invalid/result"
            assert "authorization" not in request.headers
            return httpx.Response(200, content=png)
        assert json.loads(request.content)["image"] == [
            url,
            f"data:image/webp;base64,{base64.b64encode(webp).decode()}",
            f"data:image/png;base64,{base64.b64encode(png).decode()}",
        ]
        return httpx.Response(
            200, json={"data": [{"url": "https://cdn.invalid/result"}]}
        )

    mock_http(service.api, respond)
    arguments = {**ARGUMENTS, "reference_images": [url, 0, str(path)]}
    result = await service.generate(event, arguments)
    assert "Image generation request submitted" in result
    await finish_turn()
    assert len(requests) == 2
    assert (
        yaml.safe_load((service.output_dir / "0/request.yaml").read_text())["request"]
        == arguments
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://user:secret@example.com/image.png",
        "https://example.com/image.png#fragment",
        "https:///image.png",
        "https://example.com:invalid/image.png",
    ],
)
async def test_invalid_https_reference_does_not_submit(service, event, mock_http, url):
    """Reject malformed URLs without disclosing signed links or credentials."""
    handler = Mock(side_effect=AssertionError("Unexpected request"))
    mock_http(service.api, handler)
    result = await service.generate(event, {**ARGUMENTS, "reference_images": [url]})
    assert "Invalid reference_images[0]" in result
    assert url not in result
    handler.assert_not_called()


async def test_duplicate_and_excess_https_references_rejected(
    service, event, mock_http
):
    """Apply the same uniqueness and count limits to URL references."""
    handler = Mock(side_effect=AssertionError("Unexpected request"))
    mock_http(service.api, handler)
    for urls in (
        ["https://example.com/image"] * 2,
        [f"https://example.com/{index}" for index in range(11)],
    ):
        result = await service.generate(event, {**ARGUMENTS, "reference_images": urls})
        assert "Invalid image arguments" in result
    handler.assert_not_called()
