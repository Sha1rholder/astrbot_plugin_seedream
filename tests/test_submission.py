"""Verify blocking admission ends at transmission, before generation completes."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from astrbot.api.message_components import Image

pytestmark = pytest.mark.asyncio

ARGUMENTS = {"prompt": "cat", "description": "A cat"}
SENT = "Image generation request submitted"


@pytest.mark.parametrize("status", [200, 401])
async def test_real_http_returns_after_upload_before_response(
    service, event, png, finish_turn, status
):
    received = asyncio.Event()
    respond = asyncio.Event()
    downloaded = asyncio.Event()
    download_release = asyncio.Event()
    handled = set()
    requests = []

    async def handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        handled.add(task)
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            requests.append(headers)
            if headers.startswith(b"POST"):
                length = next(
                    int(line.split(b":", 1)[1])
                    for line in headers.split(b"\r\n")
                    if line.lower().startswith(b"content-length:")
                )
                body = json.loads(await reader.readexactly(length))
                assert body["model"] == "doubao-seedream-5-5-261028"
                assert b"authorization: bearer test-key" in headers.lower()
                received.set()
                await respond.wait()
                result = (
                    {"data": [{"url": f"http://127.0.0.1:{port}/download"}]}
                    if status == 200
                    else {"error": "private"}
                )
                content = json.dumps(result).encode()
                code = status
            else:
                assert b"authorization:" not in headers.lower()
                downloaded.set()
                await download_release.wait()
                content, code = png, 200
            writer.write(
                f"HTTP/1.1 {code} Test\r\nContent-Length: {len(content)}\r\nConnection: close\r\n\r\n".encode()
                + content
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            handled.discard(task)

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    service.api.config["base_url"] = f"http://127.0.0.1:{port}/custom/generate"
    service.api.client = httpx.AsyncClient(trust_env=False)
    call = asyncio.create_task(service.generate(event, ARGUMENTS))
    try:
        await asyncio.wait_for(received.wait(), 3)
        assert SENT in await asyncio.wait_for(asyncio.shield(call), 3)
        assert not (service.output_dir / "0/0.png").exists()
        service.context.send_message.assert_not_awaited()
        respond.set()
        if status == 200:
            await asyncio.wait_for(downloaded.wait(), 3)
            assert not (service.output_dir / "0/0.png").exists()
            download_release.set()
        await finish_turn()
        service.context.send_message.assert_awaited_once()
        assert len(requests) == (2 if status == 200 else 1)
        if status == 200:
            assert (service.output_dir / "0/0.png").read_bytes() == png
        else:
            assert "HTTP 401" in service.context.llm_generate.call_args.kwargs["prompt"]
        assert not service._pending_requests
    finally:
        respond.set()
        download_release.set()
        call.cancel()
        await asyncio.gather(call, return_exceptions=True)
        server.close()
        await server.wait_closed()
        if handled:
            await asyncio.wait_for(asyncio.gather(*handled), 3)


@pytest.mark.parametrize(
    "stage", ["connect", "write", "proxy_connect", "http_rejection"]
)
async def test_pre_submission_transport_failure_returns_to_model(
    service, event, stage, mock_http
):
    """Neither entering post() nor sending a proxy CONNECT counts as submission."""
    entered = asyncio.Event()
    release = asyncio.Event()
    requests = []

    async def fail(request):
        requests.append(request)
        if stage == "proxy_connect":
            trace = request.extensions["trace"]
            await trace(
                "http11.send_request_body.started",
                {"request": SimpleNamespace(method=b"CONNECT")},
            )
            await trace("http11.send_request_body.complete", {})
        entered.set()
        await release.wait()
        if stage == "http_rejection":
            return httpx.Response(403)
        if stage == "write":
            raise httpx.WriteError("private")
        raise httpx.ConnectError("private")

    mock_http(service.api, fail)
    call = asyncio.create_task(service.generate(event, ARGUMENTS))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert not call.done()
        release.set()
        result = await asyncio.wait_for(call, 1)
        assert SENT not in result
        assert (
            "HTTP 403" if stage == "http_rejection" else "connection failed"
        ) in result
        assert "private" not in result
        assert len(requests) == 1
        service.context.send_message.assert_not_awaited()
        assert not service._pending_requests
    finally:
        release.set()
        call.cancel()
        await asyncio.gather(call, return_exceptions=True)


async def test_invalid_reference_payload_returns_to_model(service, event, mock_http):
    """Readable attachments still need format validation before acknowledging submission."""
    event.get_messages.return_value = [Image.fromBase64("invalid-image")]
    requests = []

    def unexpected_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise AssertionError("Invalid references must not trigger an HTTP request")

    mock_http(service.api, unexpected_request)
    result = await service.generate(
        event, {"prompt": "cat", "reference_images": [0], "description": "A cat"}
    )
    assert "References must be valid" in result
    assert SENT not in result
    assert not requests
    service.context.send_message.assert_not_awaited()
    assert not service._pending_requests


@pytest.mark.parametrize("protocol", ["http11", "http2"])
async def test_body_started_does_not_release_tool_but_completed_does(
    service, event, protocol, finish_turn, mock_http
):
    """A partial upload must remain blocking for both supported HTTP protocols."""
    started = asyncio.Event()
    finish_upload = asyncio.Event()
    sent = asyncio.Event()
    finish_response = asyncio.Event()

    async def transport(request):
        trace = request.extensions["trace"]
        await trace(
            f"{protocol}.send_request_body.started",
            {"request": SimpleNamespace(method=b"POST")},
        )
        started.set()
        await finish_upload.wait()
        await trace(f"{protocol}.send_request_body.complete", {})
        sent.set()
        await finish_response.wait()
        raise httpx.ReadTimeout("private")

    mock_http(service.api, transport)
    call = asyncio.create_task(service.generate(event, ARGUMENTS))
    try:
        await asyncio.wait_for(started.wait(), 1)
        assert not call.done()
        finish_upload.set()
        result = await asyncio.wait_for(call, 1)
        assert sent.is_set()
        assert SENT in result
        service.context.send_message.assert_not_awaited()
        finish_response.set()
        await finish_turn()
        assert "timed out" in service.context.llm_generate.call_args.kwargs["prompt"]
        assert (
            "Do not automatically retry"
            not in service.context.send_message.call_args.args[1]
        )
    finally:
        finish_upload.set()
        finish_response.set()
        call.cancel()
        await asyncio.gather(call, return_exceptions=True)


async def test_cancelling_tool_before_submission_cancels_pending_upload(
    service, event, mock_http
):
    """Timed out tool calls must not submit a request later in the background."""
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def transport(request: httpx.Request) -> httpx.Response:
        entered.set()
        try:
            await asyncio.Event().wait()
            raise AssertionError("Pending upload unexpectedly resumed")
        finally:
            cancelled.set()

    mock_http(service.api, transport)
    call = asyncio.create_task(service.generate(event, ARGUMENTS))
    await asyncio.wait_for(entered.wait(), 1)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    assert cancelled.is_set()
    assert not service._tasks
    assert not service._pending_requests
    service.context.send_message.assert_not_awaited()


@pytest.mark.parametrize("explanation", ["error", "empty"])
async def test_failure_explanation_fallback_never_sends_internal_instructions(
    service, event, finish_turn, explanation
):
    from unittest.mock import AsyncMock

    from data.plugins.astrbot_plugin_seedream.src.generate import (
        ImageGenerationError,
    )

    async def fail(*args, on_request_sent, **kwargs):
        on_request_sent()
        raise ImageGenerationError(
            "status=failed; error=service_unavailable. Do not automatically retry."
        )

    service.api.generate = AsyncMock(side_effect=fail)
    if explanation == "error":
        service.context.llm_generate.side_effect = RuntimeError("private")
    else:
        service.context.llm_generate.return_value.completion_text = ""
    await service.generate(event, ARGUMENTS)
    await finish_turn()
    message = service.context.send_message.call_args.args[1]
    assert "图片生成失败" in message
    assert "Do not automatically retry" not in message
    assert "private" not in message
    service.api.generate.assert_awaited_once()
