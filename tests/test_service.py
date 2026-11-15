"""Test the plugin entry point, tool calls, message handling, and image delivery."""

import asyncio
import io
import json
import time
from unittest.mock import AsyncMock, Mock, create_autospec

import httpx
import pytest
import yaml
from astrbot.api import ToolSet
from astrbot.api.message_components import Image, Plain, Reply
from astrbot.api.provider import ProviderRequest
from data.plugins.astrbot_plugin_seedream import main as plugin_module
from data.plugins.astrbot_plugin_seedream.src import (
    generate as api_module,
)
from PIL import Image as PillowImage

ARGUMENTS = {"prompt": "cat", "description": "A cat"}


@pytest.mark.asyncio
async def test_tool_executor_to_webp_delivery(
    service, event, png, mock_http, tool_context, finish_turn
):
    from astrbot.core.astr_agent_runtime.tool_executor import FunctionToolExecutor
    from astrbot.core.star.context import _resolve_tool_handler_module_path
    from mcp.types import TextContent

    tools = service.context.add_llm_tools.call_args.args
    assert {t.name for t in tools} == {
        "get_seedream_image_generate_instructions",
        "seedream_image_generate",
    }
    assert all(
        _resolve_tool_handler_module_path(t) == plugin_module.__name__ for t in tools
    )
    assert len(ToolSet(list(tools)).openai_schema()) == 2
    assert len(ToolSet(list(tools)).anthropic_schema()) == 2
    from google.genai.types import Tool

    Tool.model_validate(ToolSet(list(tools)).google_schema())
    event.chain_result.side_effect = lambda parts: (
        plugin_module.AstrMessageEvent.chain_result(event, parts)
    )
    requests = []

    def handler(request):
        requests.append(request)
        if request.method == "POST":
            assert json.loads(request.content)["prompt"] == "cat"
            return httpx.Response(
                200, json={"data": [{"url": "https://cdn.invalid/image"}]}
            )
        return httpx.Response(200, content=png)

    mock_http(service.api, handler)
    result = [
        item
        async for item in FunctionToolExecutor._execute_local(
            tools[1],
            tool_context,
            prompt=ARGUMENTS["prompt"],
            description=ARGUMENTS["description"],
        )
    ]
    assert result[0] is not None
    assert isinstance(result[0].content[0], TextContent)
    assert "Image generation request submitted" in result[0].content[0].text
    await finish_turn()
    message = service.context.send_message.call_args.args[1]
    assert isinstance(message, plugin_module.MessageEventResult)
    assert isinstance(message.chain[0], Plain)
    assert message.chain[0].text == ARGUMENTS["description"]
    assert isinstance(message.chain[1], Image)
    assert (
        message.chain[1].file
        == Image.fromFileSystem(str(service.output_dir / "0/0.webp")).file
    )
    assert len(message.chain) == 2
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_plugins_coexist_with_independent_tools_and_storage(
    service, tmp_path, monkeypatch
):
    from types import MethodType, SimpleNamespace

    from astrbot.api.star import Context, StarTools

    other_module = pytest.importorskip(
        "data.plugins.astrbot_plugin_openai_image_generation.main"
    )
    context = service.context
    context.provider_manager = SimpleNamespace(llm_tools=ToolSet([]))
    context.add_llm_tools = MethodType(Context.add_llm_tools, context)
    monkeypatch.setattr(StarTools, "get_data_dir", lambda name: tmp_path / name)
    seedream = plugin_module.SeedreamPlugin(context, service.api.config)
    other = other_module.WebpImagePlugin(context, service.api.config)
    try:
        names = [tool.name for tool in context.provider_manager.llm_tools.func_list]
        assert len(names) == len(set(names)) == 5
        assert seedream.service.output_dir != other.service.output_dir
        assert seedream.service.requests_dir != other.service.requests_dir
        assert seedream.service.system_hint != other.service.system_hint
    finally:
        await seedream.terminate()
        await other.terminate()


@pytest.mark.asyncio
async def test_cleanup_protects_pending_delivery(
    service, event, png, monkeypatch, mock_http, finish_turn
):
    entered = asyncio.Event()
    release = asyncio.Event()
    original_send = service.context.send_message

    async def slow_send(origin, message):
        entered.set()
        await release.wait()
        return await original_send(origin, message)

    service.context.send_message = slow_send
    mock_http(
        service.api,
        lambda request: (
            httpx.Response(200, json={"data": [{"url": "https://cdn.invalid/image"}]})
            if request.method == "POST"
            else httpx.Response(200, content=png)
        ),
    )
    await service.generate(event, ARGUMENTS)
    try:
        await service.mark_agent_finished(event)
        await service.release_image_results(event)
        await asyncio.wait_for(entered.wait(), 3)
        size = (service.output_dir / "0/request.yaml").stat().st_size
        monkeypatch.setattr(service.api, "storage_limit", lambda: size + 1)
        assert "No API request was made" in await service.generate(event, ARGUMENTS)
        assert (service.output_dir / "0/0.webp").is_file()
        assert not (service.output_dir / "1/0.webp").exists()
        release.set()
        await finish_turn()
        assert "Image generation request submitted" in await service.generate(
            event, ARGUMENTS
        )
        await finish_turn()
        assert not (service.output_dir / "0").exists()
        assert (service.output_dir / "2/0.webp").is_file()
    finally:
        release.set()


@pytest.mark.asyncio
async def test_hint_is_idempotent_and_unconditional(plugin, service, event):
    """Verify the hint is added once even with missing or disabled tools."""
    tools = list(service.context.add_llm_tools.call_args.args)
    req = ProviderRequest(system_prompt="Original persona", func_tool=ToolSet(tools))
    await plugin.add_image_hint(event, req)
    await plugin.add_image_hint(event, req)
    assert req.system_prompt == "Original persona\n" + service.system_hint
    for restricted in (None, ToolSet(tools[:1])):
        req = ProviderRequest(system_prompt="Original persona", func_tool=restricted)
        await plugin.add_image_hint(event, req)
        assert req.system_prompt == "Original persona\n" + service.system_hint
    tools[0].active = False
    req = ProviderRequest(system_prompt="Original persona", func_tool=ToolSet(tools))
    await plugin.add_image_hint(event, req)
    assert req.system_prompt == "Original persona\n" + service.system_hint


def test_reference_order_and_instructions(service, event):
    """Verify reference ordering, available indices, and model instructions."""
    first = Image.fromBase64("first")
    second = Image.fromBase64("second")
    quoted = Image.fromBase64("quoted")
    event.get_messages.return_value = [Reply(id="old", chain=[quoted]), first, second]
    assert service.reference_images(event) == [first, second, quoted]
    guide = service.instructions(event)
    assert "0, 1, 2" in guide
    assert (
        f"Image archive directory on the AstrBot host: {service.output_dir.resolve()}"
        in guide
    )


@pytest.mark.asyncio
async def test_help_command_uses_seedream_guide(plugin, service, event):
    messages = [message async for message in plugin.seedream_help(event)]
    assert messages == [service.instructions(event)]
    assert service.instructions_tool.guide in messages[0]


def test_instructions_report_missing_key(service, event):
    service.api.config["api_key"] = " "
    result = service.instructions(event)
    assert "no usable VOLCENGINE_API_KEY" in result
    assert service.instructions_tool.guide not in result


@pytest.mark.asyncio
async def test_generation_rechecks_key_before_enqueueing(service, event):
    """A key removed after reading instructions must prevent a queued acknowledgement."""
    assert service.instructions_tool.guide in service.instructions(event)
    service.api.config["api_key"] = ""
    service.api.generate = AsyncMock()
    result = await service.generate(event, ARGUMENTS)
    assert "no usable VOLCENGINE_API_KEY" in result
    assert "Image generation request submitted" not in result
    assert not service._tasks
    assert (service.output_dir / "0" / "request.yaml").is_file()
    assert (service.requests_dir / "123456.log").is_file()
    service.api.generate.assert_not_awaited()
    service.context.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_references_do_not_call_api(service, event):
    """Verify nonexistent references prevent generation and delivery."""
    service.api.generate = AsyncMock()
    result = await service.generate(event, {**ARGUMENTS, "reference_images": [0]})
    assert "does not exist" in result
    assert "No API request was made" in result
    assert "correct the arguments" in result
    assert "Do not automatically retry" not in result
    assert not service._tasks
    service.api.generate.assert_not_awaited()
    service.context.send_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("indices", [[], [0], [1], [1, 0]])
async def test_only_selected_references_are_loaded(
    service, event, monkeypatch, indices, finish_turn
):
    first, second = Image.fromBase64("one"), Image.fromBase64("two")
    event.get_messages.return_value = [first, second]
    loader = create_autospec(Image.convert_to_base64, return_value="selected-base64")
    monkeypatch.setattr(Image, "convert_to_base64", loader)
    service.api.generate = AsyncMock(return_value=[])
    result = await service.generate(
        event,
        {
            **ARGUMENTS,
            "reference_images": indices,
        },
    )
    assert "Image generation request submitted" in result
    await finish_turn()
    assert [call.args[0] for call in loader.await_args_list] == [
        [first, second][i] for i in indices
    ]
    assert service.api.generate.call_args.args[1] == ["selected-base64"] * len(indices)


@pytest.mark.asyncio
async def test_reference_loading_failure_prevents_api_call(service, event, monkeypatch):
    """Verify attachment loading errors prevent API calls without exposing private paths."""
    event.get_messages.return_value = [Image.fromBase64("one")]
    monkeypatch.setattr(
        Image, "convert_to_base64", AsyncMock(side_effect=OSError("private path"))
    )
    service.api.generate = AsyncMock()
    result = await service.generate(event, {**ARGUMENTS, "reference_images": [0]})
    assert "No API request" in result
    assert "Image generation request submitted" not in result
    assert "private path" not in result
    service.api.generate.assert_not_awaited()
    service.context.send_message.assert_not_awaited()
    assert not service._pending_requests


@pytest.mark.asyncio
async def test_combined_delivery_failure_never_regenerates(
    service, event, tmp_path, finish_turn
):
    """Verify failed delivery reports uncertainty without generating or sending again."""
    service.api.generate = AsyncMock(return_value=[tmp_path / "0.webp"])
    service.context.send_message.side_effect = RuntimeError("private upstream response")
    service.logger = Mock()
    result = await service.generate(
        event,
        {"prompt": "Generate a cat", "description": "A cat"},
    )
    assert "Image generation request submitted" in result
    await finish_turn()
    service.logger.warning.assert_called_once()
    assert "Delivery status is unknown" in service.logger.warning.call_args.args[0]
    assert "private upstream" not in str(service.logger.warning.call_args)
    service.api.generate.assert_awaited_once()
    service.context.send_message.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_stage", ["reference", "api"])
async def test_tool_blocks_until_reference_loading_and_api_submission(
    service, event, monkeypatch, tmp_path, blocked_stage, tool_context
):
    """Slow references and API submission must block the actual tool acknowledgement."""
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_work(*_args, **_kwargs):
        entered.set()
        await release.wait()
        return (
            "reference-base64"
            if blocked_stage == "reference"
            else [tmp_path / "image.png"]
        )

    reference = Image.fromBase64("original")
    event.get_messages.return_value = [reference]
    monkeypatch.setattr(
        Image,
        "convert_to_base64",
        AsyncMock(side_effect=slow_work)
        if blocked_stage == "reference"
        else AsyncMock(return_value="reference-base64"),
    )
    service.api.generate = (
        AsyncMock(side_effect=slow_work)
        if blocked_stage == "api"
        else AsyncMock(return_value=[tmp_path / "image.png"])
    )
    arguments = {**ARGUMENTS, "reference_images": [0]}
    tool = service.context.add_llm_tools.call_args.args[1]
    origin = event.unified_msg_origin
    call = asyncio.create_task(tool.call(tool_context, **arguments))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert not call.done()
        directory = service.output_dir / "0"
        assert (
            yaml.safe_load((directory / "request.yaml").read_text())["request"]
            == arguments
        )
        assert {path.name for path in directory.iterdir()} == {"request.yaml"}
        service.context.send_message.assert_not_awaited()
        assert len(service._tasks) == 1
        # Later event or caller changes cannot redirect or rewrite the queued job.
        event.unified_msg_origin = "webchat:FriendMessage:another-session"
        event.get_messages.return_value = []
        arguments["prompt"] = "changed"
        release.set()
        result = await asyncio.wait_for(call, timeout=1)
        assert "Image generation request submitted" in result
        assert "The result is not ready yet" in result
        assert "Do not send the image yourself" in result
        await service.mark_agent_finished(event)
        await service.release_image_results(event)
        await asyncio.wait_for(asyncio.gather(*service._tasks), timeout=1)
        assert not service._tasks
        assert service.api.generate.call_args.args[0]["prompt"] == "cat"
        assert service.api.generate.call_args.args[3]["prompt"] == "cat"
        assert service.api.generate.call_args.args[4] == directory
        service.context.send_message.assert_awaited_once()
        assert service.context.send_message.call_args.args[0] == origin
    finally:
        release.set()
        await asyncio.gather(call, return_exceptions=True)


@pytest.mark.asyncio
async def test_terminate_cancels_running_and_queued_jobs_before_closing(
    plugin, service, event
):
    """Unloading cancels both active and queued work before closing the HTTP client."""
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow_generate(*_args, **_kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def close():
        assert cancelled.is_set()
        assert not service._tasks

    service.api.generate = AsyncMock(side_effect=slow_generate)
    service.api.close = AsyncMock(side_effect=close)
    first = asyncio.create_task(service.generate(event, ARGUMENTS))
    await asyncio.wait_for(entered.wait(), timeout=1)
    queued = asyncio.Event()
    run = service._run_generation

    async def run_second(*args):
        queued.set()
        await run(*args)

    service._run_generation = run_second
    second = asyncio.create_task(service.generate(event, ARGUMENTS))
    await asyncio.wait_for(queued.wait(), 1)
    tasks = list(service._tasks)
    assert len(tasks) == 2
    await asyncio.wait_for(plugin.terminate(), timeout=1)
    assert all(task.cancelled() for task in tasks)
    service.api.generate.assert_awaited_once()
    service.api.close.assert_awaited_once()
    service.context.send_message.assert_not_awaited()
    assert "shutting down" in await service.generate(event, ARGUMENTS)
    assert not service._tasks
    results = await asyncio.gather(first, second, return_exceptions=True)
    assert all(isinstance(result, asyncio.CancelledError) for result in results)
    assert not service._pending_requests


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,expected",
    [
        (api_module.ImageGenerationError("Invalid API key"), "Invalid API key"),
        (httpx.ReadTimeout("private response"), "timed out"),
        (httpx.ConnectError("private response"), "connection failed"),
        (OSError("private path"), "Could not read or save image files"),
        (RuntimeError("private upstream response"), "unexpected error"),
    ],
)
@pytest.mark.parametrize("sent", [False, True])
async def test_failures_return_to_model_before_submission_or_deliver_afterwards(
    service, event, error, expected, finish_turn, sent
):
    """Background errors reach the user without leaking exception details or retrying."""

    async def fail(*args, on_request_sent, **kwargs):
        if sent:
            on_request_sent()
        raise error

    service.api.generate = AsyncMock(side_effect=fail)
    result = await service.generate(event, ARGUMENTS)
    await finish_turn()
    service.api.generate.assert_awaited_once()
    if sent:
        assert "Image generation request submitted" in result
        service.context.send_message.assert_awaited_once()
        origin, message = service.context.send_message.call_args.args
        assert origin == event.unified_msg_origin
        assert "Do not automatically retry" not in message
        service.context.llm_generate.assert_awaited_once()
        assert expected in service.context.llm_generate.call_args.kwargs["prompt"]
        assert "tools" not in service.context.llm_generate.call_args.kwargs
        message = service.context.llm_generate.call_args.kwargs["prompt"]
    else:
        service.context.send_message.assert_not_awaited()
        assert "Image generation request submitted" not in result
        message = result
    assert expected in message
    assert "private" not in message
    if isinstance(error, httpx.RequestError):
        assert "Do not automatically retry" in message


@pytest.mark.asyncio
async def test_invalid_arguments_do_not_enqueue(service, event):
    """Invalid arguments are returned synchronously so the agent can correct them."""
    result = await service.generate(event, {"prompt": "", "description": "A cat"})
    assert "Invalid image arguments" in result
    assert not service._tasks
    service.context.send_message.assert_not_awaited()
    assert not service.output_dir.exists()
    assert not service.requests_dir.exists()


@pytest.mark.asyncio
async def test_archive_failure_does_not_enqueue(service, event, monkeypatch):
    """The tool must not acknowledge a queued job when its request cannot be saved."""
    monkeypatch.setattr(
        service.api, "create_request", Mock(side_effect=OSError("private path"))
    )
    service.api.generate = AsyncMock()
    result = await service.generate(event, ARGUMENTS)
    assert "No image task was queued" in result
    assert "private path" not in result
    assert not service._tasks
    service.api.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_request_history_uses_event_identity_and_survives_failure(
    service, event, monkeypatch, finish_turn
):
    monkeypatch.setattr(time, "time", lambda: 1_800_000_000)
    service.api.generate = AsyncMock(side_effect=httpx.ConnectError("offline"))
    result = await service.generate(event, ARGUMENTS)
    assert "connection failed" in result
    await finish_turn()
    archive = yaml.safe_load((service.output_dir / "0" / "request.yaml").read_text())
    assert archive["user_id"] == event.get_sender_id.return_value
    assert archive["request"] == ARGUMENTS
    assert (service.requests_dir / "123456.log").read_text() == "1800000000\t0\n"
    tool = service.context.add_llm_tools.call_args.args[1]
    assert "user_id" not in tool.parameters["properties"]


@pytest.mark.asyncio
async def test_sender_id_is_escaped_only_in_log_filename(service, event, finish_turn):
    from urllib.parse import quote

    user_id = "../../someone/\n用户%"
    event.get_sender_id.return_value = user_id
    service.api.generate = AsyncMock(return_value=[])
    await service.generate(event, ARGUMENTS)
    await finish_turn()
    assert {p.name for p in service.requests_dir.iterdir()} == {
        quote(user_id, safe="") + ".log",
        "total.log",
    }
    archive = yaml.safe_load((service.output_dir / "0" / "request.yaml").read_text())
    assert archive["user_id"] == user_id


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["webchat", "qq_official"])
@pytest.mark.parametrize("outcome", ["success", "api_error", "unexpected_error"])
async def test_ready_results_wait_for_final_reply(
    plugin, service, event, platform, outcome
):
    """Neither intermediate messages nor agent completion alone permit delivery."""
    event.get_platform_name.return_value = platform
    service.api.generate = AsyncMock(return_value=[])
    if outcome != "success":

        async def fail(*args, on_request_sent, **kwargs):
            on_request_sent()
            if outcome == "api_error":
                raise api_module.ImageGenerationError("HTTP 400")
            raise RuntimeError("private details")

        service.api.generate.side_effect = fail
    ready = asyncio.Event()
    original = service._generate

    async def prepare(*args):
        try:
            return await original(*args)
        finally:
            ready.set()

    service._generate = prepare
    assert "Image generation request submitted" in await service.generate(
        event,
        ARGUMENTS,
    )
    await asyncio.wait_for(ready.wait(), 1)
    await plugin.release_image_results(event)  # An intermediate tool reply.
    await asyncio.sleep(0)
    service.context.send_message.assert_not_awaited()
    event.send.assert_not_awaited()
    await plugin.mark_agent_finished(event, Mock(), Mock())
    await asyncio.sleep(0)
    service.context.send_message.assert_not_awaited()
    event.send.assert_not_awaited()
    # Another event in the same conversation must not release this turn.
    other = create_autospec(plugin_module.AstrMessageEvent, instance=True)
    other.unified_msg_origin = event.unified_msg_origin
    await plugin.mark_agent_finished(other, Mock(), Mock())
    await plugin.release_image_results(other)
    await asyncio.sleep(0)
    service.context.send_message.assert_not_awaited()
    event.send.assert_not_awaited()
    await plugin.release_image_results(event)
    await asyncio.wait_for(asyncio.gather(*service._tasks), 1)
    sender = event.send if platform == "qq_official" else service.context.send_message
    sender.assert_awaited_once()
    other_sender = (
        service.context.send_message if platform == "qq_official" else event.send
    )
    other_sender.assert_not_awaited()
    service.api.generate.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_pipeline_completion_releases_without_final_message(
    plugin, service, event, cancelled
):
    """Empty or interrupted turns can finish without a message-sent hook."""
    queued = asyncio.Event()
    finish = asyncio.Event()
    service.api.generate = AsyncMock(return_value=[])

    async def pipeline():
        await plugin.add_image_hint(event, ProviderRequest())
        assert "Image generation request submitted" in await service.generate(
            event,
            ARGUMENTS,
        )
        queued.set()
        await finish.wait()

    owner = asyncio.create_task(pipeline())
    await asyncio.wait_for(queued.wait(), 1)
    service.context.send_message.assert_not_awaited()
    if cancelled:
        owner.cancel()
    else:
        finish.set()
    await asyncio.gather(owner, return_exceptions=True)
    await asyncio.wait_for(asyncio.gather(*service._tasks), 1)
    service.context.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_unload_cancels_result_waiting_for_reply(plugin, service, event):
    service.api.generate = AsyncMock(return_value=[])
    assert "Image generation request submitted" in await service.generate(
        event,
        ARGUMENTS,
    )
    await asyncio.sleep(0)
    await asyncio.wait_for(plugin.terminate(), 1)
    assert not service._tasks
    service.context.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_multiple_images_in_one_turn_use_distinct_lossless_archives(
    service, event, image_bytes, mock_http, finish_turn
):
    """Deliver each queued image from its own archive after the agent finishes."""
    sources = [
        image_bytes("PNG", color=color, mode="RGBA")
        for color in [(255, 0, 0, 255), (0, 255, 0, 100), (0, 0, 255, 0)]
    ]
    submitted = []

    def respond(request):
        if request.method == "POST":
            index = len(submitted)
            submitted.append(json.loads(request.content))
            return httpx.Response(
                200, json={"data": [{"url": f"https://cdn.invalid/{index}"}]}
            )
        return httpx.Response(200, content=sources[int(request.url.path[1:])])

    mock_http(service.api, respond)
    for index in range(len(sources)):
        result = await asyncio.wait_for(
            service.generate(event, {**ARGUMENTS, "description": f"Image {index}"}), 5
        )
        assert "Image generation request submitted" in result
    service.context.send_message.assert_not_awaited()
    await finish_turn()
    assert len(submitted) == len(sources)
    deliveries = service.context.send_message.call_args_list
    assert len(deliveries) == len(sources)
    for index, raw in enumerate(sources):
        directory = service.output_dir / str(index)
        path = directory / "0.webp"
        assert {p.name for p in directory.iterdir()} == {
            "request.yaml",
            "response.yaml",
            "0.webp",
        }
        with (
            PillowImage.open(path) as result,
            PillowImage.open(io.BytesIO(raw)) as source,
        ):
            assert result.convert("RGBA").tobytes() == source.convert("RGBA").tobytes()
        expected = Image.fromFileSystem(str(path)).file
        assert any(
            call.args[1][0].text == f"Image {index}"
            and call.args[1][1].file == expected
            for call in deliveries
        )
