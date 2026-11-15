"""Exercise Seedream's HTTP contract, reference limits and PNG downloads."""

import base64
import json

import httpx
import jsonschema
import pytest
import yaml
from data.plugins.astrbot_plugin_seedream.src import generate as api_module
from data.plugins.astrbot_plugin_seedream.src.generate import (
    DEFAULT_ENDPOINT,
    INPUT_SCHEMA,
    REQUEST_SCHEMA,
    ImageGenerationError,
    SeedreamAPI,
)
from PIL import Image

ARGUMENTS = {"prompt": "  A cat  ", "description": "A cat"}


@pytest.mark.parametrize("background", [None, "auto", "transparent", "opaque"])
def test_fixed_payload_and_defaults(api, background):
    arguments = {**ARGUMENTS, "background": background}
    body, references = api.prepare(arguments)
    assert body == {
        "model": "doubao-seedream-5-5-261028",
        "prompt": "A cat",
        "layer_decomposition": False,
        "output_format": "png",
        "response_format": "url",
        "watermark": False,
        "background": background or "auto",
    }
    assert not references
    assert arguments["prompt"] == "  A cat  "
    assert api.prepare(ARGUMENTS)[0]["background"] == "auto"


@pytest.mark.parametrize(
    "options",
    [
        {"size": "2K"},
        {"number_of_images": 1},
        {"model": "another"},
        {"quality": "high"},
        {"web_search": False},
        {"previous_response_id": "id"},
        {"watermark": True},
        {"layer_decomposition": True},
        {"output_format": "jpeg"},
        {"background": "invalid"},
        {"background": 1},
        {"prompt": " "},
        {"description": " "},
        {"prompt": "x" * 32001},
        {"description": "x" * 1001},
        {"reference_images": None},
        {"reference_images": [True]},
        {"reference_images": [-1]},
        {"reference_images": [0, 0]},
        {"reference_images": list(range(11))},
        {"reference_images": ["/tmp/a.gif"]},
        {"reference_images": ["/tmp/a.png", "/tmp/a.png"]},
    ],
)
def test_invalid_arguments(api, options):
    with pytest.raises(ImageGenerationError):
        api.prepare({**ARGUMENTS, **options})


@pytest.mark.parametrize(
    "format_name,mime",
    [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")],
)
def test_reference_actual_mime_and_bytes(image_bytes, format_name, mime):
    encoded = base64.b64encode(image_bytes(format_name)).decode()
    assert SeedreamAPI.reference_payload([encoded]) == [f"data:{mime};base64,{encoded}"]


@pytest.mark.parametrize(
    "size,valid",
    [
        ((15, 15), True),
        ((14, 15), False),
        ((16, 256), True),
        ((256, 16), True),
        ((16, 257), False),
        ((257, 16), False),
        ((6000, 6000), True),
        ((6001, 6000), False),
    ],
)
def test_reference_dimensions(image_bytes, size, valid):
    encoded = base64.b64encode(
        image_bytes("PNG", size=size, mode="1", color=1)
    ).decode()
    if valid:
        assert SeedreamAPI.reference_payload([encoded])
    else:
        with pytest.raises(ImageGenerationError, match="reference_images"):
            SeedreamAPI.reference_payload([encoded])


def test_reference_size_boundary_and_corruption(image_bytes, monkeypatch):
    raw = image_bytes("PNG")
    monkeypatch.setattr(api_module, "MAX_REFERENCE_BYTES", len(raw))
    assert SeedreamAPI.reference_payload([base64.b64encode(raw).decode()])
    for invalid in (raw + b"x", raw[:30], image_bytes("GIF"), b"not an image"):
        with pytest.raises(ImageGenerationError):
            SeedreamAPI.reference_payload([base64.b64encode(invalid).decode()])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint", [DEFAULT_ENDPOINT, "https://example.invalid/custom/generate"]
)
async def test_http_contract_download_redirect_and_lossless_webp(
    api, tmp_path, png, webp, mock_http, endpoint
):
    api.config["base_url"] = endpoint
    requests = []
    submitted = []
    encoded = base64.b64encode(webp).decode()

    def handler(request):
        requests.append(request)
        if request.method == "POST":
            assert str(request.url) == endpoint
            assert request.headers["Authorization"] == "Bearer test-key"
            body = json.loads(request.content)
            assert body["image"] == [f"data:image/webp;base64,{encoded}"]
            assert "description" not in body and "size" not in body
            return httpx.Response(
                200,
                json={
                    "data": [{"url": "https://images.invalid/start?signature=private"}]
                },
            )
        archived = yaml.safe_load((tmp_path / "images/0/response.yaml").read_text())
        assert (
            archived["data"][0]["url"]
            == "https://images.invalid/start?signature=private"
        )
        assert "Authorization" not in request.headers
        assert submitted == [True]
        if request.url.path == "/start":
            return httpx.Response(
                302, headers={"Location": "https://cdn.invalid/final.png"}
            )
        return httpx.Response(
            200, content=png, headers={"Content-Type": "application/octet-stream"}
        )

    mock_http(api, handler)
    arguments = {**ARGUMENTS, "background": None, "reference_images": [0]}
    body, _ = api.prepare(arguments)
    paths = await api.generate(
        body,
        [encoded],
        tmp_path / "images",
        arguments,
        on_request_sent=lambda: submitted.append(True),
    )
    assert paths[0].name == "0.webp"
    assert not (paths[0].parent / "0.png").exists()
    with Image.open(paths[0]) as image:
        assert image.format == "WEBP"
        assert image.mode == "RGBA"
        assert image.getpixel((0, 0))[3] == 100
    archive = yaml.safe_load(paths[0].with_name("request.yaml").read_text())
    jsonschema.validate(archive, REQUEST_SCHEMA)
    assert archive["request"] == arguments
    assert len(requests) == 3
    assert "test-key" not in paths[0].with_name("request.yaml").read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500, 302])
async def test_api_errors_do_not_retry_or_leak(api, tmp_path, mock_http, status):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            status,
            json={"error": "private credentials"},
            headers={"Location": "https://other.invalid"},
        )

    mock_http(api, handler)
    with pytest.raises(ImageGenerationError, match=f"HTTP {status}") as exc:
        await api.generate(api.prepare(ARGUMENTS)[0], [], tmp_path, ARGUMENTS)
    assert "private" not in str(exc.value)
    assert len(requests) == 1
    assert (tmp_path / "0/request.yaml").exists()
    assert not (tmp_path / "0/0.webp").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        None,
        {},
        [],
        {"data": []},
        {"data": [None]},
        {"data": [{}, {}]},
        {"data": [{"b64_json": "abc"}]},
        {"data": [{"url": "/relative"}]},
        {"data": [{"url": "file:///tmp/image.png"}]},
        {"data": [{"url": "https://user:secret@cdn.invalid/image"}]},
        {
            "data": [
                {"url": "https://cdn.invalid/image", "error": {"message": "private"}}
            ]
        },
        {
            "error": {"message": "private"},
            "data": [{"url": "https://cdn.invalid/image"}],
        },
    ],
)
async def test_invalid_responses_do_not_download(api, tmp_path, mock_http, result):
    requests = []

    def handler(request):
        requests.append(request)
        return (
            httpx.Response(200, json=result)
            if result is not None
            else httpx.Response(200, content=b"invalid JSON")
        )

    mock_http(api, handler)
    with pytest.raises(ImageGenerationError) as exc:
        await api.generate(api.prepare(ARGUMENTS)[0], [], tmp_path, ARGUMENTS)
    assert "private" not in str(exc.value)
    assert len(requests) == 1
    expected = (
        {"request.yaml", "response.yaml"} if result is not None else {"request.yaml"}
    )
    assert {p.name for p in (tmp_path / "0").iterdir()} == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "status", "html", "truncated"])
async def test_download_failures_preserve_request_without_retry(
    api, tmp_path, png, mock_http, failure
):
    requests = []

    def handler(request):
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(
                200, json={"data": [{"url": "https://cdn.invalid/image"}]}
            )
        if failure == "timeout":
            raise httpx.ReadTimeout("private signed URL")
        if failure == "status":
            return httpx.Response(403)
        return httpx.Response(
            200,
            content={
                "html": b"<html>error</html>",
                "truncated": png[:30],
            }[failure],
        )

    mock_http(api, handler)
    expected_error = (
        ImageGenerationError if failure in {"timeout", "status"} else OSError
    )
    with pytest.raises(expected_error) as exc:
        await api.generate(api.prepare(ARGUMENTS)[0], [], tmp_path, ARGUMENTS)
    assert "private" not in str(exc.value)
    assert len(requests) == 2
    assert {p.name for p in (tmp_path / "0").iterdir()} == {
        "request.yaml",
        "response.yaml",
    }
    assert (
        yaml.safe_load((tmp_path / "0/response.yaml").read_text())["data"][0]["url"]
        == "https://cdn.invalid/image"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key,value",
    [
        ("base_url", "file:///tmp/api"),
        ("base_url", "https://user:secret@example.invalid"),
        ("base_url", "https://example.invalid/api?token=secret"),
        ("base_url", "not a URL"),
        ("timeout_seconds", 0),
        ("timeout_seconds", -1),
        ("timeout_seconds", "nan"),
        ("timeout_seconds", "inf"),
    ],
)
async def test_invalid_transport_config(api, tmp_path, key, value):
    api.config[key] = value
    with pytest.raises(ImageGenerationError, match=key) as exc:
        await api.generate(api.prepare(ARGUMENTS)[0], [], tmp_path, ARGUMENTS)
    assert "secret" not in str(exc.value)
    assert api.client is None


def test_key_precedence_and_no_development_file_fallback(api, monkeypatch):
    monkeypatch.setenv("VOLCENGINE_API_KEY", " env-key ")
    assert api.require_api_key() == "test-key"
    api.config["api_key"] = " "
    assert api.require_api_key() == "env-key"
    monkeypatch.delenv("VOLCENGINE_API_KEY")
    with pytest.raises(ImageGenerationError, match="VOLCENGINE_API_KEY"):
        api.require_api_key()


def test_schema_and_documented_examples():
    import re
    from pathlib import Path

    jsonschema.Draft202012Validator.check_schema(REQUEST_SCHEMA)
    guide = (Path(__file__).parents[1] / "src/instructions.md").read_text()
    for example in re.findall(r"```json\s*(.*?)\s*```", guide, re.DOTALL):
        jsonschema.validate(json.loads(example), INPUT_SCHEMA)


def test_pillow_decode_failure_leaves_no_output(png, tmp_path):
    corrupted = bytearray(png)
    corrupted[29] ^= 1
    with pytest.raises(OSError):
        SeedreamAPI.save_image(bytes(corrupted), tmp_path)
    assert not list(tmp_path.iterdir())
