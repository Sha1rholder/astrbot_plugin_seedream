"""Verify atomic archive writes and bounded storage cleanup."""

import io
from pathlib import Path

import httpx
import pytest
from data.plugins.astrbot_plugin_seedream.src.generate import (
    ImageGenerationError,
    SeedreamAPI,
)
from PIL import Image

ARGUMENTS = {"prompt": "cat", "description": "A cat"}


@pytest.mark.parametrize("stage", ["request", "image"])
def test_atomic_storage_failure(api, tmp_path, png, monkeypatch, stage):
    output = tmp_path / "images"
    directory = api.create_request(output, ARGUMENTS)
    SeedreamAPI.save_image(png, directory)
    original = (directory / "request.yaml").read_bytes()
    original_image = (directory / "0.webp").read_bytes()
    if stage == "request":
        original_write = Path.write_text

        def fail_write(path, *args, **kwargs):
            if path.name == "request.yaml":
                raise OSError("disk full")
            return original_write(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", fail_write)
        with pytest.raises(OSError):
            api.create_request(output, ARGUMENTS)
        assert {p.name for p in output.iterdir()} == {"0"}
    else:
        original_replace = Path.replace

        def fail_replace(path, *args, **kwargs):
            if path.name == "0.webp":
                raise OSError("disk full")
            return original_replace(path, *args, **kwargs)

        monkeypatch.setattr(Path, "replace", fail_replace)
        with pytest.raises(OSError):
            SeedreamAPI.save_image(png, directory)
    assert {p.name for p in directory.iterdir()} == {"request.yaml", "0.webp"}
    assert (directory / "request.yaml").read_bytes() == original
    assert (directory / "0.webp").read_bytes() == original_image


@pytest.mark.parametrize("limit", [0, True, "10"])
def test_invalid_storage_limit(api, limit):
    api.config["max_storage_gb"] = limit
    with pytest.raises(ImageGenerationError, match="max_storage_gb"):
        api.storage_limit()


def test_cleanup_ignores_unrelated_files_and_symlinks(tmp_path):
    output = tmp_path / "images"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("keep")
    (output / "0").symlink_to(outside, target_is_directory=True)
    for name in ["1", "2", "02", "old-uuid"]:
        (output / name).mkdir()
    (output / "old.webp").write_text("keep")
    (output / "1" / "large").write_bytes(b"x" * 100)
    SeedreamAPI.prune_requests(output, 10, 2)
    assert not (output / "1").exists()
    assert (outside / "keep").read_text() == "keep"
    assert {p.name for p in output.iterdir()} == {
        "0",
        "2",
        "02",
        "old-uuid",
        "old.webp",
    }


def test_next_request_id_scans_only_numeric_directories(tmp_path):
    assert SeedreamAPI.next_request_id(tmp_path / "missing") == 0
    for name in ["2", "9", "10", "notes"]:
        (tmp_path / name).mkdir()
    (tmp_path / "999").write_text("not a directory")
    (tmp_path / "1000").symlink_to(tmp_path / "2", target_is_directory=True)
    assert SeedreamAPI.next_request_id(tmp_path) == 11


@pytest.mark.parametrize(
    "limit,remaining", [(100, {"0", "2", "10"}), (99, {"2", "10"}), (60, {"10"})]
)
def test_capacity_cleanup_order_and_boundary(tmp_path, limit, remaining):
    for name, size in [("10", 30), ("2", 30), ("0", 40)]:
        directory = tmp_path / name / "nested"
        directory.mkdir(parents=True)
        (directory / "image.webp").write_bytes(b"x" * size)
    SeedreamAPI.prune_requests(tmp_path, limit, 11)
    assert {p.name for p in tmp_path.iterdir()} == remaining


@pytest.mark.asyncio
async def test_cleanup_precedes_http_and_keeps_current_and_queued(
    api, tmp_path, monkeypatch, mock_http
):
    arguments = ARGUMENTS
    output = tmp_path / "images"
    old = api.create_request(output, arguments)
    (old / "large.webp").write_bytes(b"x" * 10000)
    current = api.create_request(output, arguments)
    queued = api.create_request(output, arguments)
    monkeypatch.setattr(
        api, "storage_limit", lambda: SeedreamAPI.storage_size(current) * 2 + 1
    )
    calls = []

    def respond(request):
        assert not old.exists()
        assert (current / "request.yaml").exists()
        assert (queued / "request.yaml").exists()
        calls.append(request)
        return httpx.Response(500)

    mock_http(api, respond)
    payload, _ = api.prepare(arguments)
    with pytest.raises(ImageGenerationError, match="HTTP 500"):
        await api.generate(payload, [], output, arguments, current)
    assert len(calls) == 1
    monkeypatch.setattr(api, "storage_limit", lambda: 1)
    with pytest.raises(ImageGenerationError, match="No API request"):
        await api.generate(payload, [], output, arguments, current)
    assert len(calls) == 1
    assert current.exists() and queued.exists()


@pytest.mark.parametrize(
    "config,expected",
    [({}, 10_000_000_000), ({"max_storage_gb": 25}, 25_000_000_000)],
)
def test_storage_limit_in_bytes(api, config, expected):
    api.config.update(config)
    assert api.storage_limit() == expected


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "P"])
def test_lossless_webp_preserves_pixels_and_removes_png(tmp_path, mode):
    """Preserve colors and alpha, including RGB values under transparent pixels."""
    source = Image.new("RGBA", (16, 16))
    source.putdata([(i, 255 - i, i // 2, i) for i in range(256)])
    source = source.convert(mode)
    encoded = io.BytesIO()
    source.save(encoded, "PNG")
    original = tmp_path / "0.png"
    original.write_bytes(encoded.getvalue())
    path = SeedreamAPI.save_image(original.read_bytes(), tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {"0.webp"}
    with Image.open(path) as result:
        assert result.format == "WEBP"
        assert result.size == source.size
        assert result.convert("RGBA").tobytes() == source.convert("RGBA").tobytes()


@pytest.mark.parametrize("stage", ["encode", "publish"])
def test_conversion_failure_preserves_png(tmp_path, png, monkeypatch, stage):
    """Do not remove the source or leave partial output when encoding fails."""
    original = tmp_path / "0.png"
    original.write_bytes(png)

    def fail_save(*args, **kwargs):
        raise OSError("disk full")

    if stage == "encode":
        monkeypatch.setattr(Image.Image, "save", fail_save)
    else:
        monkeypatch.setattr(Path, "replace", fail_save)
    with pytest.raises(OSError, match="disk full"):
        SeedreamAPI.save_image(png, tmp_path)
    assert original.read_bytes() == png
    assert {p.name for p in tmp_path.iterdir()} == {"0.png"}
