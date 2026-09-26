"""Tests for `anometa.data.download`: fetch, verify and extract AD2 archives.

No test touches the network: `urllib.request.urlopen` and `download` are
monkeypatched wherever a real transfer would otherwise happen.
"""

import gzip
import hashlib
import io
import random
import tarfile
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path

import pytest

import anometa.cli as cli
from anometa.cli import main
from anometa.config import Paths, Scenario
from anometa.data import download as download_module
from anometa.data.download import (
    ChecksumMismatchError,
    download,
    extract,
    fetch_scenario,
    load_sources,
    read_sha256sums,
    verify_or_pin,
)


class _FakeResponse(io.BytesIO):
    """Stand-in for `http.client.HTTPResponse` returned by `urlopen`."""

    def __init__(self, body: bytes, status: int, headers: dict[str, str]) -> None:
        """Build a fake response with a body, status and header set.

        Args:
            body: Bytes the response yields on `.read()`.
            status: HTTP status code (206 resumed, 200 fresh/restarted).
            headers: Header names and values exposed via `.headers.get`.
        """
        super().__init__(body)
        self.status = status
        self.headers = Message()
        for key, value in headers.items():
            self.headers[key] = value


def _patch_urlopen(
    monkeypatch: pytest.MonkeyPatch, reply: _FakeResponse | Exception
) -> list[urllib.request.Request]:
    """Make `urllib.request.urlopen` return or raise `reply`, recording requests.

    Args:
        monkeypatch: Pytest's monkeypatch fixture.
        reply: The response to return, or the exception to raise.

    Returns:
        The list each request is appended to, in call order.
    """
    requests: list[urllib.request.Request] = []

    def fake_urlopen(req: urllib.request.Request, timeout: float | None = None) -> _FakeResponse:
        """Record `req`, then return or raise `reply`."""
        requests.append(req)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return requests


def _write_archive(dest: Path, image: bytes = b"png") -> Path:
    """Write a minimal AD2-shaped `vial` archive to `dest`.

    Args:
        dest: Archive path; parent directories are created as needed.
        image: Content of the archive's single image file.

    Returns:
        `dest`. The archive holds `license.txt` and
        `vial/train/good/000_regular.png` at its top level, and its bytes
        depend only on `image`.
    """
    members = {"license.txt": b"license", "vial/train/good/000_regular.png": image}
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(gzip.compress(tar_bytes.getvalue(), mtime=0))
    return dest


def _fake_download(url: str, dest: Path) -> Path:
    """Stand in for `download`: write a minimal archive instead of fetching `url`.

    Args:
        url: Ignored.
        dest: Archive path to write.

    Returns:
        `dest`.
    """
    return _write_archive(dest)


def _vial_paths(tmp_path: Path) -> Paths:
    """Build `Paths` under `tmp_path` with a `vial` sources config and empty sums.

    Args:
        tmp_path: Pytest's per-test temporary directory.

    Returns:
        `Paths` whose `configs / "data"` holds `ad2.yaml` and an empty
        `sha256sums.txt`.
    """
    paths = Paths(data=tmp_path / "data/ad2", configs=tmp_path / "configs")
    configs_data = paths.configs / "data"
    configs_data.mkdir(parents=True)
    (configs_data / "ad2.yaml").write_text(
        "base: https://example.invalid/shares/\narchives:\n  vial: vial.tar.gz\n"
    )
    (configs_data / "sha256sums.txt").write_text("")
    return paths


def _record_fetches(monkeypatch: pytest.MonkeyPatch) -> list[Scenario]:
    """Replace `cli.fetch_scenario` with a recorder.

    Args:
        monkeypatch: Pytest's monkeypatch fixture.

    Returns:
        The list each requested scenario is appended to, in call order.
    """
    calls: list[Scenario] = []

    def fake_fetch(scenario: Scenario, paths: Paths, keep_archive: bool = False) -> str:
        """Record `scenario` instead of fetching it."""
        calls.append(scenario)
        return "digest"

    monkeypatch.setattr(cli, "fetch_scenario", fake_fetch)
    return calls


def test_load_sources_builds_full_urls(tmp_path: Path) -> None:
    """`base` and a relative archive path join into one full URL per scenario."""
    p = tmp_path / "ad2.yaml"
    p.write_text("base: https://example.invalid/shares/\narchives:\n  vial: 1/2/vial.tar.gz\n")
    sources = load_sources(p)
    assert sources == {Scenario.VIAL: "https://example.invalid/shares/1/2/vial.tar.gz"}


def test_read_sha256sums_parses_multiple_lines(tmp_path: Path) -> None:
    """Each `<digest>  <filename>` line becomes one dict entry."""
    p = tmp_path / "sums.txt"
    p.write_text(f"{'a' * 64}  can.tar.gz\n{'b' * 64}  vial.tar.gz\n")
    assert read_sha256sums(p) == {"can.tar.gz": "a" * 64, "vial.tar.gz": "b" * 64}


def test_verify_or_pin_pins_missing(tmp_path: Path) -> None:
    """An archive without a pinned hash gets its sha256 appended."""
    a = tmp_path / "can.tar.gz"
    a.write_bytes(b"abc")
    sums = tmp_path / "sums.txt"
    sums.write_text("")
    digest = verify_or_pin(a, sums)
    assert digest == hashlib.sha256(b"abc").hexdigest()
    assert read_sha256sums(sums) == {"can.tar.gz": digest}


def test_verify_or_pin_mismatch_raises_and_keeps_file(tmp_path: Path) -> None:
    """A hash mismatch raises and leaves the archive in place."""
    a = tmp_path / "vial.tar.gz"
    a.write_bytes(b"abc")
    sums = tmp_path / "sums.txt"
    sums.write_text(f"{'0' * 64}  vial.tar.gz\n")
    with pytest.raises(ChecksumMismatchError):
        verify_or_pin(a, sums)
    assert a.exists()


def test_download_resumes_with_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A partial file is resumed with a Range header, not restarted."""
    dest = tmp_path / "x.tar.gz"
    dest.write_bytes(b"0123456789")
    reply = _FakeResponse(b"abcdef", status=206, headers={"Content-Range": "bytes 10-15/16"})
    requests = _patch_urlopen(monkeypatch, reply)
    download("https://example.invalid/x.tar.gz", dest)
    assert requests[0].get_header("Range") == "bytes=10-"
    assert dest.read_bytes() == b"0123456789abcdef"


def test_download_restarts_when_server_ignores_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 200 response to a range request truncates the partial file and restarts."""
    dest = tmp_path / "x.tar.gz"
    dest.write_bytes(b"stale-partial")
    _patch_urlopen(
        monkeypatch, _FakeResponse(b"full-body", status=200, headers={"Content-Length": "9"})
    )
    download("https://example.invalid/x.tar.gz", dest)
    assert dest.read_bytes() == b"full-body"


def test_download_short_body_raises_and_keeps_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A body that ends before Content-Length raises and keeps the partial file to resume."""
    dest = tmp_path / "x.tar.gz"
    _patch_urlopen(monkeypatch, _FakeResponse(b"part", status=200, headers={"Content-Length": "9"}))
    with pytest.raises(ConnectionError, match="got 4 of 9 bytes"):
        download("https://example.invalid/x.tar.gz", dest)
    assert dest.read_bytes() == b"part"


def test_download_416_on_resume_means_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 416 answer to a range request returns the file already on disk unchanged."""
    dest = tmp_path / "x.tar.gz"
    dest.write_bytes(b"complete")
    url = "https://example.invalid/x.tar.gz"
    error = urllib.error.HTTPError(url, 416, "Range Not Satisfiable", Message(), None)
    requests = _patch_urlopen(monkeypatch, error)
    assert download(url, dest) == dest
    assert requests[0].get_header("Range") == "bytes=8-"
    assert dest.read_bytes() == b"complete"


def test_extract_creates_scenario_dir(tmp_path: Path) -> None:
    """Extracting a scenario archive yields root/<scenario>/train/good and top-level files."""
    arc = _write_archive(tmp_path / "vial.tar.gz")
    extract(arc, tmp_path / "root")
    assert (tmp_path / "root/vial/train/good/000_regular.png").exists()
    assert sorted(p.name for p in (tmp_path / "root").iterdir()) == ["license.txt", "vial"]


def test_extract_replaces_existing_scenario_only_on_success(tmp_path: Path) -> None:
    """A failed re-extract keeps the old scenario folder; a successful one replaces it."""
    root = tmp_path / "root"
    old = root / "vial/old.png"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old")
    image = random.Random(0).randbytes(1 << 16)  # incompressible, so truncation cuts the image
    good = _write_archive(tmp_path / "vial.tar.gz", image=image)
    truncated = tmp_path / "truncated.tar.gz"
    truncated.write_bytes(good.read_bytes()[: good.stat().st_size // 2])

    with pytest.raises(EOFError):
        extract(truncated, root)
    assert sorted(p.name for p in root.iterdir()) == ["vial"]
    assert old.exists()

    extract(good, root)
    assert not old.exists()
    assert (root / "vial/train/good/000_regular.png").read_bytes() == image


def test_fetch_scenario_skips_when_extracted_and_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An already-extracted scenario with a pinned archive hash isn't re-fetched."""
    paths = Paths(data=tmp_path / "data/ad2", configs=tmp_path / "configs")
    (paths.data / "vial").mkdir(parents=True)
    configs_data = paths.configs / "data"
    configs_data.mkdir(parents=True)
    (configs_data / "sha256sums.txt").write_text(f"{'a' * 64}  vial.tar.gz\n")

    def fail_download(url: str, dest: Path) -> Path:
        """Fail the test if a download is attempted."""
        raise AssertionError("download should not be called when already pinned")

    monkeypatch.setattr(download_module, "download", fail_download)

    assert fetch_scenario(Scenario.VIAL, paths) == "a" * 64


def test_fetch_scenario_downloads_verifies_and_extracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first fetch downloads, pins the checksum, extracts, and drops the archive."""
    paths = _vial_paths(tmp_path)
    monkeypatch.setattr(download_module, "download", _fake_download)

    digest = fetch_scenario(Scenario.VIAL, paths)

    assert (paths.data / "vial/train/good/000_regular.png").exists()
    assert read_sha256sums(paths.configs / "data/sha256sums.txt") == {"vial.tar.gz": digest}
    assert not (tmp_path / "data/archives/vial.tar.gz").exists()


def test_fetch_scenario_keep_archive_leaves_it_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`keep_archive=True` leaves the downloaded archive in place."""
    paths = _vial_paths(tmp_path)
    monkeypatch.setattr(download_module, "download", _fake_download)

    fetch_scenario(Scenario.VIAL, paths, keep_archive=True)

    assert (tmp_path / "data/archives/vial.tar.gz").exists()


def test_fetch_scenario_failed_extraction_is_redone_on_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An extraction failing after unpacking leaves no scenario folder; a rerun redoes it."""
    paths = _vial_paths(tmp_path)
    monkeypatch.setattr(download_module, "download", _fake_download)
    real_extractall = tarfile.TarFile.extractall

    def extractall_then_fail(self, *args, **kwargs):
        """Unpack the whole archive, then fail as a full disk would."""
        real_extractall(self, *args, **kwargs)
        raise OSError("No space left on device")

    monkeypatch.setattr(tarfile.TarFile, "extractall", extractall_then_fail)
    with pytest.raises(OSError, match="No space left"):
        fetch_scenario(Scenario.VIAL, paths)
    assert list(paths.data.iterdir()) == []

    monkeypatch.setattr(tarfile.TarFile, "extractall", real_extractall)
    fetch_scenario(Scenario.VIAL, paths)
    assert (paths.data / "vial/train/good/000_regular.png").exists()


def test_cli_download_dispatches_requested_scenario(monkeypatch: pytest.MonkeyPatch) -> None:
    """`anometa download --scenario vial` fetches only the requested scenario."""
    calls = _record_fetches(monkeypatch)
    assert main(["download", "--scenario", "vial"]) == 0
    assert calls == [Scenario.VIAL]


def test_cli_download_defaults_to_all_scenarios(monkeypatch: pytest.MonkeyPatch) -> None:
    """`anometa download` with no `--scenario` fetches every scenario."""
    calls = _record_fetches(monkeypatch)
    assert main(["download"]) == 0
    assert calls == list(Scenario)
