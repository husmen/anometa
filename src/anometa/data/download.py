"""Download, verify and extract MVTec AD 2 scenario archives.

`fetch_scenario` is the entry point the CLI and later tasks call: it skips
work already done, resumes interrupted downloads, pins or checks a sha256
digest per archive, and extracts into the run's data directory. Transfer,
hashing and extraction are stdlib-only (`urllib.request`, `hashlib`,
`tarfile`) so a fresh checkout can fetch data with no extra dependency.
"""

import hashlib
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from http.client import HTTPResponse
from pathlib import Path

import yaml
from pydantic import BaseModel

from anometa.config import Paths, Scenario

_CHUNK_SIZE = 1024 * 1024  # 1 MiB
_TIMEOUT_S = 30.0
_PROGRESS_STEP = 5


class ChecksumMismatchError(Exception):
    """Raised when an archive's sha256 digest doesn't match its pinned entry."""


class _Sources(BaseModel):
    """Schema of the AD2 sources config (`configs/data/ad2.yaml`)."""

    base: str
    archives: dict[Scenario, str]


def load_sources(path: Path) -> dict[Scenario, str]:
    """Load scenario download URLs from an AD2 sources config.

    Args:
        path: YAML file with a `base` URL and an `archives` mapping from
            scenario name to a path relative to `base`.

    Returns:
        Each `Scenario` mapped to its full download URL.
    """
    sources = _Sources.model_validate(yaml.safe_load(path.read_text()))
    return {name: urllib.parse.urljoin(sources.base, rel) for name, rel in sources.archives.items()}


def read_sha256sums(path: Path) -> dict[str, str]:
    """Parse a `sha256sum`-format checksum file.

    Args:
        path: Text file with one `<hex digest>  <filename>` line per archive.

    Returns:
        Each archive filename mapped to its hex sha256 digest.
    """
    sums: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        digest, filename = line.split(maxsplit=1)
        sums[filename] = digest
    return sums


def _total_size(resp: HTTPResponse, downloaded: int) -> int | None:
    """Resolve the expected final file size from response headers.

    Args:
        resp: The open response for the current request.
        downloaded: Bytes already on disk before this response's body.

    Returns:
        The expected final size in bytes, or `None` when neither
        `Content-Range` nor `Content-Length` is present.
    """
    content_range = resp.headers.get("Content-Range")
    if content_range is not None:
        total = content_range.rpartition("/")[2]
        if total.isdigit():
            return int(total)
    content_length = resp.headers.get("Content-Length")
    if content_length is not None and content_length.isdigit():
        return downloaded + int(content_length)
    return None


def download(url: str, dest: Path) -> Path:
    """Download `url` to `dest`, resuming a partial file.

    Sends `Range: bytes=<existing size>-` when `dest` already holds bytes.
    A server that answers 200 instead of 206 has ignored the range, so the
    partial file is truncated and the download restarts from scratch.
    A 416 answer to a range request means `dest` already holds the whole
    file, so it is returned as is (`verify_or_pin` catches corruption).
    Streams the body in 1 MiB chunks and prints progress every 5% once the
    total size is known from `Content-Range` or `Content-Length`.

    Args:
        url: The archive URL to fetch.
        dest: The local file to write; resumed if it already exists.

    Returns:
        `dest`, once fully written.

    Raises:
        urllib.error.HTTPError: On an HTTP error status other than a 416
            answer to a range request.
        urllib.error.URLError: If the server can't be reached.
        ConnectionError: If the body ends before (or runs past) the known
            total size. The partial file is kept so a rerun resumes it.
    """
    existing = dest.stat().st_size if dest.exists() else 0
    request = urllib.request.Request(url)
    if existing:
        request.add_header("Range", f"bytes={existing}-")

    try:
        resp: HTTPResponse = urllib.request.urlopen(request, timeout=_TIMEOUT_S)
    except urllib.error.HTTPError as err:
        if existing and err.code == 416:
            return dest
        raise
    with resp:
        resuming = existing > 0 and resp.status == 206
        downloaded = existing if resuming else 0
        total = _total_size(resp, downloaded)
        last_reported = 0
        with dest.open("ab" if resuming else "wb") as f:
            while chunk := resp.read(_CHUNK_SIZE):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded * 100 // total
                    if percent - last_reported >= _PROGRESS_STEP:
                        print(f"{dest.name}: {percent}%")
                        last_reported = percent
    if total is not None and downloaded != total:
        raise ConnectionError(f"{dest.name}: got {downloaded} of {total} bytes; rerun to resume")
    return dest


def verify_or_pin(archive: Path, sums_path: Path) -> str:
    """Verify an archive's sha256 against its pinned entry, or pin it.

    Args:
        archive: The downloaded archive file to hash.
        sums_path: `sha256sum`-format file holding pinned digests.

    Returns:
        The archive's hex sha256 digest.

    Raises:
        ChecksumMismatchError: If `archive.name` is already pinned to a
            different digest. The archive is never deleted, either way.
    """
    with archive.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    pinned = read_sha256sums(sums_path)
    existing = pinned.get(archive.name)
    if existing is None:
        with sums_path.open("a") as f:
            f.write(f"{digest}  {archive.name}\n")
        return digest
    if existing != digest:
        raise ChecksumMismatchError(f"{archive.name}: expected {existing}, got {digest}")
    return digest


def extract(archive: Path, root: Path) -> None:
    """Extract an AD2 scenario archive atomically.

    The archive is extracted into a temporary directory under `root` (same
    filesystem), and its top-level entries are moved into `root` only once
    the whole archive has been read. The scenario folder moves last, so
    `root / <scenario>` existing means the extraction completed. An existing
    entry of the same name is replaced only at that point; if extraction
    fails, `root` is left untouched.

    Args:
        archive: `.tar.gz` archive holding `license.txt`, `readme.txt` and
            the scenario folder at its top level.
        root: Directory the scenario folder is extracted into; created if
            missing.
    """
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix=".extract-") as tmp:
        staging, replaced = Path(tmp, "new"), Path(tmp, "old")
        replaced.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(staging, filter="data")
        for entry in sorted(staging.iterdir(), key=Path.is_dir):
            target = root / entry.name
            if target.is_dir():
                target.rename(replaced / entry.name)
            entry.replace(target)


def fetch_scenario(scenario: Scenario, paths: Paths, keep_archive: bool = False) -> str:
    """Fetch, verify and extract one AD2 scenario, skipping if already done.

    Sources and pinned checksums are read from `paths.configs / "data"`
    (`ad2.yaml` and `sha256sums.txt`); the archive is downloaded to
    `paths.data.parent / "archives"` and extracted into `paths.data`.

    Args:
        scenario: The scenario to fetch.
        paths: Run paths.
        keep_archive: Keep the downloaded archive after extraction instead
            of deleting it.

    Returns:
        The archive's pinned or newly verified hex sha256 digest.
    """
    config_dir = paths.configs / "data"
    sums_path = config_dir / "sha256sums.txt"
    archive_name = f"{scenario}.tar.gz"
    scenario_dir = paths.data / scenario

    pinned = read_sha256sums(sums_path)
    if scenario_dir.exists() and archive_name in pinned:
        return pinned[archive_name]

    sources = load_sources(config_dir / "ad2.yaml")
    archives_dir = paths.data.parent / "archives"
    archives_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archives_dir / archive_name

    download(sources[scenario], archive_path)
    digest = verify_or_pin(archive_path, sums_path)
    extract(archive_path, paths.data)
    if not keep_archive:
        archive_path.unlink()
    return digest
