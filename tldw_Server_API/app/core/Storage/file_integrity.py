"""Synchronous generated-file integrity checks for off-loop replay validation."""

import hashlib
import stat
from pathlib import Path


def generated_file_bytes_match(
    path: Path, *, expected_size: int, checksum: str | None,
) -> bool:
    """Validate nonempty bytes and an optional SHA-256 using bounded reads.

    Legacy records without a checksum retain size-only validation. Missing files
    return False; other filesystem errors propagate for retry classification.
    Callers must enforce path containment and offload this function from asyncio.
    """
    if expected_size <= 0:
        return False
    if checksum not in (None, "") and (
        not isinstance(checksum, str) or len(checksum) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in checksum)
    ):
        return False
    try:
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_size:
            return False
        if not checksum:
            return True
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(chunk)
                if size > expected_size:
                    return False
                digest.update(chunk)
        return size == expected_size and digest.hexdigest() == checksum.lower()
    except (FileNotFoundError, NotADirectoryError):
        return False
