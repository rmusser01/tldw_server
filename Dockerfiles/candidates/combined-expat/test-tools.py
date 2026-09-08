"""Emit the reviewed test-only wheel cohort from uv.lock without resolving it.

This does not install anything or alter the application environment. The caller
must install these hashed wheels into a separate target directory with no deps.
"""

import argparse
import re
from pathlib import Path
from urllib.parse import urlsplit

import tomllib

VERSIONS = {
    "iniconfig": "2.3.0",
    "packaging": "26.3",
    "pluggy": "1.6.0",
    "pygments": "2.21.0",
    "pytest": "9.1.1",
    "pytest-asyncio": "1.4.0",
    "pytest-timeout": "2.4.0",
    "typing-extensions": "4.16.0",
}


def requirements(lock: dict) -> str:
    """Return exact universal wheel requirements; reject cohort/identity drift."""
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise ValueError("lock must contain a package array")
    selected = {name: [] for name in VERSIONS}
    for package in packages:
        if not isinstance(package, dict) or not isinstance(package.get("name"), str):
            raise ValueError("invalid package record")
        name = re.sub(r"[-_.]+", "-", package["name"]).lower()
        if name in selected:
            selected[name].append(package)

    lines = []
    for name, version in VERSIONS.items():
        records = selected[name]
        if len(records) != 1:
            raise ValueError(f"{name}: expected exactly one locked package")
        package = records[0]
        if package.get("version") != version or package.get("source") != {"registry": "https://pypi.org/simple"}:
            raise ValueError(f"{name}: reviewed version or registry changed")
        wheels = package.get("wheels")
        if not isinstance(wheels, list) or len(wheels) != 1:
            raise ValueError(f"{name}: expected exactly one universal wheel")
        wheel = wheels[0]
        if not isinstance(wheel, dict):
            raise ValueError(f"{name}: invalid wheel record")
        url, digest = wheel.get("url"), wheel.get("hash")
        if not isinstance(url, str) or any(ord(char) <= 32 or ord(char) >= 127 for char in url):
            raise ValueError(f"{name}: invalid wheel URL")
        parsed = urlsplit(url)
        filename = f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
        if (
            parsed.scheme != "https"
            or parsed.netloc != "files.pythonhosted.org"
            or parsed.query
            or parsed.fragment
            or not re.fullmatch(r"/packages/[A-Za-z0-9/_-]+/" + re.escape(filename), parsed.path)
        ):
            raise ValueError(f"{name}: unexpected wheel URL or filename")
        if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ValueError(f"{name}: invalid wheel SHA-256")
        lines.append(f"{name} @ {url} --hash={digest}")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Validate the entire lock selection before emitting any requirements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lock", type=Path)
    args = parser.parse_args()
    try:
        output = requirements(tomllib.loads(args.lock.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(output, end="")


if __name__ == "__main__":
    main()
