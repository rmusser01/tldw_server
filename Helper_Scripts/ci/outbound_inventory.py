"""Load reviewed outbound declarations without importing the app or resolving a host."""

from __future__ import annotations

import ipaddress
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from urllib.parse import urlsplit

_PORTS = {"http": 80, "https": 443}
_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_ID = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_SYMBOL = re.compile(r"[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)*")


class InventoryError(ValueError):
    """A stable, safe-to-report inventory failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _origin(url: str, *, declaration: bool) -> str:
    try:
        if not isinstance(url, str) or any(ord(char) <= 32 or char in "\\\x7f" for char in url):
            raise ValueError
        parts = urlsplit(url)
        default_port = _PORTS[parts.scheme]
        host = parts.hostname
        if not host or "%" in host or parts.username is not None or parts.netloc.endswith(":"):
            raise ValueError
        if "[" in parts.netloc and not re.fullmatch(r"\[[^\]]+\](?::[0-9]+)?", parts.netloc):
            raise ValueError
        if declaration and (parts.path not in {"", "/"} or "?" in url or "#" in url):
            raise ValueError
        port = parts.port
        if port == 0:
            raise ValueError
        host = host.removesuffix(".")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if "[" in parts.netloc:
                raise
            host = host.encode("idna").decode("ascii").lower().removesuffix(".")
            if len(host) > 253 or not all(_LABEL.fullmatch(label) for label in host.split(".")):
                raise ValueError from None
        else:
            host = f"[{address}]" if address.version == 6 else str(address)
        suffix = f":{port}" if port is not None and port != default_port else ""
        return f"{parts.scheme}://{host}{suffix}"
    except (KeyError, ValueError):
        raise InventoryError("invalid_origin") from None


def origin_only(request_url: str) -> str:
    """Return a canonical HTTP(S) scheme, host and effective port."""
    return _origin(request_url, declaration=False)


def _unique_json(items: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in items:
        if key in result:
            raise InventoryError("duplicate_json_key")
        result[key] = value
    return result


def _fields(raw: object, keys: str) -> dict:
    if not isinstance(raw, dict) or raw.keys() != set(keys.split()):
        raise InventoryError("invalid_schema")
    return raw


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or any(ord(char) < 32 for char in value):
        raise InventoryError("invalid_schema")
    return value


def _optional_text(value: object) -> str | None:
    return None if value is None else _text(value)


def _id(value: object) -> str:
    value = _text(value)
    if not _ID.fullmatch(value):
        raise InventoryError("invalid_schema")
    return value


def _source(value: object) -> str:
    value = _text(value)
    path = PurePosixPath(value)
    if (path.is_absolute() or ".." in path.parts or not path.parts
            or str(path) != value or any(char in value for char in "\\:*?[]#")):
        raise InventoryError("invalid_source_anchor")
    return value


def _list(value: object) -> list:
    if not isinstance(value, list):
        raise InventoryError("invalid_schema")
    return value


@dataclass(frozen=True)
class AssetIdentity:
    source: str
    repository: str | None
    resource: str
    revision: str | None


@dataclass(frozen=True)
class AssetEntry:
    id: str
    identity: AssetIdentity
    evidence: str


@dataclass(frozen=True)
class Capability:
    id: str
    kind: str
    component: str
    source_path: str
    purpose: str
    phase: str
    enabled_when: str
    origins: frozenset[str]
    assets: Mapping[str, AssetEntry]


@dataclass(frozen=True)
class ReviewedException:
    id: str
    source_path: str
    symbol: str
    owner: str
    follow_up: str


def _capability(raw: object) -> Capability:
    row = _fields(raw, "id kind component source_path purpose phase enabled_when origins assets")
    kind = _text(row["kind"])
    phase = _text(row["phase"])
    raw_origins = _list(row["origins"])
    origins = frozenset(_origin(url, declaration=True) for url in raw_origins)
    if len(origins) != len(raw_origins):
        raise InventoryError("duplicate_origin")
    raw_assets = _list(row["assets"])
    if (kind not in {"builtin", "user-directed"} or phase not in {"setup", "runtime"}
            or (kind == "user-directed" and (origins or raw_assets))
            or (kind == "builtin" and not (origins or raw_assets))):
        raise InventoryError("invalid_schema")
    assets = {}
    for raw_asset in raw_assets:
        asset = _fields(raw_asset, "id identity evidence")
        asset_id = _id(asset["id"])
        if asset_id in assets:
            raise InventoryError("duplicate_asset")
        if asset["evidence"] != "declared":
            raise InventoryError("invalid_schema")
        identity = _fields(asset["identity"], "source repository resource revision")
        assets[asset_id] = AssetEntry(asset_id, AssetIdentity(
            _id(identity["source"]), _optional_text(identity["repository"]),
            _text(identity["resource"]), _optional_text(identity["revision"]),
        ), "declared")
    return Capability(_id(row["id"]), kind, _id(row["component"]), _source(row["source_path"]),
                      _text(row["purpose"]), phase, _text(row["enabled_when"]),
                      origins, MappingProxyType(assets))


@dataclass(frozen=True)
class Inventory:
    capabilities: Mapping[str, Capability]
    exceptions: Mapping[str, ReviewedException]

    @classmethod
    def load(cls, path: Path) -> Inventory:
        """Validate a versioned review file and return immutable, indexed declarations."""
        try:
            raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json)
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise InventoryError("unreadable_inventory") from None
        data = _fields(raw, "version capabilities exceptions")
        if type(data["version"]) is not int or data["version"] != 1:
            raise InventoryError("unsupported_version")
        capabilities = {}
        for item in _list(data["capabilities"]):
            entry = _capability(item)
            if entry.id in capabilities:
                raise InventoryError("duplicate_capability")
            capabilities[entry.id] = entry
        exceptions = {}
        anchors = set()
        for item in _list(data["exceptions"]):
            row = _fields(item, "id source_path symbol owner follow_up")
            entry = ReviewedException(_id(row["id"]), _source(row["source_path"]), _text(row["symbol"]),
                                      _text(row["owner"]), _text(row["follow_up"]))
            if not _SYMBOL.fullmatch(entry.symbol):
                raise InventoryError("invalid_source_anchor")
            anchor = (entry.source_path, entry.symbol)
            if entry.id in exceptions or anchor in anchors:
                raise InventoryError("duplicate_exception")
            exceptions[entry.id] = entry
            anchors.add(anchor)
        return cls(MappingProxyType(capabilities), MappingProxyType(exceptions))

    def check_origin(self, capability: str, request_url: str) -> None:
        """Require a reviewed exact origin for the given capability."""
        origin = origin_only(request_url)
        entry = self.capabilities.get(capability)
        if entry is None or origin not in entry.origins:
            raise InventoryError("undeclared_origin")

    def require_asset(self, capability: str, asset_id: str, actual: AssetIdentity) -> AssetEntry:
        """Require a reviewed resource identity, including its repository and revision."""
        entry = self.capabilities.get(capability)
        asset = None if entry is None else entry.assets.get(asset_id)
        if asset is None:
            raise InventoryError("undeclared_asset")
        if actual != asset.identity:
            raise InventoryError("asset_identity_mismatch")
        return asset
