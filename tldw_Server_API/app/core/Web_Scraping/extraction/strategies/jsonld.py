"""JSON-LD and microdata extraction strategy."""

import asyncio
import json
import re
from typing import Any

from bs4 import BeautifulSoup

_JSONLD_PRIMARY_TYPES = {
    "newsarticle",
    "article",
    "blogposting",
    "report",
    "techarticle",
    "medicalscholarlyarticle",
    "analysisnewsarticle",
    "opinionnewsarticle",
    "reviewnewsarticle",
    "scholarlyarticle",
}
_JSONLD_SECONDARY_TYPES = {"webpage", "webcontent", "creativework", "blog"}
_JSONLD_PARSE_ERROR = "jsonld_parse_failed"


def _jsonld_type_tokens(value: Any) -> list[str]:
    tokens: list[str] = []
    if isinstance(value, list):
        for item in value:
            tokens.extend(_jsonld_type_tokens(item))
        return tokens
    if isinstance(value, str):
        for entry in value.split():
            token = entry.strip()
            if not token:
                continue
            if "/" in token:
                token = token.rsplit("/", 1)[-1]
            if ":" in token:
                token = token.split(":", 1)[-1]
            token = token.strip().lower()
            if token:
                tokens.append(token)
    return tokens


def _jsonld_type_set(node: dict[str, Any]) -> set[str]:
    return set(_jsonld_type_tokens(node.get("@type") or node.get("type")))


def _jsonld_collect_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(_jsonld_collect_text(item))
        return parts
    if isinstance(value, dict):
        if "@value" in value:
            return _jsonld_collect_text(value.get("@value"))
        for key in ("name", "headline", "text", "description", "value"):
            if key in value:
                parts = _jsonld_collect_text(value.get(key))
                if parts:
                    return parts
    return []


def _jsonld_join_text(value: Any, join_with: str) -> str:
    parts = [part for part in _jsonld_collect_text(value) if part]
    return join_with.join(parts) if parts else ""


def _jsonld_extract_author(node: dict[str, Any]) -> str | None:
    for key in ("author", "creator", "publisher"):
        if key not in node:
            continue
        names = _jsonld_collect_text(node.get(key))
        unique = list(dict.fromkeys(names))
        if unique:
            return ", ".join(unique)
    return None


def _jsonld_score_candidate(node: dict[str, Any]) -> tuple[int, int]:
    types = _jsonld_type_set(node)
    score = 8 if types & _JSONLD_PRIMARY_TYPES else 4 if types & _JSONLD_SECONDARY_TYPES else 0
    content = _jsonld_join_text(node.get("articleBody"), "\n\n")
    if content:
        score += 3
    text = _jsonld_join_text(node.get("text"), "\n\n")
    if text:
        score += 2
    if _jsonld_join_text(node.get("description"), " "):
        score += 1
    if _jsonld_join_text(node.get("headline") or node.get("name"), " "):
        score += 1
    return score, len(content) if content else len(text)


def _collect_jsonld_nodes(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [node for item in data for node in _collect_jsonld_nodes(item)]
    if not isinstance(data, dict):
        return []
    nodes = [data]
    graph = data.get("@graph")
    if isinstance(graph, list):
        nodes.extend(item for item in graph if isinstance(item, dict))
    return nodes


def _resolve_jsonld_refs(value: Any, id_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [node for item in value for node in _resolve_jsonld_refs(item, id_map)]
    if isinstance(value, dict):
        ref_id = value.get("@id")
        return [id_map[ref_id]] if isinstance(ref_id, str) and ref_id in id_map else [value]
    return [id_map[value]] if isinstance(value, str) and value in id_map else []


def _microdata_prop_value(tag: Any) -> str | None:
    if not tag:
        return None
    if tag.has_attr("content"):
        content = str(tag.get("content") or "").strip()
        if content:
            return content
    name = getattr(tag, "name", "") or ""
    if name in {"a", "area", "link"}:
        return str(tag.get("href") or "").strip() or None
    if name in {"img", "audio", "video", "source"}:
        return str(tag.get("src") or "").strip() or None
    if name == "time":
        datetime_value = str(tag.get("datetime") or "").strip()
        if datetime_value:
            return datetime_value
    text = tag.get_text(" ", strip=True) if hasattr(tag, "get_text") else ""
    return text or None


def _extract_microdata_items(soup: BeautifulSoup) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for scope in soup.find_all(attrs={"itemscope": True}):
        properties: dict[str, Any] = {}
        for prop in scope.find_all(attrs={"itemprop": True}):
            parent_scope = prop.find_parent(attrs={"itemscope": True})
            if parent_scope is not None and parent_scope is not scope:
                continue
            prop_name = prop.get("itemprop")
            value = _microdata_prop_value(prop)
            if not prop_name or not value:
                continue
            existing = properties.get(prop_name)
            properties[prop_name] = (
                value if existing is None else existing + [value] if isinstance(existing, list) else [existing, value]
            )
        if not properties:
            continue
        item: dict[str, Any] = dict(properties)
        if item_type := scope.get("itemtype"):
            item["@type"] = item_type
        if item_id := scope.get("itemid"):
            item["@id"] = item_id
        items.append(item)
    return items


def _decode_all_json(payload: str) -> list[Any]:
    decoder = json.JSONDecoder()
    index = 0
    objects: list[Any] = []
    while index < len(payload):
        brace = payload.find("{", index)
        bracket = payload.find("[", index)
        if brace == -1 and bracket == -1:
            break
        start = bracket if brace == -1 or bracket != -1 and bracket < brace else brace
        try:
            obj, end = decoder.raw_decode(payload, start)
        except asyncio.CancelledError:
            raise
        except (json.JSONDecodeError, ValueError):
            index = start + 1
            continue
        objects.append(obj)
        index = end
    return objects


def extract_jsonld_entities(html_text: str, url: str) -> dict[str, Any]:
    """Extract the best article-like JSON-LD or microdata item from HTML."""
    result: dict[str, Any] = {
        "url": url,
        "title": "N/A",
        "author": "N/A",
        "content": "",
        "date": "N/A",
        "extraction_successful": False,
    }
    if not html_text:
        return result
    soup = BeautifulSoup(html_text, "html.parser")
    nodes: list[dict[str, Any]] = []
    parse_failed = False
    scripts = soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.IGNORECASE)})
    for script in scripts:
        payload = (script.string or script.get_text() or "").strip()
        if not payload:
            continue
        try:
            objects = _decode_all_json(payload) or [json.loads(payload)]
        except asyncio.CancelledError:
            raise
        except (json.JSONDecodeError, TypeError, ValueError):
            parse_failed = True
            continue
        for obj in objects:
            nodes.extend(_collect_jsonld_nodes(obj))

    nodes.extend(_extract_microdata_items(soup))
    if not nodes:
        if parse_failed:
            result["jsonld_error"] = _JSONLD_PARSE_ERROR
        return result

    id_map = {node_id: node for node in nodes if isinstance(node_id := node.get("@id"), str)}
    expanded_nodes = list(nodes)
    for node in nodes:
        for reference_key in ("mainEntity", "mainEntityOfPage"):
            expanded_nodes.extend(_resolve_jsonld_refs(node.get(reference_key), id_map))
    unique_nodes = list({id(node): node for node in expanded_nodes}.values())
    best_node = max(unique_nodes, key=_jsonld_score_candidate, default=None)
    if not best_node:
        return result

    result["jsonld_types"] = sorted(_jsonld_type_set(best_node))
    if title := _jsonld_join_text(best_node.get("headline") or best_node.get("name") or best_node.get("title"), " "):
        result["title"] = title
    if author := _jsonld_extract_author(best_node):
        result["author"] = author
    if date_value := (
        _jsonld_join_text(best_node.get("datePublished"), " ")
        or _jsonld_join_text(best_node.get("dateCreated"), " ")
        or _jsonld_join_text(best_node.get("dateModified"), " ")
    ):
        result["date"] = date_value
    if summary := (
        _jsonld_join_text(best_node.get("description"), " ")
        or _jsonld_join_text(best_node.get("abstract"), " ")
        or _jsonld_join_text(best_node.get("summary"), " ")
    ):
        result["summary"] = summary
    if content := (
        _jsonld_join_text(best_node.get("articleBody"), "\n\n") or _jsonld_join_text(best_node.get("text"), "\n\n")
    ):
        result["content"] = content
    result["extraction_successful"] = bool(result["content"].strip())
    return result
