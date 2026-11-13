"""Bounded isolated-world serialization over an already owned CDP session."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from .browser_accounting import non_negative_integer
from .controls import ParentStop

DEFAULT_MAX_ARTICLE_BYTES = 16_777_216

_ISOLATED_WORLD_NAME = "tldw-article-serialization-v1"
_HTML_SERIALIZATION_EXPRESSION = """(maxBytes) => {
  const doctype = document.doctype
    ? new XMLSerializer().serializeToString(document.doctype) + "\\n"
    : "";
  const html = doctype + document.documentElement.outerHTML;
  let size = 0;
  for (const character of html) {
    const codePoint = character.codePointAt(0);
    size += codePoint <= 0x7f ? 1 : codePoint <= 0x7ff ? 2 : codePoint <= 0xffff ? 3 : 4;
    if (size > maxBytes) {
      return { ok: false, size };
    }
  }
  return { ok: true, html };
}"""
_DOCUMENT_SERIALIZATION_EXPRESSION = """(maxBytes) => {
  const serialized = (HTML_SERIALIZER)(maxBytes);
  if (!serialized.ok) return serialized;
  let content = "";
  for (const selector of ["main", "article", '[role="main"]', "#content", ".content"]) {
    for (const element of document.querySelectorAll(selector)) {
      const text = element.innerText;
      if (text.length > content.length) content = text;
    }
  }
  if (!content) content = document.body.innerText;
  const author = document.querySelector('meta[name="author"]');
  const date = document.querySelector('meta[property="article:published_time"]');
  const dom = {title: document.title, author: author ? author.content : "Unknown",
               date: date ? date.content : "", content};
  let size = 0;
  for (const text of Object.values(dom)) {
    for (const character of text) {
      const codePoint = character.codePointAt(0);
      size += codePoint <= 0x7f ? 1 : codePoint <= 0x7ff ? 2 : codePoint <= 0xffff ? 3 : 4;
      if (size > maxBytes) return {ok: false, size};
    }
  }
  return {...serialized, dom};
}""".replace("HTML_SERIALIZER", _HTML_SERIALIZATION_EXPRESSION)


async def serialize_document(
    session: Any, max_bytes: int, *, capture_dom: bool = False, failure: Callable[[str, str], Exception]
) -> tuple[str, Mapping[str, str] | None]:
    if type(max_bytes) is not int or max_bytes < 0:
        raise failure("browser_error", "capability")
    send = getattr(session, "send", None)
    if not callable(send):
        raise failure("browser_error", "capability")

    async def command(
        method: str,
        params: dict[str, object] | None = None,
    ) -> Mapping[str, Any]:
        try:
            operation = send(method) if params is None else send(method, params)
            if not inspect.isawaitable(operation):
                raise TypeError("CDP send is not awaitable")
            response = await operation
        except asyncio.CancelledError:
            raise
        except ParentStop:
            raise
        except Exception:  # noqa: BLE001 - CDP failures are sanitized
            raise failure("browser_error", "capability") from None
        if not isinstance(response, Mapping) or "exceptionDetails" in response:
            raise failure("browser_error", "capability")
        return response

    frame_tree_response = await command("Page.getFrameTree")
    frame_tree = frame_tree_response.get("frameTree")
    if not isinstance(frame_tree, Mapping):
        raise failure("browser_error", "capability")
    frame = frame_tree.get("frame")
    if not isinstance(frame, Mapping):
        raise failure("browser_error", "capability")
    frame_id = frame.get("id")
    if not isinstance(frame_id, str) or not frame_id:
        raise failure("browser_error", "capability")

    isolated_world_response = await command(
        "Page.createIsolatedWorld",
        {
            "frameId": frame_id,
            "worldName": _ISOLATED_WORLD_NAME,
            "grantUniveralAccess": False,
        },
    )
    context_id = isolated_world_response.get("executionContextId")
    if type(context_id) is not int or context_id < 0:
        raise failure("browser_error", "capability")

    evaluation_response = await command(
        "Runtime.evaluate",
        {
            "expression": f"({(_DOCUMENT_SERIALIZATION_EXPRESSION if capture_dom else _HTML_SERIALIZATION_EXPRESSION)})({max_bytes})",
            "contextId": context_id,
            "returnByValue": True,
        },
    )
    remote_result = evaluation_response.get("result")
    if not isinstance(remote_result, Mapping):
        raise failure("browser_error", "capability")
    result = remote_result.get("value")
    if not isinstance(result, Mapping) or type(result.get("ok")) is not bool:
        raise failure("browser_error", "capability")
    if result["ok"] is True:
        html = result.get("html")
        if not isinstance(html, str) or len(html.encode("utf-8")) > max_bytes:
            raise failure("browser_error", "capability")
        if not capture_dom:
            return html, None
        dom = result.get("dom")
        if (
            not isinstance(dom, Mapping)
            or set(dom) != {"title", "author", "date", "content"}
            or any(not isinstance(value, str) for value in dom.values())
            or sum(len(value.encode("utf-8")) for value in dom.values()) > max_bytes
        ):
            raise failure("browser_error", "capability")
        return html, MappingProxyType(dict(dom))
    if "html" in result:
        raise failure("browser_error", "capability")
    try:
        non_negative_integer(result.get("size"))
    except (TypeError, ValueError, OverflowError):
        raise failure("browser_error", "capability") from None
    raise failure("response_too_large", "rendered_html")
