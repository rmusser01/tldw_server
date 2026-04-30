from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlencode

from loguru import logger

from tldw_Server_API.app.core.http_client import afetch

from .connector_base import BaseConnector


class NotionConnector(BaseConnector):
    name = "notion"

    def __init__(self, client_id: str | None = None, client_secret: str | None = None, redirect_base: str | None = None):
        super().__init__(
            client_id=client_id or os.getenv("CONNECTOR_NOTION_CLIENT_ID"),
            client_secret=client_secret or os.getenv("CONNECTOR_NOTION_SECRET"),
            redirect_base=redirect_base or os.getenv("CONNECTOR_REDIRECT_BASE_URL"),
        )

    def authorize_url(self, state: str | None = None, scopes: list[str] | None = None, redirect_path: str = "/api/v1/connectors/providers/notion/callback") -> str:
        redirect_uri = f"{self.redirect_base}{redirect_path}"
        if not self.client_id:
            return f"{redirect_uri}?scaffold=1&state={state or ''}"
        params = {
            "owner": "user",
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            # Notion does not require scopes in the same way; left for compatibility
        }
        if state:
            params["state"] = state
        return f"https://api.notion.com/v1/oauth/authorize?{urlencode(params)}"

    async def exchange_code(self, code: str, redirect_uri: str) -> dict[str, Any]:
        oauth_endpoint = "https://api.notion.com/v1/oauth/token"
        headers = {"Content-Type": "application/json"}
        body = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
        }
        # Notion uses Basic auth with client_id:client_secret base64
        import base64
        if self.client_id and self.client_secret:
            basic = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
            headers["Authorization"] = f"Basic {basic}"
        resp = await afetch(method="POST", url=oauth_endpoint, json=body, headers=headers, timeout=30)
        try:
            resp.raise_for_status()
            tok = resp.json()
        finally:
            close = getattr(resp, "aclose", None)
            if callable(close):
                await close()
        return {
            "access_token": tok.get("access_token"),
            "refresh_token": tok.get("refresh_token"),
            "expires_in": tok.get("expires_in"),
            "token_type": tok.get("token_type"),
            "provider": self.name,
            "display_name": "Notion Account",
            "email": None,
            # Capture workspace metadata when available for policy enforcement
            "workspace_id": tok.get("workspace_id"),
            "workspace_name": tok.get("workspace_name"),
        }

    async def list_sources(self, account: dict[str, Any], parent_remote_id: str | None = None, *, page_size: int = 50, cursor: str | None = None):
        token = (account.get("tokens") or {}).get("access_token") or account.get("access_token")
        if not token:
            return [], None
        headers = {
            "Authorization": f"Bearer {token}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        }
        # If a parent_remote_id is provided, treat it as a database id and list its row pages.
        if parent_remote_id:
            url = f"https://api.notion.com/v1/databases/{parent_remote_id}/query"
            payload = {"page_size": int(page_size)}
            if cursor:
                payload["start_cursor"] = cursor
            resp = await afetch(method="POST", url=url, headers=headers, json=payload, timeout=30)
            try:
                resp.raise_for_status()
                data = resp.json()
            finally:
                close = getattr(resp, "aclose", None)
                if callable(close):
                    await close()
            items = []
            for r in data.get("results", []):
                pid = r.get("id")
                title = None
                try:
                    props = r.get("properties") or {}
                    for v in props.values():
                        if isinstance(v, dict) and v.get("type") == "title":
                            arr = v.get("title") or []
                            if arr:
                                title = arr[0].get("plain_text")
                                break
                except Exception:
                    logger.debug("Notion connector failed to extract page title")
                items.append({"id": pid, "name": title or pid, "type": "page", "last_edited_time": r.get("last_edited_time")})
            return items, data.get("next_cursor")
        else:
            url = "https://api.notion.com/v1/search"
            payload = {
                "page_size": int(page_size),
                # No query; filter both pages and databases
            }
            if cursor:
                payload["start_cursor"] = cursor
            resp = await afetch(method="POST", url=url, headers=headers, json=payload, timeout=30)
            try:
                resp.raise_for_status()
                data = resp.json()
            finally:
                close = getattr(resp, "aclose", None)
                if callable(close):
                    await close()
        items = []
        for r in data.get("results", []):
            obj = r.get("object")
            if obj == "page":
                pid = r.get("id")
                # title from properties
                title = None
                try:
                    props = r.get("properties") or {}
                    for v in props.values():
                        if isinstance(v, dict) and v.get("type") == "title":
                            arr = v.get("title") or []
                            if arr:
                                title = arr[0].get("plain_text")
                                break
                except Exception:
                    logger.debug("Notion connector failed to extract search result title")
                items.append({"id": pid, "name": title or pid, "type": "page", "last_edited_time": r.get("last_edited_time")})
            elif obj == "database":
                did = r.get("id")
                items.append({"id": did, "name": (r.get("title") or [{}])[0].get("plain_text") if r.get("title") else did, "type": "database"})
        return items, data.get("next_cursor")

    async def download_file(self, account: dict[str, Any], file_id: str) -> bytes:
        """Return Markdown bytes for a page by traversing blocks recursively, with code/table handling."""
        token = (account.get("tokens") or {}).get("access_token") or account.get("access_token")
        if not token:
            return b""
        headers = {
            "Authorization": f"Bearer {token}",
            "Notion-Version": "2022-06-28",
        }
        async def _fetch_children(block_id: str) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            cursor = None
            while True:
                params = {"page_size": 50}
                if cursor:
                    params["start_cursor"] = cursor
                resp = await afetch(
                    method="GET",
                    url=f"https://api.notion.com/v1/blocks/{block_id}/children",
                    headers=headers,
                    params=params,
                    timeout=30,
                )
                try:
                    resp.raise_for_status()
                    data = resp.json()
                finally:
                    close = getattr(resp, "aclose", None)
                    if callable(close):
                        await close()
                out.extend(data.get("results", []))
                if not data.get("has_more"):
                    break
                cursor = data.get("next_cursor")
            return out

        def _rich_text_to_md(rich: list[dict[str, Any]]) -> str:
            out = []
            for p in rich or []:
                txt = p.get("plain_text") or ""
                ann = (p.get("annotations") or {})
                if ann.get("code"):
                    txt = f"`{txt}`"
                if ann.get("bold"):
                    txt = f"**{txt}**"
                if ann.get("italic"):
                    txt = f"*{txt}*"
                if ann.get("strikethrough"):
                    txt = f"~~{txt}~~"
                out.append(txt)
            return "".join(out)

        async def _render_block(b: dict[str, Any], depth: int = 0) -> list[str]:
            t = b.get("type")
            bt = b.get(t) or {}
            lines: list[str] = []
            indent = "  " * depth
            # Headings
            if t in {"heading_1", "heading_2", "heading_3"}:
                text = _rich_text_to_md(bt.get("rich_text") or [])
                prefix = "#" * (1 if t == "heading_1" else 2 if t == "heading_2" else 3)
                lines.append(f"{prefix} {text}")
            # Paragraph
            elif t == "paragraph":
                text = _rich_text_to_md(bt.get("rich_text") or [])
                if text:
                    lines.append(text)
            # Lists
            elif t in {"bulleted_list_item", "numbered_list_item"}:
                text = _rich_text_to_md(bt.get("rich_text") or [])
                bullet = "-" if t == "bulleted_list_item" else "1."
                lines.append(f"{indent}{bullet} {text}")
                if b.get("has_children"):
                    kids = await _fetch_children(b.get("id"))
                    for kb in kids:
                        lines.extend(await _render_block(kb, depth + 1))
            # Toggle
            elif t == "toggle":
                text = _rich_text_to_md(bt.get("rich_text") or [])
                lines.append(f"<details><summary>{text}</summary>")
                if b.get("has_children"):
                    kids = await _fetch_children(b.get("id"))
                    for kb in kids:
                        lines.extend(await _render_block(kb, depth))
                lines.append("</details>")
            # Code block
            elif t == "code":
                lang = (bt.get("language") or "").strip() or ""
                text = _rich_text_to_md(bt.get("rich_text") or [])
                lines.append(f"```{lang}\n{text}\n```")
            # Quote
            elif t == "quote":
                text = _rich_text_to_md(bt.get("rich_text") or [])
                lines.append(f"> {text}")
            # Callout
            elif t == "callout":
                text = _rich_text_to_md(bt.get("rich_text") or [])
                emoji = (bt.get("icon") or {}).get("emoji") if isinstance(bt.get("icon"), dict) else None
                prefix = f"{emoji} " if emoji else ""
                lines.append(f"> {prefix}{text}")
            # Table (render table and its rows)
            elif t == "table":
                if b.get("has_children"):
                    rows = await _fetch_children(b.get("id"))
                    # Build Markdown table; infer column count from first row
                    head_cells = []
                    body_rows: list[list[str]] = []
                    for i, row in enumerate(rows):
                        if row.get("type") != "table_row":
                            continue
                        cells = []
                        for cell in (row.get("table_row") or {}).get("cells", []) or []:
                            cells.append(_rich_text_to_md(cell))
                        if i == 0:
                            head_cells = cells
                        else:
                            body_rows.append(cells)
                    if head_cells:
                        lines.append("| " + " | ".join(head_cells) + " |")
                        lines.append("| " + " | ".join(["---"] * len(head_cells)) + " |")
                        for r in body_rows:
                            lines.append("| " + " | ".join(r) + " |")
            # Image (external URL if available, include caption)
            elif t == "image":
                cap = _rich_text_to_md(bt.get("caption") or [])
                src = None
                if bt.get("type") == "external":
                    src = (bt.get("external") or {}).get("url")
                elif bt.get("type") == "file":
                    src = (bt.get("file") or {}).get("url")
                alt = cap or "image"
                if src:
                    lines.append(f"![{alt}]({src})")
                else:
                    lines.append(f"![{alt}]")
            # Unsupported types: render a placeholder and children if any
            else:
                typename = t or "unknown"
                lines.append(f"<!-- unsupported block: {typename} -->")
                if b.get("has_children"):
                    kids = await _fetch_children(b.get("id"))
                    for kb in kids:
                        lines.extend(await _render_block(kb, depth))
            return lines

        # Render the whole page
        top_level = await _fetch_children(file_id)
        lines: list[str] = []
        for b in top_level:
            lines.extend(await _render_block(b, 0))
        md = "\n".join(lines).encode("utf-8")
        return md

    async def refresh_access_token(self, refresh_token: str) -> dict[str, Any] | None:
        """Refresh Notion access token using Basic auth."""
        if not (self.client_id and self.client_secret and refresh_token):
            return None
        oauth_endpoint = "https://api.notion.com/v1/oauth/token"
        headers = {"Content-Type": "application/json"}
        import base64
        basic = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        headers["Authorization"] = f"Basic {basic}"
        body = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        resp = await afetch(method="POST", url=oauth_endpoint, json=body, headers=headers, timeout=30)
        try:
            resp.raise_for_status()
            tok = resp.json()
        finally:
            close = getattr(resp, "aclose", None)
            if callable(close):
                await close()
        return {
            "access_token": tok.get("access_token"),
            "refresh_token": tok.get("refresh_token") or refresh_token,
            "expires_in": tok.get("expires_in"),
            "token_type": tok.get("token_type"),
        }
