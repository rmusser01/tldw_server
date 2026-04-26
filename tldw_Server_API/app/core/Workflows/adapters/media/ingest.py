"""Media ingestion adapters.

This module includes adapters for media ingestion:
- media_ingest: Ingest media files
- process_media: Process media files (web scraping, PDF, ebook, XML, etc.)
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.exceptions import AdapterError
from tldw_Server_API.app.core.DB_Management.media_db.api import (
    get_media_repository,
    managed_media_database,
)
from tldw_Server_API.app.core.Security.egress import is_url_allowed, is_url_allowed_for_tenant
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.core.Workflows.adapters._common import (
    resolve_workflow_file_uri,
)
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.media._config import (
    MediaIngestConfig,
    ProcessMediaConfig,
)
from tldw_Server_API.app.core.Workflows.subprocess_utils import start_process, terminate_process

_MEDIA_INGEST_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ImportError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
)


@registry.register(
    "media_ingest",
    category="media",
    description="Ingest media files",
    parallelizable=True,
    tags=["media", "ingest"],
    config_model=MediaIngestConfig,
)
async def run_media_ingest_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Media ingestion step (v0.1 minimal) with optional yt-dlp/ffmpeg integration.

    Config:
      - sources: [{uri, media_type?}]
      - download: {enabled: bool, ydl_format?, max_filesize_mb?, retries?}
      - limits: {max_download_mb?, max_duration_sec?}
      - safety: {allowed_domains?: [string]}
      - timeout_seconds: int (enforced internally)
    Output:
      - { media_ids: [], metadata: [...], transcripts: [], rag_indexed: False }
    """
    sources = config.get("sources") or []
    download = (config.get("download") or {}).copy()
    safety = config.get("safety") or {}
    timeout_seconds = int(config.get("timeout_seconds", 300))

    out = {
        "media_ids": [],
        "metadata": [],
        "transcripts": [],
        "rag_indexed": False,
    }

    if not sources:
        return out

    # Security: allowed domains for HTTP(S)
    allowed_domains = set(safety.get("allowed_domains") or [])

    start_ts = time.time()
    for idx, src in enumerate(sources):
        uri = str(src.get("uri", "")).strip()
        if not uri:
            continue

        # file:// URIs: read and optionally chunk locally
        if uri.startswith("file://"):
            try:
                resolved_path = resolve_workflow_file_uri(uri, context, config)
            except AdapterError:
                out["metadata"].append({"source": uri, "status": "file_access_denied"})
                continue
            try:
                try:
                    text = resolved_path.read_text(encoding="utf-8")
                except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
                    text = resolved_path.read_text(errors="ignore")
            except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
                out["metadata"].append({"source": uri, "status": "read_error"})
                continue

            extracted_text = text if (config.get("extraction", {}).get("extract_text", True)) else ""
            if extracted_text:
                out["text"] = (out.get("text") or "") + ("\n\n" if out.get("text") else "") + extracted_text

            chunks_desc: list[dict[str, Any]] = []
            try:
                from tldw_Server_API.app.core.Chunking import Chunker
                chunker = Chunker()
                ch_cfg = config.get("chunking") or {}
                # Determine method/params
                method = None
                max_size = None
                overlap = None
                if ch_cfg.get("strategy"):
                    if ch_cfg.get("strategy") == "hierarchical":
                        method = ch_cfg.get("hierarchical", {}).get("levels", [{}])[0].get("strategy") or "sentences"
                        hierarchical = True
                    else:
                        method = ch_cfg.get("strategy")
                        hierarchical = False
                    max_size = int(ch_cfg.get("max_tokens") or ch_cfg.get("max_size") or 400)
                    overlap = int(ch_cfg.get("overlap") or 0)
                elif ch_cfg.get("name"):
                    method = ch_cfg.get("name")
                    params = ch_cfg.get("params") or {}
                    hierarchical = False
                    max_size = int(params.get("max_tokens") or params.get("max_size") or 400)
                    overlap = int(params.get("overlap") or 0)
                else:
                    hierarchical = False

                if method:
                    if ch_cfg.get("strategy") == "hierarchical" or hierarchical:
                        flat = chunker.chunk_text_hierarchical_flat(
                            text=extracted_text,
                            method=method,
                            max_size=max_size or 400,
                            overlap=overlap or 0,
                        )
                        for i, item in enumerate(flat):
                            md = item.get("metadata") or {}
                            chunks_desc.append({
                                "id": f"{idx}-{i}",
                                "order": i,
                                "level": md.get("ancestry_titles") and len(md.get("ancestry_titles")) or 1,
                                "parent_id": None,
                                "chunker_name": method,
                                "chunker_version": "1.0.0",
                                "metadata": md,
                            })
                    else:
                        parts = chunker.chunk_text_with_metadata(
                            text=extracted_text,
                            method=method,
                            max_size=max_size or 400,
                            overlap=overlap or 0,
                        )
                        for i, part in enumerate(parts):
                            chunks_desc.append({
                                "id": f"{idx}-{i}",
                                "order": i,
                                "level": 1,
                                "parent_id": None,
                                "chunker_name": method,
                                "chunker_version": "1.0.0",
                                "metadata": {
                                    "index": part.metadata.index,
                                    "start_char": part.metadata.start_char,
                                    "end_char": part.metadata.end_char,
                                    "word_count": part.metadata.word_count,
                                    "language": part.metadata.language,
                                },
                            })
            except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
                pass

            if chunks_desc:
                out.setdefault("chunks", []).extend(chunks_desc)
            meta_local = {
                "source": uri,
                "media_type": src.get("media_type", "auto"),
                "status": "local_ok",
                "chunk_count": len(out.get("chunks", [])),
            }
            # Optional: persist to Media DB if indexing requested
            try:
                indexing = config.get("indexing") or {}
                if isinstance(indexing, dict) and indexing.get("index_in_rag") and extracted_text:
                    try:
                        from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
                        _mdb_path = str(DatabasePaths.get_media_db_path(DatabasePaths.get_single_user_id()))
                    except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
                        logger.exception("Failed to resolve Media DB path for workflow indexing")
                        raise
                    title = (config.get("metadata", {}) or {}).get("title") or resolved_path.name
                    keywords = (config.get("metadata", {}) or {}).get("tags") or []
                    media_type = src.get("media_type") or "document"
                    with managed_media_database(
                        "workflow_engine",
                        db_path=_mdb_path,
                        initialize=False,
                    ) as mdb:
                        media_id, _, msg = get_media_repository(mdb).add_media_with_keywords(
                            url=uri,
                            title=title,
                            media_type=media_type,
                            content=extracted_text,
                            keywords=keywords,
                            overwrite=False,
                            chunk_options=None,
                            chunks=None,
                        )
                    if media_id:
                        out.setdefault("media_ids", []).append(media_id)
                        meta_local["stored_media_id"] = media_id
                        meta_local["db_message"] = msg
                        # Mark as indexed at DB level (vectorization may still be pending)
                        out["rag_indexed"] = True
            except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
                # Non-fatal; proceed without DB write
                pass
            out["metadata"].append(meta_local)
            continue

        # HTTP(S) URIs: honor allowed_domains if provided
        if uri.startswith("http://") or uri.startswith("https://"):
            from urllib.parse import urlparse
            host = (urlparse(uri).hostname or "").lower().rstrip(".")
            if allowed_domains:
                host_allowed = False
                for domain in allowed_domains:
                    try:
                        if not domain:
                            continue
                        dom = str(domain).lower().lstrip(".")
                        if not dom:
                            continue
                        if host == dom or host.endswith(f".{dom}"):
                            host_allowed = True
                            break
                    except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
                        continue
                if not host_allowed:
                    out["metadata"].append({
                        "source": uri,
                        "status": "skipped_disallowed_domain",
                    })
                    continue
            # Global egress policy: private IPs and allowlist
            try:
                tenant_id = str(context.get("tenant_id") or "default") if isinstance(context, dict) else "default"
                allowed = False
                try:
                    allowed = is_url_allowed_for_tenant(uri, tenant_id)
                except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
                    allowed = is_url_allowed(uri)
                if not allowed:
                    out["metadata"].append({
                        "source": uri,
                        "status": "blocked_egress",
                    })
                    continue
            except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
                out["metadata"].append({"source": uri, "status": "blocked_egress_err"})
                continue

            # Limits: basic max_download_mb gate if provided (prevents invoking yt-dlp for obviously large files via URL params)
            limits = config.get("limits") or {}
            max_download_mb = limits.get("max_download_mb")
            if isinstance(max_download_mb, (int, float)) and (download.get("max_filesize_mb") or 0) > max_download_mb:
                out["metadata"].append({
                    "source": uri,
                    "status": "skipped_exceeds_limit",
                    "limit_mb": max_download_mb,
                })
                continue

            # Skip actual download in tests/no-network; detect env var
            if is_test_mode():
                out["metadata"].append({
                    "source": uri,
                    "status": "simulated_download",
                })
                continue

            # Attempt yt-dlp via subprocess for better isolation
            ydl_format = download.get("ydl_format", "bestvideo+bestaudio/best")
            workdir = Path(os.getenv("WORKFLOWS_TMP", ".tmp")) / "workflows"
            step_dir = workdir / f"ingest_{int(time.time()*1000)}_{idx}"
            step_dir.mkdir(parents=True, exist_ok=True)
            log_dir = step_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            # Build command with safe options
            output_tpl = str(step_dir / "%(title).80s.%(ext)s")
            cmd = [
                sys.executable,
                "-m",
                "yt_dlp",
                "-f",
                ydl_format,
                "-o",
                output_tpl,
                "--no-playlist",
                "--no-cache-dir",
                uri,
            ]
            # Optional max filesize
            max_mb = download.get("max_filesize_mb")
            if max_mb:
                try:
                    _mb = int(max_mb)
                    cmd.extend(["--max-filesize", f"{_mb}M"])
                except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
                    pass

            task = start_process(cmd, workdir=step_dir, log_dir=log_dir)
            # Record subprocess info for engine-driven cancellation
            try:
                if callable(context.get("record_subprocess")):
                    context["record_subprocess"](
                        pid=task.pid,
                        pgid=task.pgid,
                        workdir=str(step_dir),
                        stdout_path=str(task.stdout_path),
                        stderr_path=str(task.stderr_path),
                    )
            except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
                pass

            # Poll with timeout
            exited = False
            while time.time() - start_ts < timeout_seconds:
                # cooperative cancel
                if callable(context.get("is_cancelled")) and context["is_cancelled"]():
                    terminate_process(task)
                    return {"__status__": "cancelled"}
                # heartbeat callback
                try:
                    if callable(context.get("heartbeat")):
                        context["heartbeat"]()
                except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
                    pass
                await asyncio.sleep(0.25)
                # Check if any file downloaded
                if any(step_dir.glob("*.*")):
                    exited = True
                    break
            if not exited:
                terminate_process(task)
                # Attach small tails for debugging
                stdout_tail = None
                stderr_tail = None
                try:
                    if task.stdout_path.exists():
                        data = task.stdout_path.read_bytes()
                        if len(data) > 4096:
                            data = data[-4096:]
                        stdout_tail = data.decode("utf-8", errors="replace")
                except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
                    pass
                try:
                    if task.stderr_path.exists():
                        data = task.stderr_path.read_bytes()
                        if len(data) > 4096:
                            data = data[-4096:]
                        stderr_tail = data.decode("utf-8", errors="replace")
                except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
                    pass
                meta_timeout = {
                    "source": uri,
                    "status": "timeout",
                }
                if stdout_tail:
                    meta_timeout["stdout_tail"] = stdout_tail
                if stderr_tail:
                    meta_timeout["stderr_tail"] = stderr_tail
                out["metadata"].append(meta_timeout)
                try:
                    if callable(context.get("append_event")):
                        context["append_event"]("step_log_tail", {"stdout_tail": stdout_tail, "stderr_tail": stderr_tail, "source": uri})
                except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
                    pass
                continue

            # Build metadata including small log tails for debugging
            stdout_tail2 = None
            stderr_tail2 = None
            try:
                if task.stdout_path.exists():
                    data = task.stdout_path.read_bytes()
                    if len(data) > 4096:
                        data = data[-4096:]
                    stdout_tail2 = data.decode("utf-8", errors="replace")
            except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
                pass
            try:
                if task.stderr_path.exists():
                    data = task.stderr_path.read_bytes()
                    if len(data) > 4096:
                        data = data[-4096:]
                    stderr_tail2 = data.decode("utf-8", errors="replace")
            except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
                pass

            meta_entry = {
                "source": uri,
                "status": "downloaded",
                "dir": str(step_dir),
            }
            if stdout_tail2:
                meta_entry["stdout_tail"] = stdout_tail2
            if stderr_tail2:
                meta_entry["stderr_tail"] = stderr_tail2
            try:
                if (stdout_tail2 or stderr_tail2) and callable(context.get("append_event")):
                    context["append_event"]("step_log_tail", {"stdout_tail": stdout_tail2, "stderr_tail": stderr_tail2, "source": uri})
            except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
                pass

            # Attach chunking/indexing metadata if requested in config
            chunking = config.get("chunking") or {}
            if isinstance(chunking, dict):
                # Support both preset strategy and registry name@version
                if chunking.get("name"):
                    meta_entry["chunker_name"] = str(chunking.get("name"))
                    if chunking.get("version"):
                        meta_entry["chunker_version"] = str(chunking.get("version"))
                elif chunking.get("strategy"):
                    meta_entry["chunker_name"] = str(chunking.get("strategy"))
                    meta_entry["chunker_version"] = "1.0.0"

            indexing = config.get("indexing") or {}
            if isinstance(indexing, dict):
                meta_entry["index_requested"] = bool(indexing.get("index_in_rag", False))
                if indexing.get("collection"):
                    meta_entry["index_collection"] = str(indexing.get("collection"))

            out["metadata"].append(meta_entry)
            # Persist artifacts for downloaded files
            try:
                if callable(context.get("add_artifact")):
                    import hashlib
                    import mimetypes
                    for fp in step_dir.glob("*.*"):
                        # Skip log files
                        if fp.name in {"stdout.log", "stderr.log"} or fp.parent.name == "logs":
                            continue
                        try:
                            size_b = fp.stat().st_size
                        except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
                            size_b = None
                        try:
                            mime, _ = mimetypes.guess_type(str(fp))
                        except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
                            mime = None
                        sha256 = None
                        try:
                            h = hashlib.sha256()
                            with fp.open("rb") as f:
                                for chunk in iter(lambda: f.read(65536), b""):
                                    h.update(chunk)
                            sha256 = h.hexdigest()
                        except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
                            logger.debug("Media ingest adapter: failed to compute sha256")
                        context["add_artifact"](
                            type="download",
                            uri=f"file://{fp}",
                            size_bytes=size_b,
                            mime_type=mime,
                            checksum_sha256=sha256,
                            metadata={"workdir": str(step_dir)},
                        )
            except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
                pass

    return out


@registry.register(
    "process_media",
    category="media",
    description="Process media files",
    parallelizable=False,
    tags=["media", "processing"],
    config_model=ProcessMediaConfig,
)
async def run_process_media_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Process media ephemerally using internal services (no persistence).

    Supports kinds:
      - web_scraping (existing)
      - pdf (file_uri)
      - ebook (not yet implemented in workflows)
      - xml (not yet implemented in workflows)
      - mediawiki_dump (file_uri)
      - podcast (not yet implemented in workflows)

    For smoother chains, the adapter emits a best-effort `text` field in
    outputs (e.g., first article summary/content, or extracted text), so
    downstream steps like `prompt` and `tts` can use `last.text` directly.
    """
    def _emit(out: dict[str, Any]) -> dict[str, Any]:
        # Attach best-effort text for chaining convenience
        try:
            if "text" not in out or not out.get("text"):
                # Try to find first rich text content
                txt: str | None = None
                # Web scraping shape: results -> list of {content, summary}
                results = out.get("results") if isinstance(out, dict) else None
                if isinstance(results, list) and results:
                    item0 = results[0]
                    if isinstance(item0, dict):
                        txt = (item0.get("summary") or item0.get("content") or item0.get("text") or "")
                # Generic shapes
                if not txt:
                    txt = out.get("content") or out.get("text") or ""
                if txt:
                    out["text"] = txt
        except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
            pass
        return out

    # Early cancel
    try:
        if callable(context.get("is_cancelled")) and context["is_cancelled"]():
            return {"__status__": "cancelled"}
    except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
        pass
    kind = str(config.get("kind") or "web_scraping").strip().lower()
    if kind in {"ebook", "xml", "podcast"}:
        return {
            "error": "not_implemented",
            "kind": kind,
            "message": f"process_media kind '{kind}' is not implemented in workflows",
        }
    # Web scraping
    if kind == "web_scraping":
        try:
            from tldw_Server_API.app.services.web_scraping_service import process_web_scraping_task
        except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
            return {"error": "web_scraping_service_unavailable"}
        # Extract and sanitize config
        scrape_method = str(config.get("scrape_method") or "Individual URLs")
        url_input = str(config.get("url_input") or "").strip()
        url_level = config.get("url_level")
        try:
            url_level = int(url_level) if url_level is not None else None
        except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
            url_level = None
        max_pages = int(config.get("max_pages", 10))
        max_depth = int(config.get("max_depth", 3))
        summarize = bool(config.get("summarize") or config.get("summarize_checkbox") or False)
        custom_prompt = config.get("custom_prompt")
        api_name = config.get("api_name")
        system_prompt = config.get("system_prompt")
        try:
            temperature = float(config.get("temperature", 0.7))
        except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
            temperature = 0.7
        custom_cookies = config.get("custom_cookies") if isinstance(config.get("custom_cookies"), list) else None
        user_agent = config.get("user_agent")
        custom_headers = config.get("custom_headers") if isinstance(config.get("custom_headers"), dict) else None

        try:
            result = await process_web_scraping_task(
                scrape_method=scrape_method,
                url_input=url_input,
                url_level=url_level,
                max_pages=max_pages,
                max_depth=max_depth,
                summarize_checkbox=summarize,
                custom_prompt=custom_prompt,
                api_name=api_name,
                api_key=None,
                keywords="",
                custom_titles=None,
                system_prompt=system_prompt,
                temperature=temperature,
                custom_cookies=custom_cookies,
                mode="ephemeral",
                user_id=None,
                user_agent=user_agent,
                custom_headers=custom_headers,
            )
        except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
            logger.exception("Web scraping process media failed")
            return {"error": "process_media_error"}
        # Normalize response
        articles = []
        try:
            articles = result.get("results") or result.get("articles") or []
            if isinstance(articles, dict):
                articles = [articles]
        except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
            articles = []
        return _emit({"kind": "web_scraping", "status": result.get("status", "ok"), "count": len(articles), "results": articles})

    # PDF (file_uri required)
    if kind == "pdf":
        file_uri = str(config.get("file_uri") or "").strip()
        if not file_uri.startswith("file://"):
            return {"error": "missing_or_invalid_file_uri"}
        try:
            resolved_path = resolve_workflow_file_uri(file_uri, context, config)
        except AdapterError:
            return {"error": "file_access_denied"}
        try:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import process_pdf_task
            fb = resolved_path.read_bytes()
            performed_analysis = bool(config.get("perform_analysis", True))
            chunk_opts = config.get("chunking") or {}
            result = await process_pdf_task(
                file_bytes=fb,
                filename=resolved_path.name,
                parser=str(config.get("parser") or "pymupdf4llm"),
                perform_analysis=performed_analysis,
                api_name=config.get("api_name") if performed_analysis else None,
                custom_prompt=config.get("custom_prompt"),
                system_prompt=config.get("system_prompt"),
                perform_chunking=bool(chunk_opts.get("perform", performed_analysis)),
                chunk_method=chunk_opts.get("method"),
                max_chunk_size=chunk_opts.get("max_size"),
                chunk_overlap=chunk_opts.get("overlap"),
            )
        except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
            logger.exception("PDF process media failed")
            return {"error": "pdf_process_error"}
        # Map to a simple shape
        out = {
            "kind": "pdf",
            "status": result.get("status", "Success"),
            "content": result.get("text") or result.get("content") or "",
            "metadata": result.get("metadata") or {},
        }
        return _emit(out)

    # MediaWiki dump (file_uri) - ephemeral process
    if kind == "mediawiki_dump":
        file_uri = str(config.get("file_uri") or "").strip()
        if not file_uri.startswith("file://"):
            return {"error": "missing_or_invalid_file_uri"}
        # In workflows, we return a placeholder summary; full streaming is endpoint-only
        try:
            resolved_path = resolve_workflow_file_uri(file_uri, context, config)
        except AdapterError:
            return {"error": "file_access_denied"}
        try:
            content = resolved_path.read_text(errors="ignore")
        except _MEDIA_INGEST_NONCRITICAL_EXCEPTIONS:
            content = ""
        return _emit({"kind": "mediawiki_dump", "content": content[:5000], "metadata": {"file": resolved_path.name}})

    return {"error": f"unsupported_process_media_kind:{kind}"}
