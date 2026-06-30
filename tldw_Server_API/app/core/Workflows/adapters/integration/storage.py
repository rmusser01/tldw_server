"""Cloud storage integration adapters.

This module includes adapters for cloud storage operations:
- s3_upload: Upload to S3-compatible storage
- s3_download: Download from S3-compatible storage
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Workflows.adapters._common import resolve_workflow_file_path
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.integration._config import S3DownloadConfig, S3UploadConfig


@registry.register(
    "s3_upload",
    category="integration",
    description="Upload to S3",
    parallelizable=True,
    tags=["integration", "storage"],
    config_model=S3UploadConfig,
)
async def run_s3_upload_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Upload content to S3-compatible storage."""
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    bucket = config.get("bucket")
    key = config.get("key")
    if not bucket or not key:
        return {"error": "missing_bucket_or_key", "uploaded": False}

    # Get content
    content = config.get("content")
    file_path = config.get("file_path")

    if file_path:
        if isinstance(file_path, str):
            file_path = _tmpl(file_path, context) or file_path
        try:
            resolved_path = resolve_workflow_file_path(file_path, context, config)
            content = resolved_path.read_bytes()
        except Exception:
            return {"error": "file_read_error", "uploaded": False}
    elif content is None:
        prev = context.get("prev") or context.get("last") or {}
        content = prev.get("content") or prev.get("text") or ""

    if isinstance(content, str):
        content = content.encode("utf-8")

    endpoint_url = config.get("endpoint_url") or os.getenv("S3_ENDPOINT_URL")
    access_key = config.get("access_key") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = config.get("secret_key") or os.getenv("AWS_SECRET_ACCESS_KEY")
    region = config.get("region") or os.getenv("AWS_REGION", "us-east-1")

    try:
        import boto3

        client_kwargs = {"region_name": region}
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        if access_key and secret_key:
            client_kwargs["aws_access_key_id"] = access_key
            client_kwargs["aws_secret_access_key"] = secret_key

        s3 = boto3.client("s3", **client_kwargs)
        s3.put_object(Bucket=bucket, Key=key, Body=content)

        return {"uploaded": True, "bucket": bucket, "key": key, "size_bytes": len(content)}

    except ImportError:
        return {"error": "boto3_not_installed", "uploaded": False}
    except Exception:
        logger.exception("S3 upload failed")
        return {"error": "S3 upload failed", "uploaded": False}


@registry.register(
    "s3_download",
    category="integration",
    description="Download from S3",
    parallelizable=True,
    tags=["integration", "storage"],
    config_model=S3DownloadConfig,
)
async def run_s3_download_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Download content from S3-compatible storage."""
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    bucket = config.get("bucket")
    key = config.get("key")
    if not bucket or not key:
        return {"error": "missing_bucket_or_key", "content": None}

    endpoint_url = config.get("endpoint_url") or os.getenv("S3_ENDPOINT_URL")
    access_key = config.get("access_key") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = config.get("secret_key") or os.getenv("AWS_SECRET_ACCESS_KEY")
    region = config.get("region") or os.getenv("AWS_REGION", "us-east-1")
    as_text = config.get("as_text", True)

    try:
        import boto3

        client_kwargs = {"region_name": region}
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        if access_key and secret_key:
            client_kwargs["aws_access_key_id"] = access_key
            client_kwargs["aws_secret_access_key"] = secret_key

        s3 = boto3.client("s3", **client_kwargs)
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read()

        if as_text:
            content = content.decode("utf-8", errors="ignore")

        return {"content": content, "bucket": bucket, "key": key, "size_bytes": len(content) if isinstance(content, (str, bytes)) else 0}

    except ImportError:
        return {"error": "boto3_not_installed", "content": None}
    except Exception:
        logger.exception("S3 download failed")
        return {"error": "S3 download failed", "content": None}
