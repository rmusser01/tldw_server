"""GitHub integration adapters.

This module includes adapters for GitHub operations:
- github_create_issue: Create GitHub issues
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.integration._config import GitHubCreateIssueConfig


@registry.register(
    "github_create_issue",
    category="integration",
    description="Create GitHub issue",
    parallelizable=True,
    tags=["integration", "github"],
    config_model=GitHubCreateIssueConfig,
)
async def run_github_create_issue_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Create a GitHub issue."""
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    repo = config.get("repo")  # format: owner/repo
    title = config.get("title")
    if not repo or not title:
        return {"error": "missing_repo_or_title", "issue_url": None}

    if isinstance(title, str):
        title = _tmpl(title, context) or title

    body = config.get("body") or ""
    if isinstance(body, str):
        body = _tmpl(body, context) or body

    labels = config.get("labels") or []
    assignees = config.get("assignees") or []

    token = config.get("token") or os.getenv("GITHUB_TOKEN")
    if not token:
        return {"error": "missing_github_token", "issue_url": None}

    try:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.github.com/repos/{repo}/issues",
                headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github.v3+json",
                },
                json={
                    "title": title,
                    "body": body,
                    "labels": labels,
                    "assignees": assignees,
                },
                timeout=30,
            )

            if response.status_code == 201:
                data = response.json()
                return {"issue_url": data.get("html_url"), "issue_number": data.get("number"), "created": True}
            else:
                return {"error": f"github_api_error: {response.status_code} - {response.text}", "created": False}

    except Exception:
        logger.exception("GitHub issue creation failed")
        return {"error": "GitHub issue creation failed", "issue_url": None, "created": False}
