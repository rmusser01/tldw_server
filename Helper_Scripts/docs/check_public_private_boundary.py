#!/usr/bin/env python3
"""Check that public OSS surfaces avoid private hosted and commercial runtime references."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

SCAN_TARGETS = (
    "Docs/Published",
    "Docs/Operations",
    "Docs/Code_Documentation/Docs_Site_Guide.md",
    "Docs/User_Guides/Server/Production_Hardening_Checklist.md",
    "Dockerfiles",
    "Helper_Scripts/refresh_docs_published.sh",
    "Helper_Scripts/Deployment",
    "Helper_Scripts/Samples/Caddy",
    "Helper_Scripts/validate_hosted_saas_profile.py",
    ".github/workflows/mkdocs.yml",
    "apps/tldw-frontend/pages/_app.tsx",
    "apps/tldw-frontend/pages/login.tsx",
    "apps/tldw-frontend/pages/signup.tsx",
    "apps/tldw-frontend/pages/account/index.tsx",
    "apps/tldw-frontend/pages/billing/index.tsx",
    "apps/tldw-frontend/pages/auth/verify-email.tsx",
    "apps/tldw-frontend/pages/auth/reset-password.tsx",
    "apps/tldw-frontend/pages/auth/magic-link.tsx",
    "apps/tldw-frontend/components/layout/Header.tsx",
    "tldw_Server_API/Config_Files",
    "tldw_Server_API/app/core/Billing/README.md",
)

DENYLIST = {
    "Hosted_SaaS_Profile.md": "public surfaces should not point to the hosted SaaS profile doc",
    "Hosted_Staging_Runbook.md": "public surfaces should not point to the hosted staging runbook",
    "Hosted_Production_Runbook.md": "public surfaces should not point to the hosted production runbook",
    "Hosted_Staging_Operations_Runbook.md": "public surfaces should not point to the hosted staging operations runbook",
    "Hosted_Stripe_Test_Mode_Runbook.md": "public surfaces should not point to the hosted Stripe operations runbook",
    "docker-compose.hosted-saas-staging.yml": "public surfaces should not point to hosted staging deploy overlays",
    "docker-compose.hosted-saas-prod.yml": "public surfaces should not point to hosted production deploy overlays",
    "docker-compose.hosted-saas-prod.local-postgres.yml": "public surfaces should not point to hosted fallback deploy overlays",
    ".env.hosted-staging.example": "public surfaces should not point to hosted staging env templates",
    ".env.hosted-production.example": "public surfaces should not point to hosted production env templates",
    "Caddyfile.hosted-saas.compose": "public surfaces should not point to hosted Caddy samples",
    "Caddyfile.hosted-saas.prod.compose": "public surfaces should not point to hosted production Caddy samples",
    "@web/components/hosted": "public frontend entrypoints should not import hosted private components",
    "@web/lib/hosted-route-allowlist": "public frontend entrypoints should not import hosted route gating helpers",
    "tldw_Server_API/app/api/v1/endpoints/billing_webhooks.py": "public OSS surfaces should not reference Stripe webhook runtime",
    "tldw_Server_API/app/api/v1/endpoints/billing.py": "public OSS surfaces should not reference private billing runtime",
    "tldw_Server_API/app/core/Billing/stripe_client.py": "public OSS surfaces should not reference Stripe client runtime",
    "tldw_Server_API/app/services/stripe_metering_service.py": "public OSS surfaces should not reference Stripe metering runtime",
    "tldw_Server_API/app/api/v1/endpoints/admin/admin_billing.py": "public OSS surfaces should not reference extracted admin billing runtime",
    "tldw_Server_API/app/services/admin_billing_service.py": "public OSS surfaces should not reference extracted admin billing services",
    "apps/tldw-frontend/pages/account/index.tsx": "public OSS surfaces should not point to hosted customer account pages",
    "apps/tldw-frontend/pages/billing/index.tsx": "public OSS surfaces should not point to hosted billing pages",
}

TEXT_SUFFIXES = {".example", ".md", ".py", ".sh", ".ts", ".tsx", ".yaml", ".yml"}
TEXT_NAME_PREFIXES = ("Caddyfile", "Dockerfile")


def _is_scannable_text_file(path: Path) -> bool:
    """Return whether a candidate file should be scanned as a text asset."""
    return path.suffix in TEXT_SUFFIXES or any(
        path.name.startswith(prefix) for prefix in TEXT_NAME_PREFIXES
    )


def _token_pattern(token: str) -> re.Pattern[str]:
    """Build a denylist token matcher with boundaries tuned to file and import references."""
    escaped_token = re.escape(token)
    if token.startswith("@"):
        pattern = rf"(?<![A-Za-z0-9_]){escaped_token}(?=$|[/\"'`\s,);:#?])"
    else:
        pattern = rf"(?<![A-Za-z0-9_]){escaped_token}(?=$|[^A-Za-z0-9_./-])"
    return re.compile(pattern)


def _line_contains_token(line: str, token: str) -> bool:
    """Return whether a line contains a denylist token as a real reference."""
    return _token_pattern(token).search(line) is not None


def _iter_candidate_files() -> list[Path]:
    """Collect unique text-like files from the configured public scan targets."""
    files: list[Path] = []
    for target in SCAN_TARGETS:
        path = REPO_ROOT / target
        if not path.exists():
            continue
        if path.is_file():
            if _is_scannable_text_file(path):
                files.append(path)
            continue
        files.extend(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and _is_scannable_text_file(candidate)
        )
    return sorted(set(files))


def _should_skip(path: Path) -> bool:
    """Return whether a scanned path should be skipped from boundary validation."""
    return False


def _find_violations(path: Path) -> list[str]:
    """Scan a file for denylisted private references and return formatted violations."""
    relative_path = path.relative_to(REPO_ROOT)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="ignore")

    violations: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for token, message in DENYLIST.items():
            if _line_contains_token(line, token):
                violations.append(
                    f"{relative_path}:{line_number}: {message} ({token})"
                )
    return violations


def main() -> int:
    """Run the boundary scan and exit non-zero when public references violate policy."""
    violations: list[str] = []

    for path in _iter_candidate_files():
        if _should_skip(path):
            continue
        violations.extend(_find_violations(path))

    if violations:
        print("Public/private boundary violations detected:")
        for violation in violations:
            print(f"- {violation}")
        return 1

    print(
        "OK: public OSS surfaces do not reference private hosted assets or extracted commercial runtime."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
