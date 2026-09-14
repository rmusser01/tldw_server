"""Dedicated entry point for canonical admin-webhook operator commands."""

from __future__ import annotations

from tldw_Server_API.cli.commands.admin_webhooks import admin_webhooks_group


def main() -> None:
    """Run the canonical admin-webhook command group."""
    admin_webhooks_group()


if __name__ == "__main__":
    main()
