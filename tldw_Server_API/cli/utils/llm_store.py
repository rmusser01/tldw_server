"""A click group whose LLM commands load the admin provider-override store first (TASK-15581)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import click


class LLMStoreGroup(click.Group):
    """Load the provider-override store before one of ``llm_commands`` runs, and keep it fresh until exit.

    Every LLM call refuses while the store is unloaded, and nothing else loads it in a CLI process. A request for help
    (an option, or an LLM group given no subcommand) and shell completion skip the load, so they never need the AuthNZ
    database.
    """

    def __init__(self, *args: Any, llm_commands: Iterable[str] = (), **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.llm_commands = frozenset(llm_commands)

    def resolve_command(self, ctx: click.Context, args: list[str]) -> tuple[Any, Any, list[str]]:
        cmd_name, cmd, rest = super().resolve_command(ctx, args)
        prints_help = not set(ctx.help_option_names).isdisjoint(rest) or (
            not rest and isinstance(cmd, click.Group) and cmd.no_args_is_help
        )
        if cmd_name in self.llm_commands and not ctx.resilient_parsing and not prints_help:
            from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
                LLMProviderOverridesRefreshError,
                standalone_llm_provider_overrides_sync,
            )

            try:
                ctx.with_resource(standalone_llm_provider_overrides_sync())
            except LLMProviderOverridesRefreshError:
                raise click.ClickException(
                    "The admin LLM provider overrides could not be loaded from the AuthNZ database, and every LLM "
                    "call refuses without them. Check DATABASE_URL, and that AuthNZ is set up "
                    "(python -m tldw_Server_API.app.core.AuthNZ.initialize)."
                ) from None
        return cmd_name, cmd, rest
