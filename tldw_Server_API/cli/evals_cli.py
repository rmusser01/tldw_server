"""
Main CLI entry point for tldw Evaluations module.

Provides a comprehensive command-line interface for:
- Health checks and status monitoring
- Evaluation execution (G-Eval, RAG, Quality Assessment)
- Database management and migrations
- Configuration management
- User and rate limit management
- Webhook management and testing
- Development and testing utilities
"""

import io
import sys
from pathlib import Path
from typing import Optional

import click
from loguru import logger
from rich.traceback import install as install_rich_traceback

# Install rich traceback for better error display
install_rich_traceback()

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tldw_Server_API.app.core.Evaluations.cli.benchmark_cli import benchmark_group
from tldw_Server_API.app.core.Evaluations.cli.evals_cli_enhanced import recipes_group
from tldw_Server_API.app.core.Evaluations.cli.persona_chat_judge_cli import persona_chat_judge_group
from tldw_Server_API.cli.commands.config import config_group
from tldw_Server_API.cli.commands.database import db_group
from tldw_Server_API.cli.commands.evaluation import eval_group
from tldw_Server_API.cli.commands.export import export_group
from tldw_Server_API.cli.commands.health import health_group
from tldw_Server_API.cli.commands.testing import test_group
from tldw_Server_API.cli.commands.users import users_group
from tldw_Server_API.cli.commands.watchlists import watchlists_group
from tldw_Server_API.cli.commands.webhooks import webhook_group
from tldw_Server_API.cli.utils.config import ConfigError, load_cli_config
from tldw_Server_API.cli.utils.output import print_error, print_info


# Global CLI context
class CLIContext:
    """Global CLI context for sharing configuration and state."""

    def __init__(self):
        self.config_path: Optional[str] = None
        self.db_path: Optional[str] = None
        self.log_level: str = "INFO"
        self.quiet: bool = False
        self.config: Optional[dict] = None

    def load_config(self):
        """Load configuration with error handling."""
        try:
            self.config = load_cli_config(self.config_path, self.db_path)
        except ConfigError as e:
            if not self.quiet:
                print_error(f"Configuration error: {e}")
            sys.exit(1)
        except Exception as e:
            if not self.quiet:
                print_error(f"Failed to load configuration: {e}")
            sys.exit(1)


# Create global context
cli_context = CLIContext()

_IO_EXCEPTIONS = (BrokenPipeError, OSError, ValueError, io.UnsupportedOperation)


class _SafeStreamWrapper:
    """Best-effort stream proxy that swallows teardown-time IO errors."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, message: str):
        try:
            self._stream.write(message)
            self.flush()
        except _IO_EXCEPTIONS:
            pass

    def flush(self):
        try:
            self._stream.flush()
        except _IO_EXCEPTIONS:
            pass

    def isatty(self) -> bool:
        try:
            return bool(getattr(self._stream, "isatty", lambda: False)())
        except _IO_EXCEPTIONS:
            return False

    def __getattr__(self, attr):
        return getattr(self._stream, attr)


def _safe_stderr_stream():
    """
    Resolve the best available stderr stream and make sink writes teardown-safe.
    """
    stream = sys.stderr if sys.stderr is not None else sys.__stderr__
    return _SafeStreamWrapper(stream or sys.__stderr__)


@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.option(
    '--config',
    type=click.Path(exists=True, readable=True, path_type=Path),
    help='Path to configuration file (defaults to evaluations_config.yaml)'
)
@click.option(
    '--db-path',
    type=click.Path(path_type=Path),
    help='Database path override'
)
@click.option(
    '--log-level',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], case_sensitive=False),
    default='INFO',
    help='Logging level'
)
@click.option(
    '--quiet', '-q',
    is_flag=True,
    help='Suppress output except errors'
)
@click.version_option(version="0.1.2", prog_name="tldw-evals")
@click.pass_context
def main(ctx, config, db_path, log_level, quiet):
    """
    tldw Evaluations CLI - Comprehensive evaluation management tool.

    This CLI provides standalone access to all tldw Evaluations functionality
    including health monitoring, evaluation execution, database management,
    and configuration management.

    Examples:
        tldw-evals health                    # Check system health
        tldw-evals eval geval "text" "summary"    # Run G-Eval
        tldw-evals db status                 # Show database status
        tldw-evals config validate           # Validate configuration
    """
    # Configure logging
    logger.remove()  # Remove default handler

    if not quiet:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        logger.add(_safe_stderr_stream(), format=log_format, level=log_level.upper())
    else:
        # In quiet mode, only show errors
        logger.add(_safe_stderr_stream(), level="ERROR", format="{message}")

    # Store global options in context
    cli_context.config_path = str(config) if config else None
    cli_context.db_path = str(db_path) if db_path else None
    cli_context.log_level = log_level.upper()
    cli_context.quiet = quiet

    # Make context available to subcommands
    ctx.ensure_object(dict)
    ctx.obj['cli_context'] = cli_context


# Add command groups
main.add_command(health_group, name='health')
main.add_command(eval_group, name='eval')
main.add_command(db_group, name='db')
main.add_command(config_group, name='config')
main.add_command(users_group, name='users')
main.add_command(webhook_group, name='webhook')
main.add_command(test_group, name='test')
main.add_command(export_group, name='export')
main.add_command(watchlists_group, name='watchlists')
main.add_command(benchmark_group, name='benchmark')
main.add_command(recipes_group, name='recipes')
main.add_command(persona_chat_judge_group, name='persona-chat-judge')


@main.command(name="list-benchmarks")
@click.option("--detailed", "-d", is_flag=True, help="Show detailed information")
@click.option(
    "--output-format",
    "-o",
    type=click.Choice(["table", "json"]),
    default="table",
)
@click.pass_context
def list_benchmarks_alias(ctx, detailed, output_format):
    """Compatibility alias for benchmark list."""
    return ctx.invoke(
        benchmark_group.commands["list"],
        detailed=detailed,
        output_format=output_format,
    )


@main.command()
@click.pass_context
def interactive(ctx):
    """
    Launch interactive mode for guided operations.

    Provides a menu-driven interface for common tasks.
    """
    from tldw_Server_API.cli.utils.interactive import run_interactive_mode

    try:
        cli_context.load_config()
        run_interactive_mode(cli_context)
    except KeyboardInterrupt:
        if not cli_context.quiet:
            print_info("\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        logger.exception("Interactive mode error")
        if not cli_context.quiet:
            print_error(f"Interactive mode failed: {e}")
        sys.exit(1)


@main.command()
@click.option(
    '--shell',
    type=click.Choice(['bash', 'zsh', 'fish']),
    default='bash',
    help='Shell type for completion'
)
def completion(shell):
    """
    Generate shell completion script.

    Examples:
        # Bash
        eval "$(tldw-evals completion --shell bash)"

        # Zsh (add to ~/.zshrc)
        eval "$(tldw-evals completion --shell zsh)"

        # Fish (add to ~/.config/fish/config.fish)
        tldw-evals completion --shell fish | source
    """
    from click_completion import get_completion_script

    try:
        completion_script = get_completion_script(prog_name='tldw-evals', shell=shell)  # nosec B604
        click.echo(completion_script)
    except Exception as e:
        print_error(f"Failed to generate completion script: {e}")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        if not cli_context.quiet:
            print_info("\nOperation cancelled.")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        logger.exception("CLI error")
        if not cli_context.quiet:
            print_error(f"CLI error: {e}")
        sys.exit(1)
