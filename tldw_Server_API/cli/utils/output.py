"""
Output utilities for the CLI.

Provides formatted output helpers including tables, JSON output,
progress indicators, and rich formatting.
"""

import json
import sys
from datetime import datetime
from typing import Any, Optional

from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree
from tabulate import tabulate

# Global console instance
console = Console()


def print_error(message: str, exit_code: Optional[int] = None):
    """Print error message in red and optionally exit."""
    console.print(f"[bold red]Error:[/bold red] {message}")
    if exit_code is not None:
        sys.exit(exit_code)


def print_success(message: str):
    """Print success message in green."""
    console.print(f"[bold green]Success:[/bold green] {message}")


def print_info(message: str):
    """Print info message in blue."""
    console.print(f"[bold blue]Info:[/bold blue] {message}")


def print_warning(message: str):
    """Print warning message in yellow."""
    console.print(f"[bold yellow]Warning:[/bold yellow] {message}")


def print_json(data: Any, title: Optional[str] = None):
    """Print data as formatted JSON."""
    if title:
        console.print(f"\n[bold]{title}[/bold]")

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            console.print(data)
            return

    json_str = json.dumps(data, indent=2, default=str)
    json_renderable = JSON(json_str)
    console.print(json_renderable)


def print_table(data: list[dict[str, Any]], title: Optional[str] = None, format_style: str = "rich"):
    """
    Print data as a formatted table.

    Args:
        data: List of dictionaries to display
        title: Optional table title
        format_style: "rich" for Rich tables, "simple" for tabulate
    """
    if not data:
        print_info("No data to display")
        return

    if format_style == "rich":
        _print_rich_table(data, title)
    else:
        _print_simple_table(data, title)


def _print_rich_table(data: list[dict[str, Any]], title: Optional[str] = None):
    """Print table using Rich formatting."""
    if not data:
        return

    table = Table(show_header=True, header_style="bold magenta")

    # Add columns
    columns = list(data[0].keys())
    for column in columns:
        table.add_column(column.replace('_', ' ').title())

    # Add rows
    for row in data:
        table.add_row(*[str(row.get(col, '')) for col in columns])

    if title:
        console.print(f"\n[bold]{title}[/bold]")
    console.print(table)


def _print_simple_table(data: list[dict[str, Any]], title: Optional[str] = None):
    """Print table using tabulate."""
    if not data:
        return

    if title:
        print(f"\n{title}")
        print("=" * len(title))

    print(tabulate(data, headers="keys", tablefmt="grid"))


def print_health_status(health_data: dict[str, Any]):
    """Print health status with color coding."""
    status = health_data.get("status", "unknown")

    # Color coding based on status
    if status == "healthy":
        status_color = "bold green"
    elif status == "degraded":
        status_color = "bold yellow"
    elif status == "unhealthy":
        status_color = "bold red"
    else:
        status_color = "bold white"

    console.print(f"Overall Status: [{status_color}]{status.upper()}[/{status_color}]")

    # Print timestamp if available
    if "timestamp" in health_data:
        console.print(f"Checked at: {health_data['timestamp']}")

    # Print components if available
    if "components" in health_data:
        console.print("\n[bold]Component Status:[/bold]")
        components_tree = Tree("Components")

        for component, info in health_data["components"].items():
            comp_status = info.get("status", "unknown")
            if comp_status == "ok" or comp_status == "healthy":
                comp_color = "green"
            elif comp_status == "degraded":
                comp_color = "yellow"
            else:
                comp_color = "red"

            branch = components_tree.add(f"[{comp_color}]{component}[/{comp_color}]")

            # Add component details
            for key, value in info.items():
                if key != "status":
                    branch.add(f"{key}: {value}")

        console.print(components_tree)


def print_metrics_summary(metrics: dict[str, Any]):
    """Print metrics in a formatted summary."""
    console.print("\n[bold]Metrics Summary[/bold]")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_column("Description")

    for key, value in metrics.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                table.add_row(
                    f"{key}.{subkey}",
                    str(subvalue),
                    ""
                )
        else:
            table.add_row(key, str(value), "")

    console.print(table)


def print_config_section(config: dict[str, Any], section: Optional[str] = None):
    """Print configuration section with syntax highlighting."""
    if section:
        if section in config:
            data = {section: config[section]}
            console.print(f"\n[bold]Configuration Section: {section}[/bold]")
        else:
            print_error(f"Section '{section}' not found in configuration")
            return
    else:
        data = config
        console.print("\n[bold]Full Configuration[/bold]")

    yaml_content = json.dumps(data, indent=2, default=str)
    syntax = Syntax(yaml_content, "yaml", theme="monokai", line_numbers=True)
    console.print(syntax)


def print_progress_bar(total: int, description: str = "Processing"):
    """Create and return a progress bar context manager."""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    )


def print_evaluation_results(results: dict[str, Any]):
    """Print evaluation results in a formatted way."""
    console.print("\n[bold]Evaluation Results[/bold]")

    # Basic info
    info_table = Table(show_header=False)
    info_table.add_column("Field", style="cyan")
    info_table.add_column("Value")

    if "evaluation_id" in results:
        info_table.add_row("Evaluation ID", str(results["evaluation_id"]))
    if "evaluation_type" in results:
        info_table.add_row("Type", str(results["evaluation_type"]))
    if "provider" in results:
        info_table.add_row("Provider", str(results["provider"]))
    if "timestamp" in results:
        info_table.add_row("Timestamp", str(results["timestamp"]))

    console.print(info_table)

    # Metrics/scores
    if "metrics" in results:
        console.print("\n[bold]Scores:[/bold]")
        metrics_table = Table(show_header=True, header_style="bold magenta")
        metrics_table.add_column("Metric")
        metrics_table.add_column("Score")
        metrics_table.add_column("Explanation")

        for metric_name, metric_data in results["metrics"].items():
            if isinstance(metric_data, dict):
                score = metric_data.get("score", "N/A")
                explanation = metric_data.get("explanation", "")
            else:
                score = metric_data
                explanation = ""

            # Color code scores
            if score is None:
                score_display = "[dim]unscored[/dim]"
            elif isinstance(score, (int, float)):
                if score >= 0.8:
                    score_color = "bold green"
                elif score >= 0.6:
                    score_color = "bold yellow"
                else:
                    score_color = "bold red"
                score_display = f"[{score_color}]{score:.3f}[/{score_color}]"
            else:
                score_display = str(score)

            metrics_table.add_row(metric_name, score_display, explanation[:100] + "..." if len(explanation) > 100 else explanation)

        console.print(metrics_table)

    # Additional details
    if "details" in results:
        console.print("\n[bold]Details:[/bold]")
        print_json(results["details"])


def print_banner(title: str, subtitle: Optional[str] = None):
    """Print a banner with title and optional subtitle."""
    content = f"[bold cyan]{title}[/bold cyan]"
    if subtitle:
        content += f"\n[dim]{subtitle}[/dim]"

    panel = Panel(content, expand=False, border_style="cyan")
    console.print(panel)


def format_timestamp(timestamp) -> str:
    """Format timestamp for display."""
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            return str(timestamp)

    if isinstance(timestamp, datetime):
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")

    return str(timestamp)


def format_bytes(bytes_value: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in seconds as human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:.0f}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"
