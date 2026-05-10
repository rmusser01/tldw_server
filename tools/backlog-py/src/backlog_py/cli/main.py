from __future__ import annotations

from pathlib import Path

import click

from backlog_py.core.repository import ReadOnlyRepository, TaskRecord
from backlog_py.storage.project import discover_project


@click.group()
@click.option("--cwd", type=click.Path(path_type=Path), default=None, help="Backlog project directory.")
@click.pass_context
def main(ctx: click.Context, cwd: Path | None) -> None:
    """Python compatibility clone of Backlog.md."""
    ctx.obj = {"cwd": cwd}


@main.command("task")
@click.argument("args", nargs=-1)
@click.option("--plain", is_flag=True, help="Print plain text output.")
@click.pass_context
def task_command(ctx: click.Context, args: tuple[str, ...], plain: bool) -> None:
    """View tasks."""
    if args == ("list",):
        for task_record in _repository(ctx).list_tasks():
            click.echo(_format_task_line(task_record, plain=plain))
        return
    if len(args) != 1:
        raise click.UsageError("Missing task id.")
    task_id = args[0]
    task_record = _repository(ctx).get_task(task_id)
    click.echo(_format_task_detail(task_record, plain=plain))


@main.command("search")
@click.argument("query")
@click.option("--plain", is_flag=True, help="Print plain text output.")
@click.pass_context
def search_command(ctx: click.Context, query: str, plain: bool) -> None:
    """Search active tasks."""
    for task_record in _repository(ctx).search_tasks(query):
        click.echo(_format_task_line(task_record, plain=plain))


@main.command("board")
@click.pass_context
def board_command(ctx: click.Context) -> None:
    """Print task board grouped by status."""
    for status, tasks in _repository(ctx).board().items():
        click.echo(f"{status}:")
        for task_record in tasks:
            click.echo(f"  {_format_task_line(task_record, plain=True)}")


@main.group("config")
def config_group() -> None:
    """Inspect Backlog.md configuration."""


@config_group.command("list")
@click.pass_context
def config_list(ctx: click.Context) -> None:
    """Print effective configuration."""
    project = _project(ctx)
    config = project.config
    click.echo(f"projectName: {config.project_name}")
    click.echo(f"defaultStatus: {config.default_status}")
    click.echo(f"remoteOperations: {_bool_text(config.remote_operations)}")
    click.echo(f"autoCommit: {_bool_text(config.auto_commit)}")
    click.echo(f"bypassGitHooks: {_bool_text(config.bypass_git_hooks)}")
    click.echo(f"checkActiveBranches: {_bool_text(config.check_active_branches)}")
    click.echo(f"activeBranchDays: {config.active_branch_days}")
    if config.statuses is not None:
        click.echo("statuses:")
        for status in config.statuses:
            click.echo(f"  - {status}")
    if config.definition_of_done is not None:
        click.echo("definitionOfDone:")
        for item in config.definition_of_done:
            click.echo(f"  - {item}")


def _project(ctx: click.Context):
    cwd = ctx.obj.get("cwd") if ctx.obj else None
    return discover_project(Path.cwd(), explicit_cwd=cwd)


def _repository(ctx: click.Context) -> ReadOnlyRepository:
    return ReadOnlyRepository(_project(ctx))


def _format_task_line(task_record: TaskRecord, *, plain: bool) -> str:
    if plain:
        return f"{task_record.id} [{task_record.status}] {task_record.title}"
    return f"{task_record.id} - {task_record.title} ({task_record.status})"


def _format_task_detail(task_record: TaskRecord, *, plain: bool) -> str:
    if plain:
        parts = [
            f"{task_record.id} [{task_record.status}] {task_record.title}",
            "",
            task_record.description or task_record.body.strip(),
        ]
        return "\n".join(parts).rstrip()
    return task_record.raw_source


def _bool_text(value: bool) -> str:
    return "true" if value else "false"
