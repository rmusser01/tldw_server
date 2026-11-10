"""
Benchmark evaluation commands for the CLI.
"""

import asyncio
import json
from typing import Any, Optional

import click
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from tldw_Server_API.app.core.Evaluations.benchmark_loaders import load_benchmark_dataset
from tldw_Server_API.app.core.Evaluations.benchmark_registry import BenchmarkConfig, get_registry
from tldw_Server_API.app.core.Evaluations.benchmark_utils import format_benchmark_summary
from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager
from tldw_Server_API.cli.utils.async_runner import run_async_safely
from tldw_Server_API.cli.utils.output import (
    print_error,
    print_info,
    print_json,
    print_success,
    print_table,
    print_warning,
)


@click.group(name='benchmark')
def benchmark_group():
    """Benchmark evaluation commands."""
    pass


@benchmark_group.command('list')
@click.option('--detailed', '-d', is_flag=True, help='Show detailed information')
@click.option('--output-format', '-o', type=click.Choice(['table', 'json']), default='table')
@click.pass_context
def list_benchmarks(ctx, detailed, output_format):
    """List available benchmarks."""
    registry = get_registry()
    benchmarks = registry.list_benchmarks()

    if not benchmarks:
        print_warning("No benchmarks registered")
        return

    if output_format == 'json':
        data = {}
        for name in benchmarks:
            info = registry.get_benchmark_info(name)
            data[name] = info
        print_json(data, "Available Benchmarks")
    else:
        if detailed:
            for name in benchmarks:
                info = registry.get_benchmark_info(name)
                print_info(f"\n{name}")
                print(f"  Description: {info.get('description', 'N/A')}")
                print(f"  Type: {info.get('evaluation_type', 'N/A')}")
                print(f"  Source: {info.get('dataset_source', 'N/A')}")
                if info.get('metadata'):
                    print(f"  Metadata: {json.dumps(info['metadata'], indent=4)}")
        else:
            headers = ["Name", "Type", "Description"]
            rows = []
            for name in benchmarks:
                info = registry.get_benchmark_info(name)
                rows.append([
                    name,
                    info.get('evaluation_type', 'N/A'),
                    info.get('description', 'N/A')[:50] + '...' if len(info.get('description', '')) > 50 else info.get('description', 'N/A')
                ])
            print_table(rows, headers, "Available Benchmarks")


@benchmark_group.command('info')
@click.argument('benchmark_name')
@click.pass_context
def benchmark_info(ctx, benchmark_name):
    """Show detailed information about a benchmark."""
    registry = get_registry()
    config = registry.get(benchmark_name)

    if not config:
        print_error(f"Benchmark '{benchmark_name}' not found")
        return

    print_info(f"Benchmark: {config.name}")
    print(f"Description: {config.description}")
    print(f"Evaluation Type: {config.evaluation_type}")
    print(f"Dataset Source: {config.dataset_source}")
    print(f"Dataset Format: {config.dataset_format}")

    if config.field_mappings:
        print("\nField Mappings:")
        for target, source in config.field_mappings.items():
            print(f"  {target} <- {source}")

    if config.evaluation_params:
        print("\nEvaluation Parameters:")
        for key, value in config.evaluation_params.items():
            print(f"  {key}: {value}")

    if config.metadata:
        print("\nMetadata:")
        for key, value in config.metadata.items():
            print(f"  {key}: {value}")


@benchmark_group.command('run')
@click.argument('benchmark_name')
@click.option('--model', '-m', help='Model/API to use for evaluation')
@click.option('--api-key', help='API key for the model')
@click.option('--limit', '-l', type=click.IntRange(min=1), help='Limit number of samples to evaluate')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--resume-from', type=click.Path(exists=True), help='Resume from previous results')
@click.option('--parallel', '-p', type=int, default=4, help='Number of parallel workers')
@click.option('--save-progress', is_flag=True, help='Save progress after each batch')
@click.pass_context
def run_benchmark(ctx, benchmark_name, model, api_key, limit, output, resume_from, parallel, save_progress):
    """Run a benchmark evaluation."""
    registry = get_registry()
    config = registry.get(benchmark_name)

    if not config:
        print_error(f"Benchmark '{benchmark_name}' not found")
        return

    # Load dataset
    print_info(f"Loading dataset for {benchmark_name}...")
    try:
        dataset = load_benchmark_dataset(benchmark_name, limit=limit)
        if not dataset:
            print_error(f"Failed to load dataset for {benchmark_name}")
            return
        print_success(f"Loaded {len(dataset)} samples")
    except Exception as e:
        print_error(f"Error loading dataset: {e}")
        return

    # Resume from previous results if specified
    completed_ids = set()
    results = []
    if resume_from:
        try:
            with open(resume_from) as f:
                previous_results = json.load(f)
                results = previous_results.get('results', [])
                completed_ids = {r.get('id', r.get('question_id', i))
                               for i, r in enumerate(results)}
                print_info(f"Resuming from {len(completed_ids)} completed evaluations")
        except Exception as e:
            print_warning(f"Could not load previous results: {e}")

    # Filter out completed samples
    remaining_dataset = []
    for i, item in enumerate(dataset):
        item_id = item.get('id', item.get('question_id', i))
        if item_id not in completed_ids:
            remaining_dataset.append(item)

    if not remaining_dataset:
        print_success("All samples already evaluated!")
        if output and results:
            _save_results(results, benchmark_name, output)
        return

    print_info(f"Evaluating {len(remaining_dataset)} remaining samples...")

    # Create evaluator
    evaluator = registry.create_evaluator(benchmark_name)
    if not evaluator:
        print_error(f"Could not create evaluator for {benchmark_name}")
        return

    # Run evaluation with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task(f"Evaluating {benchmark_name}", total=len(remaining_dataset))

        # Process in batches
        batch_size = 10
        for i in range(0, len(remaining_dataset), batch_size):
            batch = remaining_dataset[i:i+batch_size]

            # Evaluate batch
            batch_results = run_async_safely(_evaluate_batch(
                batch, evaluator, model or "openai", api_key, parallel
            ))

            results.extend(batch_results)
            progress.update(task, advance=len(batch))

            # Save progress if requested
            if save_progress and output:
                _save_results(results, benchmark_name, output)

    # Generate summary
    summary = format_benchmark_summary(results, benchmark_name)
    print("\n" + summary)

    # Save final results
    if output:
        _save_results(results, benchmark_name, output)
        print_success(f"Results saved to {output}")


@benchmark_group.command('register')
@click.option('--name', '-n', required=True, help='Benchmark name')
@click.option('--description', '-d', help='Benchmark description')
@click.option('--type', '-t', 'eval_type', required=True,
              type=click.Choice(['multiple_choice', 'code_generation', 'instruction_following',
                               'function_calling', 'honesty', 'multi_turn']))
@click.option('--source', '-s', required=True, help='Dataset source (URL, file, or HF dataset)')
@click.option('--format', '-f', 'data_format',
              type=click.Choice(['json', 'jsonl', 'csv', 'huggingface', 'custom']),
              default='json')
@click.option('--config-file', type=click.Path(), help='Load configuration from file')
@click.option('--save-config', type=click.Path(), help='Save configuration to file')
@click.pass_context
def register_benchmark(ctx, name, description, eval_type, source, data_format, config_file, save_config):
    """Register a new benchmark."""
    registry = get_registry()

    if config_file:
        # Load from config file
        config = registry.load_config(config_file)
        if not config:
            print_error(f"Failed to load config from {config_file}")
            return
    else:
        # Create new config
        config = BenchmarkConfig(
            name=name,
            description=description or f"Custom benchmark: {name}",
            evaluation_type=eval_type,
            dataset_source=source,
            dataset_format=data_format,
            field_mappings={},  # User would need to customize these
            evaluation_params={},
            metadata={}
        )

    # Register the benchmark
    registry.register(config)
    print_success(f"Registered benchmark: {name}")

    # Save config if requested
    if save_config:
        registry.save_config(config, save_config)
        print_success(f"Config saved to {save_config}")


@benchmark_group.command('unregister')
@click.argument('benchmark_name')
@click.pass_context
def unregister_benchmark(ctx, benchmark_name):
    """Unregister a benchmark."""
    registry = get_registry()

    if benchmark_name not in registry.list_benchmarks():
        print_error(f"Benchmark '{benchmark_name}' not found")
        return

    registry.unregister(benchmark_name)
    print_success(f"Unregistered benchmark: {benchmark_name}")


@benchmark_group.command('validate')
@click.argument('benchmark_name')
@click.option('--samples', '-s', type=int, default=5, help='Number of samples to validate')
@click.pass_context
def validate_benchmark(ctx, benchmark_name, samples):
    """Validate a benchmark configuration by loading sample data."""
    registry = get_registry()
    config = registry.get(benchmark_name)

    if not config:
        print_error(f"Benchmark '{benchmark_name}' not found")
        return

    print_info(f"Validating benchmark: {benchmark_name}")

    # Try to load dataset
    try:
        dataset = load_benchmark_dataset(benchmark_name, limit=samples)
        if not dataset:
            print_error("Failed to load dataset")
            return

        print_success(f"Successfully loaded {len(dataset)} samples")

        # Show sample structure
        if dataset:
            print_info("\nSample structure:")
            sample = dataset[0]
            for key in sorted(sample.keys()):
                if key != '_original':
                    value = sample[key]
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    elif isinstance(value, list) and len(str(value)) > 100:
                        value = f"[list with {len(value)} items]"
                    print(f"  {key}: {value}")

        # Try to create evaluator
        evaluator = registry.create_evaluator(benchmark_name)
        if evaluator:
            print_success("Successfully created evaluator")

            # Try formatting first sample
            if dataset and config.field_mappings:
                try:
                    evaluator.format_for_custom_metric(dataset[0])
                    print_success("Successfully formatted sample for evaluation")
                except Exception as e:
                    print_warning(f"Error formatting sample: {e}")
        else:
            print_warning("Could not create evaluator (evaluation type may not be implemented)")

    except Exception as e:
        print_error(f"Validation failed: {e}")
        return


async def _evaluate_batch(batch: list[dict[str, Any]], evaluator,
                         model: str, api_key: Optional[str],
                         parallel: int) -> list[dict[str, Any]]:
    """Evaluate a batch of samples."""
    manager = EvaluationManager()
    results = []

    # Create tasks for parallel evaluation
    tasks = []
    for item in batch:
        # Format for evaluation
        eval_data = evaluator.format_for_custom_metric(item)

        # Create evaluation task
        task = manager.evaluate_custom_metric(
            metric_name=eval_data['name'],
            description=eval_data['description'],
            evaluation_prompt=eval_data['evaluation_prompt'],
            input_data=eval_data['input_data'],
            scoring_criteria=eval_data['scoring_criteria'],
            api_name=model,
            api_key=api_key,
        )
        tasks.append(task)

    # Run evaluations in parallel
    if parallel > 1:
        # Batch into smaller groups to avoid overwhelming the API
        for i in range(0, len(tasks), parallel):
            batch_tasks = tasks[i:i+parallel]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
    else:
        # Run sequentially
        for task in tasks:
            result = await task
            results.append(result)

    return results


def _save_results(results: list[dict[str, Any]], benchmark_name: str, output_path: str):
    """Save evaluation results to file."""
    scores = [r["score"] for r in results if r.get("score") is not None]  # an unscored item is in no average
    output_data = {
        "benchmark": benchmark_name,
        "num_samples": len(results),
        "results": results,
        "summary": {
            "total": len(results),
            "unscored": len(results) - len(scores),
            "average_score": sum(scores) / len(scores) if scores else 0
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)


# Add the benchmark group to the main CLI
def register_commands(cli):
    """Register benchmark commands with the main CLI."""
    cli.add_command(benchmark_group)
