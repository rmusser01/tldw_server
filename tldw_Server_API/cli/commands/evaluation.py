"""
Evaluation execution commands for tldw Evaluations CLI.

Provides command-line access to all evaluation types including
G-Eval, RAG evaluation, response quality assessment, and batch processing.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional

import click
from loguru import logger

from tldw_Server_API.app.core.Chat.chat_orchestrator import chat_api_call
from tldw_Server_API.cli.utils.output import (
    format_timestamp,
    print_error,
    print_evaluation_results,
    print_info,
    print_json,
    print_progress_bar,
    print_success,
    print_table,
)

_EVAL_CLI_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    json.JSONDecodeError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    click.ClickException,
)


@click.group()
def eval_group():
    """Evaluation execution commands."""
    pass


@eval_group.command('geval')
@click.argument('text', required=True)
@click.argument('summary', required=True)
@click.option('--provider', default='openai', help='LLM provider to use')
@click.option('--model', help='Specific model to use')
@click.option('--api-key', help='API key (overrides config)')
@click.option('--save/--no-save', default=True, help='Save evaluation results')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def geval(ctx, text, summary, provider, model, api_key, save, output_format):
    """
    Run G-Eval summarization assessment.

    Evaluates the quality of a summary against the original text using
    G-Eval methodology with multiple criteria including relevance,
    consistency, fluency, and coherence.

    Arguments:
        TEXT    Original text to evaluate against
        SUMMARY Summary text to evaluate

    Examples:
        tldw-evals eval geval "Original text..." "Summary text..."
        tldw-evals eval geval "file://input.txt" "file://summary.txt"
        tldw-evals eval geval "text" "summary" --provider anthropic --no-save
    """
    cli_context = ctx.obj['cli_context']

    try:
        cli_context.load_config()

        # Load text from files if needed
        text_content = _load_text_content(text)
        summary_content = _load_text_content(summary)

        if not cli_context.quiet:
            print_info(f"Running G-Eval assessment with {provider}...")

        # Run evaluation
        result = _run_geval_assessment(
            text_content, summary_content, provider, model, api_key, save, cli_context.config
        )

        # Display results
        if output_format == 'json':
            print_json(result, "G-Eval Results")
        else:
            print_evaluation_results(result)

        print_success("G-Eval assessment completed successfully")

    except _EVAL_CLI_NONCRITICAL_EXCEPTIONS as e:
        logger.exception("G-Eval assessment failed")
        print_error(f"G-Eval assessment failed: {e}")
        sys.exit(1)


@eval_group.command('rag')
@click.argument('query', required=True)
@click.argument('context', required=True)
@click.argument('response', required=True)
@click.option('--provider', default='openai', help='LLM provider to use')
@click.option('--model', help='Specific model to use')
@click.option('--api-key', help='API key (overrides config)')
@click.option('--save/--no-save', default=True, help='Save evaluation results')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def rag_eval(ctx, query, context, response, provider, model, api_key, save, output_format):
    """
    Run RAG (Retrieval-Augmented Generation) evaluation.

    Evaluates RAG system performance by assessing context relevance,
    answer faithfulness, answer relevance, and overall quality.

    Arguments:
        QUERY    User query/question
        CONTEXT  Retrieved context/documents
        RESPONSE Generated response/answer

    Examples:
        tldw-evals eval rag "What is AI?" "Context about AI..." "AI response..."
        tldw-evals eval rag "file://query.txt" "file://context.txt" "file://response.txt"
    """
    cli_context = ctx.obj['cli_context']

    try:
        cli_context.load_config()

        # Load content from files if needed
        query_content = _load_text_content(query)
        context_content = _load_text_content(context)
        response_content = _load_text_content(response)

        if not cli_context.quiet:
            print_info(f"Running RAG evaluation with {provider}...")

        # Run evaluation
        result = _run_rag_evaluation(
            query_content, context_content, response_content,
            provider, model, api_key, save, cli_context.config
        )

        # Display results
        if output_format == 'json':
            print_json(result, "RAG Evaluation Results")
        else:
            print_evaluation_results(result)

        print_success("RAG evaluation completed successfully")

    except _EVAL_CLI_NONCRITICAL_EXCEPTIONS as e:
        logger.exception("RAG evaluation failed")
        print_error(f"RAG evaluation failed: {e}")
        sys.exit(1)


@eval_group.command('quality')
@click.argument('text', required=True)
@click.argument('response', required=True)
@click.option('--provider', default='openai', help='LLM provider to use')
@click.option('--model', help='Specific model to use')
@click.option('--api-key', help='API key (overrides config)')
@click.option('--criteria', multiple=True, help='Specific quality criteria to evaluate')
@click.option('--save/--no-save', default=True, help='Save evaluation results')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def quality_eval(ctx, text, response, provider, model, api_key, criteria, save, output_format):
    """
    Run response quality evaluation.

    Evaluates response quality across multiple dimensions including
    accuracy, completeness, clarity, relevance, and helpfulness.

    Arguments:
        TEXT     Original text/prompt
        RESPONSE Response to evaluate

    Examples:
        tldw-evals eval quality "Question..." "Answer..."
        tldw-evals eval quality "text" "response" --criteria accuracy --criteria clarity
    """
    cli_context = ctx.obj['cli_context']

    try:
        cli_context.load_config()

        # Load content from files if needed
        text_content = _load_text_content(text)
        response_content = _load_text_content(response)

        if not cli_context.quiet:
            print_info(f"Running response quality evaluation with {provider}...")

        # Run evaluation
        result = _run_quality_evaluation(
            text_content, response_content, provider, model, api_key,
            list(criteria) if criteria else None, save, cli_context.config
        )

        # Display results
        if output_format == 'json':
            print_json(result, "Quality Evaluation Results")
        else:
            print_evaluation_results(result)

        print_success("Quality evaluation completed successfully")

    except _EVAL_CLI_NONCRITICAL_EXCEPTIONS as e:
        logger.exception("Quality evaluation failed")
        print_error(f"Quality evaluation failed: {e}")
        sys.exit(1)


@eval_group.command('batch')
@click.argument('input_file', type=click.Path(exists=True, readable=True, path_type=Path))
@click.option('--output', type=click.Path(path_type=Path), help='Output file for results')
@click.option('--provider', default='openai', help='Default LLM provider to use')
@click.option('--model', help='Default model to use')
@click.option('--api-key', help='API key (overrides config)')
@click.option('--parallel', type=int, default=1, help='Number of parallel evaluations')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def batch_eval(ctx, input_file, output, provider, model, api_key, parallel, output_format):
    """
    Run batch evaluations from input file.

    Processes multiple evaluations from a JSON or JSONL input file.
    Each line/entry should contain evaluation parameters.

    Input file format (JSONL):
        {"type": "geval", "text": "...", "summary": "...", "provider": "openai"}
        {"type": "rag", "query": "...", "context": "...", "response": "..."}
        {"type": "quality", "text": "...", "response": "...", "criteria": ["accuracy"]}

    Examples:
        tldw-evals eval batch evaluations.jsonl
        tldw-evals eval batch batch_input.json --output results.json --parallel 3
    """
    cli_context = ctx.obj['cli_context']

    try:
        cli_context.load_config()

        # Load batch input
        batch_data = _load_batch_input(input_file)

        if not cli_context.quiet:
            print_info(f"Processing {len(batch_data)} evaluations with {parallel} parallel workers...")

        # Run batch evaluation
        results = _run_batch_evaluation(
            batch_data, provider, model, api_key, parallel, cli_context.config, not cli_context.quiet
        )

        # Save results if output file specified
        if output:
            _save_batch_results(results, output)
            if not cli_context.quiet:
                print_success(f"Results saved to {output}")

        # Display summary
        if output_format == 'json':
            print_json(results, "Batch Evaluation Results")
        else:
            _display_batch_summary(results)

        print_success(f"Batch evaluation completed: {len(results)} evaluations processed")

    except _EVAL_CLI_NONCRITICAL_EXCEPTIONS as e:
        logger.exception("Batch evaluation failed")
        print_error(f"Batch evaluation failed: {e}")
        sys.exit(1)


@eval_group.command('custom')
@click.argument('metric_name', required=True)
@click.argument('text', required=True)
@click.option('--prompt', help='Custom evaluation prompt')
@click.option('--provider', default='openai', help='LLM provider to use')
@click.option('--model', help='Specific model to use')
@click.option('--api-key', help='API key (overrides config)')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def custom_eval(ctx, metric_name, text, prompt, provider, model, api_key, output_format):
    """
    Run custom metric evaluation.

    Evaluates text using a custom metric with optional custom prompt.
    Useful for domain-specific or specialized evaluation criteria.

    Arguments:
        METRIC_NAME Name of the custom metric
        TEXT        Text to evaluate

    Examples:
        tldw-evals eval custom "creativity" "Story text..." --prompt "Rate creativity 1-10"
        tldw-evals eval custom "technical_accuracy" "file://technical_doc.txt"
    """
    cli_context = ctx.obj['cli_context']

    try:
        cli_context.load_config()

        # Load text content
        text_content = _load_text_content(text)

        if not cli_context.quiet:
            print_info(f"Running custom evaluation '{metric_name}' with {provider}...")

        # Run evaluation
        result = _run_custom_evaluation(
            metric_name, text_content, prompt, provider, model, api_key, cli_context.config
        )

        # Display results
        if output_format == 'json':
            print_json(result, f"Custom Evaluation Results: {metric_name}")
        else:
            print_evaluation_results(result)

        if result['status'] == 'unscored':  # the judge call failed or its reply held no score: not a success
            print_error(f"Custom evaluation '{metric_name}' produced no score: {result['details'].get('explanation')}")
            sys.exit(1)
        print_success(f"Custom evaluation '{metric_name}' completed successfully")

    except _EVAL_CLI_NONCRITICAL_EXCEPTIONS as e:
        logger.exception("Custom evaluation failed")
        print_error(f"Custom evaluation failed: {e}")
        sys.exit(1)


@eval_group.command('label-choice')
@click.argument('question', required=True)
@click.option('--labels', '-l', multiple=True, required=True, help='Allowed labels (repeatable)')
@click.option('--context', default='', help='Optional context')
@click.option('--expected', default=None, help='Expected label for scoring')
@click.option('--provider', default='openai', help='LLM provider to use')
@click.option('--model', help='Specific model to use')
@click.option('--api-key', help='API key (overrides config)')
@click.option('--json-mode/--no-json-mode', default=True, help='Use native provider JSON mode when supported')
@click.option('--mapping', default=None, help='JSON string mapping aliases to canonical labels')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def label_choice(ctx, question, labels, context, expected, provider, model, api_key, json_mode, mapping, output_format):
    """
    Run label-choice evaluation (single sample).

    The model must return exactly one of the allowed labels.
    """
    cli_context = ctx.obj['cli_context']
    try:
        cli_context.load_config()

        mapping_dict = None
        if mapping:
            try:
                import json as _json
                mapping_dict = _json.loads(mapping)
            except _EVAL_CLI_NONCRITICAL_EXCEPTIONS as e:
                raise click.ClickException(f"Invalid mapping JSON: {e}") from e

        allowed = [str(l).strip() for l in labels]
        allowed_up = [l.upper() for l in allowed]

        system_prompt = "You return strict JSON only." if json_mode else "You must reply with exactly one label token."
        user_prompt = (
            f"Allowed labels: [{', '.join(allowed)}]\n"
            + (f"Context:\n{context}\n\n" if context else "")
            + f"Question:\n{question}\n\n"
            + ("Respond with exactly: {\"label\": \"<one of the allowed labels>\"}" if json_mode else "Answer (one token):")
        )

        messages = [{"role": "user", "content": user_prompt}]

        resp = chat_api_call(
            api_endpoint=(provider or 'openai').lower(),
            messages_payload=messages,
            system_message=system_prompt,
            response_format={"type": "json_object"} if json_mode else None,
            temp=0.0,
            max_tokens=16,
            model=model
        )

        def _extract_content(r):
            if isinstance(r, str):
                return r
            if isinstance(r, dict):
                ch = r.get('choices')
                if ch:
                    msg = ch[0].get('message', {})
                    content = msg.get('content')
                    if isinstance(content, list):
                        return ''.join(part.get('text', '') if isinstance(part, dict) else str(part) for part in content)
                    return content if isinstance(content, str) else str(content)
                if 'text' in r:
                    return str(r['text'])
            return str(r)

        raw = _extract_content(resp)
        parsed = None
        if json_mode:
            import json as _json
            try:
                obj = _json.loads(raw)
                if isinstance(obj, dict):
                    parsed = obj.get('label')
            except _EVAL_CLI_NONCRITICAL_EXCEPTIONS:
                parsed = None
        if parsed is None:
            up = str(raw).strip().upper()
            for lab in allowed_up:
                if lab in up or up.strip() == lab:
                    parsed = lab
                    break

        def norm_label(x: Optional[str]) -> Optional[str]:
            if x is None:
                return None
            s = str(x).strip().upper()
            if mapping_dict and s in {k.upper(): v for k, v in mapping_dict.items()}:
                s = {k.upper(): v.upper() for k, v in mapping_dict.items()}[s]
            return s

        pred_norm = norm_label(parsed)
        gold_norm = norm_label(expected) if expected else None
        accuracy = 1.0 if (gold_norm is not None and pred_norm == gold_norm) else 0.0 if gold_norm is not None else None

        result = {
            'evaluation_type': 'label_choice',
            'provider': provider,
            'model': model,
            'allowed_labels': allowed,
            'prediction': pred_norm,
            'expected': gold_norm,
            'accuracy': accuracy,
            'raw': raw
        }

        if output_format == 'json':
            print_json(result, 'Label Choice Result')
        else:
            rows = [
                {'Field': 'Prediction', 'Value': pred_norm},
                {'Field': 'Expected', 'Value': gold_norm},
                {'Field': 'Accuracy', 'Value': accuracy if accuracy is not None else 'N/A'},
            ]
            print_table(rows, 'Label Choice Result')
            if not cli_context.quiet:
                print_info(f"Allowed: {', '.join(allowed)}")

        print_success('Label-choice evaluation completed')

    except _EVAL_CLI_NONCRITICAL_EXCEPTIONS as e:
        logger.exception('Label-choice evaluation failed')
        print_error(f"Label-choice evaluation failed: {e}")
        sys.exit(1)


@eval_group.command('nli-factcheck')
@click.argument('claim', required=True)
@click.argument('evidence', required=True)
@click.option('--labels', '-l', multiple=True, help='Allowed labels (repeatable); default SUPPORTED, REFUTED, NEI')
@click.option('--expected', default=None, help='Expected label for scoring')
@click.option('--provider', default='openai', help='LLM provider to use')
@click.option('--model', help='Specific model to use')
@click.option('--api-key', help='API key (overrides config)')
@click.option('--json-mode/--no-json-mode', default=True, help='Use native provider JSON mode when supported')
@click.option('--mapping', default=None, help='JSON string mapping aliases to canonical labels')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def nli_factcheck(ctx, claim, evidence, labels, expected, provider, model, api_key, json_mode, mapping, output_format):
    """
    Run NLI-style factuality check (single sample).

    Labels typically: SUPPORTED, REFUTED, NEI.
    """
    cli_context = ctx.obj['cli_context']
    try:
        cli_context.load_config()

        mapping_dict = None
        if mapping:
            try:
                import json as _json
                mapping_dict = _json.loads(mapping)
            except _EVAL_CLI_NONCRITICAL_EXCEPTIONS as e:
                raise click.ClickException(f"Invalid mapping JSON: {e}") from e

        allowed = [str(l).strip() for l in labels] if labels else ["SUPPORTED", "REFUTED", "NEI"]
        allowed_up = [l.upper() for l in allowed]

        system_prompt = "You return strict JSON only." if json_mode else "You must reply with exactly one label token."
        user_prompt = (
            f"Allowed labels: [{', '.join(allowed)}]\n"
            f"Evidence:\n{_load_text_content(evidence)}\n\nClaim:\n{_load_text_content(claim)}\n\n"
            + ("Respond with exactly: {\"label\": \"<one of the allowed labels>\"}" if json_mode else "Answer (one token):")
        )

        messages = [{"role": "user", "content": user_prompt}]

        resp = chat_api_call(
            api_endpoint=(provider or 'openai').lower(),
            messages_payload=messages,
            system_message=system_prompt,
            response_format={"type": "json_object"} if json_mode else None,
            temp=0.0,
            max_tokens=16,
            model=model
        )

        def _extract_content(r):
            if isinstance(r, str):
                return r
            if isinstance(r, dict):
                ch = r.get('choices')
                if ch:
                    msg = ch[0].get('message', {})
                    content = msg.get('content')
                    if isinstance(content, list):
                        return ''.join(part.get('text', '') if isinstance(part, dict) else str(part) for part in content)
                    return content if isinstance(content, str) else str(content)
                if 'text' in r:
                    return str(r['text'])
            return str(r)

        raw = _extract_content(resp)
        parsed = None
        if json_mode:
            import json as _json
            try:
                obj = _json.loads(raw)
                if isinstance(obj, dict):
                    parsed = obj.get('label')
            except _EVAL_CLI_NONCRITICAL_EXCEPTIONS:
                parsed = None
        if parsed is None:
            up = str(raw).strip().upper()
            for lab in allowed_up:
                if lab in up or up.strip() == lab:
                    parsed = lab
                    break

        # Alias normalization
        base_alias = {
            "TRUE": "SUPPORTED",
            "ENTAILMENT": "SUPPORTED",
            "FALSE": "REFUTED",
            "CONTRADICTION": "REFUTED",
            "NEUTRAL": "NEI",
            "NOT_ENTAILED": "NEI",
            "NOT_ENTAILMENT": "NEI",
        }

        def norm_label(x: Optional[str]) -> Optional[str]:
            if x is None:
                return None
            s = str(x).strip().upper()
            if s in base_alias:
                s = base_alias[s]
            if mapping_dict and s in {k.upper(): v for k, v in mapping_dict.items()}:
                s = {k.upper(): v.upper() for k, v in mapping_dict.items()}[s]
            return s

        pred_norm = norm_label(parsed)
        gold_norm = norm_label(expected) if expected else None
        accuracy = 1.0 if (gold_norm is not None and pred_norm == gold_norm) else 0.0 if gold_norm is not None else None

        result = {
            'evaluation_type': 'nli_factcheck',
            'provider': provider,
            'model': model,
            'allowed_labels': allowed,
            'prediction': pred_norm,
            'expected': gold_norm,
            'accuracy': accuracy,
            'raw': raw
        }

        if output_format == 'json':
            print_json(result, 'NLI Fact-Check Result')
        else:
            rows = [
                {'Field': 'Prediction', 'Value': pred_norm},
                {'Field': 'Expected', 'Value': gold_norm},
                {'Field': 'Accuracy', 'Value': accuracy if accuracy is not None else 'N/A'},
            ]
            print_table(rows, 'NLI Fact-Check Result')
            if not cli_context.quiet:
                print_info(f"Allowed: {', '.join(allowed)}")

        print_success('NLI fact-check evaluation completed')

    except _EVAL_CLI_NONCRITICAL_EXCEPTIONS as e:
        logger.exception('NLI fact-check evaluation failed')
        print_error(f"NLI fact-check evaluation failed: {e}")
        sys.exit(1)


def _load_text_content(input_str: str) -> str:
    """Load text content from string or file."""
    if input_str.startswith('file://'):
        file_path = Path(input_str[7:])  # Remove 'file://' prefix
        if not file_path.exists():
            raise click.ClickException(f"File not found: {file_path}")

        try:
            return file_path.read_text(encoding='utf-8')
        except _EVAL_CLI_NONCRITICAL_EXCEPTIONS as e:
            raise click.ClickException(f"Failed to read file {file_path}: {e}") from e
    else:
        return input_str


def _run_geval_assessment(text: str, summary: str, provider: str, model: str,
                         api_key: str, save: bool, config: dict[str, Any]) -> dict[str, Any]:
    """Run G-Eval assessment."""
    try:
        from tldw_Server_API.app.core.Evaluations.ms_g_eval import run_geval

        # Extract API key from config if not provided
        if not api_key:
            api_key = _get_api_key_from_config(config, provider)

        result = run_geval(
            transcript=text,
            summary=summary,
            api_key=api_key,
            api_name=provider,
            save=save
        )

        return {
            'evaluation_type': 'geval',
            'evaluation_id': result.get('evaluation_id'),
            'provider': provider,
            'model': model,
            'timestamp': format_timestamp(None),
            'metrics': result.get('metrics', {}),
            'details': result
        }

    except _EVAL_CLI_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"G-Eval execution failed: {e}")
        raise


def _run_rag_evaluation(query: str, context: str, response: str, provider: str,
                       model: str, api_key: str, save: bool, config: dict[str, Any]) -> dict[str, Any]:
    """Run RAG evaluation."""
    try:
        from tldw_Server_API.app.core.Evaluations.rag_evaluator import RAGEvaluator

        # Extract API key from config if not provided
        if not api_key:
            api_key = _get_api_key_from_config(config, provider)

        evaluator = RAGEvaluator(api_key=api_key, api_name=provider)

        result = asyncio.run(evaluator.evaluate(
            query=query,
            contexts=[context],  # RAGEvaluator expects a list
            response=response,
            save_results=save
        ))

        return {
            'evaluation_type': 'rag',
            'evaluation_id': result.get('evaluation_id'),
            'provider': provider,
            'model': model,
            'timestamp': format_timestamp(None),
            'metrics': result.get('metrics', {}),
            'details': result
        }

    except _EVAL_CLI_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"RAG evaluation execution failed: {e}")
        raise


def _run_quality_evaluation(text: str, response: str, provider: str, model: str,
                           api_key: str, criteria: Optional[list[str]], save: bool,
                           config: dict[str, Any]) -> dict[str, Any]:
    """Run response quality evaluation."""
    try:
        from tldw_Server_API.app.core.Evaluations.response_quality_evaluator import ResponseQualityEvaluator

        # Extract API key from config if not provided
        if not api_key:
            api_key = _get_api_key_from_config(config, provider)

        evaluator = ResponseQualityEvaluator(api_key=api_key, api_name=provider)

        result = asyncio.run(evaluator.evaluate(
            original_text=text,
            response_text=response,
            criteria=criteria,
            save_results=save
        ))

        return {
            'evaluation_type': 'quality',
            'evaluation_id': result.get('evaluation_id'),
            'provider': provider,
            'model': model,
            'timestamp': format_timestamp(None),
            'metrics': result.get('metrics', {}),
            'details': result
        }

    except _EVAL_CLI_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Quality evaluation execution failed: {e}")
        raise


def _run_custom_evaluation(metric_name: str, text: str, prompt: Optional[str], provider: str,
                          model: str, api_key: str, config: dict[str, Any]) -> dict[str, Any]:
    """Run custom metric evaluation."""
    try:
        from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager

        # Extract API key from config if not provided
        if not api_key:
            api_key = _get_api_key_from_config(config, provider)

        eval_manager = EvaluationManager()

        result = asyncio.run(eval_manager.evaluate_custom_metric(
            metric_name=metric_name,
            description=metric_name,
            evaluation_prompt=prompt or f"Evaluate the text below on the metric '{metric_name}'.\n\n{{text}}",
            input_data={'text': text},
            scoring_criteria={metric_name: prompt or 'Overall quality on this metric'},
            api_name=provider,
            api_key=api_key,
        ))
        score = result.get('score')  # None: no score, kept apart from a 0.0 (TASK-15485)

        return {
            'evaluation_type': 'custom',
            'status': 'unscored' if score is None else 'completed',
            'metric_name': metric_name,
            'provider': provider,
            'model': model,
            'timestamp': format_timestamp(None),
            'metrics': {metric_name: {'score': score, 'explanation': result.get('explanation', '')}},
            'details': result
        }

    except _EVAL_CLI_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Custom evaluation execution failed: {e}")
        raise


def _load_batch_input(input_file: Path) -> list[dict[str, Any]]:
    """Load batch input from file."""
    try:
        content = input_file.read_text(encoding='utf-8')

        # Try JSON first
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
            else:
                return [data]
        except json.JSONDecodeError:
            pass

        # Try JSONL
        batch_data = []
        for line_num, line in enumerate(content.strip().split('\n'), 1):
            if line.strip():
                try:
                    batch_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise click.ClickException(f"Invalid JSON on line {line_num}: {e}") from e

        return batch_data

    except _EVAL_CLI_NONCRITICAL_EXCEPTIONS as e:
        raise click.ClickException(f"Failed to load batch input file: {e}") from e


def _run_batch_evaluation(batch_data: list[dict[str, Any]], default_provider: str,
                         default_model: str, default_api_key: str, parallel: int,
                         config: dict[str, Any], show_progress: bool) -> list[dict[str, Any]]:
    """Run batch evaluation with progress tracking."""
    results = []

    if show_progress:
        with print_progress_bar(len(batch_data), "Evaluating") as progress:
            task = progress.add_task("Processing evaluations...", total=len(batch_data))

            for i, eval_spec in enumerate(batch_data):
                try:
                    result = _run_single_evaluation(eval_spec, default_provider, default_model, default_api_key, config)
                    results.append(result)
                except _EVAL_CLI_NONCRITICAL_EXCEPTIONS as e:
                    logger.error(f"Evaluation {i+1} failed: {e}")
                    results.append({
                        'evaluation_index': i + 1,
                        'status': 'failed',
                        'error': str(e),
                        'input': eval_spec
                    })

                progress.update(task, advance=1)
    else:
        for i, eval_spec in enumerate(batch_data):
            try:
                result = _run_single_evaluation(eval_spec, default_provider, default_model, default_api_key, config)
                results.append(result)
            except _EVAL_CLI_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Evaluation {i+1} failed: {e}")
                results.append({
                    'evaluation_index': i + 1,
                    'status': 'failed',
                    'error': str(e),
                    'input': eval_spec
                })

    return results


def _load_items_file(path: Path) -> list[dict[str, Any]]:
    """Load items for OCR evaluation from JSON or JSONL file."""
    try:
        content = path.read_text(encoding='utf-8')
        try:
            data = json.loads(content)
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            items = []
            for line in content.strip().split('\n'):
                if not line.strip():
                    continue
                items.append(json.loads(line))
            return items
    except _EVAL_CLI_NONCRITICAL_EXCEPTIONS as e:
        raise click.ClickException(f"Failed to load items file {path}: {e}") from e


def _run_single_evaluation(eval_spec: dict[str, Any], default_provider: str,
                          default_model: str, default_api_key: str, config: dict[str, Any]) -> dict[str, Any]:
    """Run a single evaluation from specification."""
    eval_type = eval_spec.get('type')
    provider = eval_spec.get('provider', default_provider)
    model = eval_spec.get('model', default_model)
    api_key = eval_spec.get('api_key', default_api_key)
    save = eval_spec.get('save', True)

    if eval_type == 'geval':
        return _run_geval_assessment(
            eval_spec['text'], eval_spec['summary'],
            provider, model, api_key, save, config
        )
    elif eval_type == 'rag':
        return _run_rag_evaluation(
            eval_spec['query'], eval_spec['context'], eval_spec['response'],
            provider, model, api_key, save, config
        )
    elif eval_type == 'quality':
        return _run_quality_evaluation(
            eval_spec['text'], eval_spec['response'],
            provider, model, api_key, eval_spec.get('criteria'), save, config
        )
    elif eval_type == 'custom':
        return _run_custom_evaluation(
            eval_spec['metric_name'], eval_spec['text'],
            eval_spec.get('prompt'), provider, model, api_key, config
        )
    else:
        raise ValueError(f"Unknown evaluation type: {eval_type}")


def _save_batch_results(results: list[dict[str, Any]], output_file: Path):
    """Save batch results to file."""
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)

    except _EVAL_CLI_NONCRITICAL_EXCEPTIONS as e:
        raise click.ClickException(f"Failed to save results to {output_file}: {e}") from e


def _display_batch_summary(results: list[dict[str, Any]]):
    """Display batch evaluation summary."""
    unscored = sum(1 for r in results if r.get('status') == 'unscored')
    successful = sum(1 for r in results if r.get('status') not in ('failed', 'unscored'))
    failed = len(results) - successful - unscored

    summary_data = [
        {'Metric': 'Total Evaluations', 'Count': len(results)},
        {'Metric': 'Successful', 'Count': successful},
        {'Metric': 'Failed', 'Count': failed},
        {'Metric': 'Unscored', 'Count': unscored},
        {'Metric': 'Success Rate', 'Count': f"{successful/len(results)*100:.1f}%" if results else "0%"}
    ]

    print_table(summary_data, "Batch Evaluation Summary")

    if failed > 0:
        print_info(f"Check logs for details on {failed} failed evaluation(s)")


def _get_api_key_from_config(config: dict[str, Any], provider: str) -> Optional[str]:
    """Extract API key for provider from config."""
    # This is a simplified version - in reality, you'd access the config properly
    # Based on how the main config system works
    provider_key_map = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'google': 'GOOGLE_API_KEY',
        'cohere': 'COHERE_API_KEY',
        'groq': 'GROQ_API_KEY'
    }

    key_name = provider_key_map.get(provider.lower())
    if key_name:
        # Try to get from config
        for _section, values in config.items():
            if isinstance(values, dict) and key_name in values:
                return values[key_name]

        # Try environment variable
        import os
        return os.getenv(key_name)

    return None
@eval_group.command('ocr')
@click.option('--items-file', type=click.Path(exists=True, readable=True, path_type=Path), help='JSON/JSONL with items: {id, extracted_text|pdf_path, ground_truth_text}')
@click.option('--pdf', 'pdfs', multiple=True, type=click.Path(exists=True, readable=True, path_type=Path), help='PDF file(s) to OCR')
@click.option('--ground-truths-file', type=click.Path(exists=True, readable=True, path_type=Path), help='JSON array of ground-truth texts (aligned to --pdf order)')
@click.option('--ground-truths-pages-file', type=click.Path(exists=True, readable=True, path_type=Path), help='JSON array of arrays of ground-truth page texts (aligned to --pdf order)')
@click.option('--metrics', multiple=True, type=click.Choice(['cer','wer','coverage','page_coverage']), help='Metrics to compute (repeatable)')
@click.option('--max-cer', type=float, default=None, help='Threshold: maximum CER')
@click.option('--max-wer', type=float, default=None, help='Threshold: maximum WER')
@click.option('--min-coverage', type=float, default=None, help='Threshold: minimum coverage')
@click.option('--min-page-coverage', type=float, default=None, help='Threshold: minimum page coverage')
@click.option('--ocr-backend', type=str, default=None, help='OCR backend (e.g., tesseract)')
@click.option('--ocr-lang', type=str, default='eng', help='OCR language')
@click.option('--ocr-dpi', type=int, default=300, help='OCR DPI for rendering (72-600)')
@click.option('--ocr-mode', type=click.Choice(['fallback','always']), default='fallback', help='OCR mode')
@click.option('--ocr-min-page-text-chars', type=int, default=40, help='Threshold per page to skip OCR when text present')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='json', help='Output format')
@click.pass_context
def ocr_eval(ctx, items_file, pdfs, ground_truths_file, ground_truths_pages_file, metrics, max_cer, max_wer, min_coverage, min_page_coverage,
             ocr_backend, ocr_lang, ocr_dpi, ocr_mode, ocr_min_page_text_chars, output_format):
    """
    Run OCR evaluation on extracted text or PDF files.

    Examples:
      tldw-evals eval ocr --items-file items.jsonl
      tldw-evals eval ocr --pdf a.pdf --pdf b.pdf --ground-truths-file gts.json
    """
    cli_context = ctx.obj['cli_context']
    try:
        cli_context.load_config()

        items: list[dict[str, Any]] = []
        if items_file:
            items = _load_items_file(items_file)
        elif pdfs:
            gts = []
            if ground_truths_file:
                try:
                    gts = json.loads(Path(ground_truths_file).read_text(encoding='utf-8'))
                    if not isinstance(gts, list):
                        raise click.ClickException('ground-truths-file must be a JSON array of strings')
                except _EVAL_CLI_NONCRITICAL_EXCEPTIONS as e:
                    raise click.ClickException(f'Failed to read ground-truths-file: {e}') from e
            gt_pages_all = None
            if ground_truths_pages_file:
                try:
                    gt_pages_all = json.loads(Path(ground_truths_pages_file).read_text(encoding='utf-8'))
                    if not isinstance(gt_pages_all, list):
                        gt_pages_all = None
                except _EVAL_CLI_NONCRITICAL_EXCEPTIONS as e:
                    raise click.ClickException(f'Failed to read ground-truths-pages-file: {e}') from e

            for idx, p in enumerate(pdfs):
                b = Path(p).read_bytes()
                gt = gts[idx] if idx < len(gts) else None
                item = { 'id': Path(p).name, 'pdf_bytes': b, 'ground_truth_text': gt }
                if gt_pages_all and idx < len(gt_pages_all) and isinstance(gt_pages_all[idx], list):
                    item['ground_truth_pages'] = gt_pages_all[idx]
                items.append(item)
        else:
            raise click.ClickException('Provide either --items-file or --pdf inputs')

        thresholds = {}
        if max_cer is not None:
            thresholds['max_cer'] = max_cer
        if max_wer is not None:
            thresholds['max_wer'] = max_wer
        if min_coverage is not None:
            thresholds['min_coverage'] = min_coverage
        if min_page_coverage is not None:
            thresholds['min_page_coverage'] = min_page_coverage
        if not thresholds:
            thresholds = None

        ocr_options = {
            'enable_ocr': True,
            'ocr_backend': ocr_backend,
            'ocr_lang': ocr_lang,
            'ocr_dpi': ocr_dpi,
            'ocr_mode': ocr_mode,
            'ocr_min_page_text_chars': ocr_min_page_text_chars,
        }

        # Run via unified service
        from tldw_Server_API.app.core.Evaluations.unified_evaluation_service import UnifiedEvaluationService
        service = UnifiedEvaluationService()
        result = asyncio.run(service.evaluate_ocr(
            items=items,
            metrics=list(metrics) if metrics else None,
            thresholds=thresholds,
            ocr_options=ocr_options,
            user_id='cli'
        ))

        if output_format == 'json':
            print_json(result, 'OCR Evaluation Results')
        else:
            # Minimal table: summary only
            summary = result.get('results', {}).get('summary') or result.get('summary') or {}
            rows = [
                ['Count', summary.get('count')],
                ['Avg CER', summary.get('avg_cer')],
                ['Avg WER', summary.get('avg_wer')],
                ['Avg Coverage', summary.get('avg_coverage')],
                ['Avg Page Coverage', summary.get('avg_page_coverage')],
                ['Pass Rate', summary.get('pass_rate')],
            ]
            print_table(['Metric','Value'], rows, title='OCR Evaluation Summary')

        print_success('OCR evaluation completed successfully')
    except click.ClickException as e:
        print_error(str(e))
        sys.exit(2)
    except _EVAL_CLI_NONCRITICAL_EXCEPTIONS as e:
        logger.exception('OCR evaluation failed')
        print_error(f'OCR evaluation failed: {e}')
        sys.exit(1)
