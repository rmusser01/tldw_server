"""Trusted, stdlib-only single-operation moderation child.

Launch this file with ``sys.executable -I -S <absolute file>``. Resource limits
belong exclusively to main(), never to execute_request() or the parent process.
Linux requires address-space (512 MiB), CPU (3 s) and no-core backstops. macOS
uses CPU/no-core only: RLIMIT_AS is not a reliable address-space guarantee there.
Windows has no stdlib resource backstop; parent supervision remains mandatory.
"""

from __future__ import annotations

import contextlib
import os
import re
import struct
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

if __name__ == "__main__" and not __package__:
    # Bypass application package initializers and load only the trusted graph.
    root = Path(__file__).resolve().parents[4]
    for package in (
        "tldw_Server_API",
        "tldw_Server_API.app",
        "tldw_Server_API.app.core",
        "tldw_Server_API.app.core.Moderation",
        "tldw_Server_API.app.core.Web_Scraping",
    ):
        module = ModuleType(package)
        module.__path__ = [str(root.joinpath(*package.split(".")))]
        sys.modules[package] = module

from tldw_Server_API.app.core.exception_types import ModerationExecutionError
from tldw_Server_API.app.core.Moderation.execution_models import (
    ExecutionBounds,
    OutputBudget,
    RegexRuleSnapshot,
    SubstitutionBatchResult,
    SubstitutionRuleSnapshot,
    WorkerRequest,
    WorkerResponse,
)
from tldw_Server_API.app.core.Moderation.execution_protocol import decode_request, encode_request, encode_response
from tldw_Server_API.app.core.Moderation.models import ModerationPolicy, PatternRule
from tldw_Server_API.app.core.Moderation.policy_evaluator import EvaluationLimits, PolicyEvaluator
from tldw_Server_API.app.core.Moderation.supervised_evaluator import evaluate_supervised
from tldw_Server_API.app.core.Web_Scraping.safe_regex_worker import parse_replacement_template


def _reconstruct_policy(snapshot: dict) -> ModerationPolicy:
    rules = []
    for item in snapshot["block_patterns"]:
        pattern = re.compile(item["pattern"], item["flags"])
        if item["kind"] == "raw":
            rules.append(pattern)
        else:
            rules.append(
                PatternRule(
                    pattern,
                    item["action"],
                    item["replacement"],
                    set(item["categories"]) if item["categories"] is not None else None,
                    item["phase"],
                )
            )
    values = dict(snapshot, block_patterns=rules)
    categories = values["categories_enabled"]
    values["categories_enabled"] = set(categories) if categories is not None else None
    return ModerationPolicy(**values)


def _substitute(text: str, rules: list[SubstitutionRuleSnapshot], budget: OutputBudget) -> SubstitutionBatchResult:
    total = 0
    skipped = []
    for index, rule in enumerate(rules):
        try:
            pattern = re.compile(rule.pattern, rule.flags)
            parsed = parse_replacement_template(
                rule.replacement,
                group_count=pattern.groups,
                group_names=pattern.groupindex,
                dialect="stdlib",
                legacy_numeric_group_ids=sys.version_info < (3, 12),
            )
            if parsed is None:
                skipped.append(index)
                continue
            parts = []
            last = written = count = 0
            for match in pattern.finditer(text):
                start, end = match.span()
                expanded_size = parsed.literal_chars
                for group in parsed.references:
                    group_start, group_end = match.span(group)
                    if group_start >= 0:
                        expanded_size += group_end - group_start
                written += start - last + expanded_size
                budget.check(written)
                # Expansion is permitted only after group spans prove it fits.
                expanded = match.expand(parsed.execution_repl)
                parts.extend((text[last:start], expanded))
                last = end
                count += 1
            budget.check(written + len(text) - last)
            parts.append(text[last:])
            text = "".join(parts)
            total += count
        except (re.error, IndexError):
            skipped.append(index)
    budget.check(len(text))
    return SubstitutionBatchResult(text, total, tuple(skipped))


def execute_request(request: WorkerRequest) -> WorkerResponse:
    """Execute one already-bounded batch; direct unit tests must be benign."""
    try:
        # Defend direct callers too; no policy compilation occurs in this validation.
        encode_request(request)
        budget = OutputBudget(request.bounds.max_output_chars)
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            if request.operation == "supervised_evaluate":
                result = evaluate_supervised(
                    request.text,
                    [RegexRuleSnapshot(**rule) for rule in request.policy["rules"]],
                    request.phase,
                    output_budget=budget,
                )
            elif request.operation == "custom_substitute":
                result = _substitute(
                    request.text, [SubstitutionRuleSnapshot(**rule) for rule in request.policy["rules"]], budget
                )
            else:
                policy = _reconstruct_policy(request.policy)
                limits = EvaluationLimits(**request.limits)
                evaluator = PolicyEvaluator()
                if request.operation == "evaluate":
                    result = evaluator.evaluate_text(
                        request.text, policy, request.phase, limits, include_redacted_text=False, output_budget=budget
                    )
                elif request.operation == "redact":
                    result = evaluator.redact_text(request.text, policy, request.phase, limits, output_budget=budget)
                elif request.operation == "redact_many":
                    result = []
                    remaining = request.bounds.max_output_chars
                    for text in request.text:
                        redacted = evaluator.redact_text(
                            text, policy, request.phase, limits, output_budget=OutputBudget(remaining)
                        )
                        remaining -= len(redacted)
                        result.append(redacted)
                else:
                    result = evaluator.redact_text_with_count(
                        request.text, policy, request.phase, limits, output_budget=budget
                    )
            if isinstance(result, str):
                budget.check(len(result))
            elif isinstance(result, tuple):
                budget.check(len(result[0]))
        response = WorkerResponse(request.operation, result)
        if len(encode_response(response)) > request.bounds.max_response_bytes:
            raise ModerationExecutionError("size_exceeded")
        return response
    except ModerationExecutionError as error:
        return WorkerResponse(request.operation, reason=error.reason)
    except Exception:  # noqa: BLE001 - fail closed without serializing exception text.
        return WorkerResponse(request.operation, reason="worker_crashed")


def _apply_resource_limits() -> dict[str, bool]:
    """Install and verify supported child-only backstops before policy decoding."""
    applied = {"address_space": False, "cpu": False, "core": False}
    if os.name != "posix":
        return applied
    try:
        import resource

        limits = [("cpu", resource.RLIMIT_CPU, 3), ("core", resource.RLIMIT_CORE, 0)]
        if sys.platform.startswith("linux"):
            limits.insert(0, ("address_space", resource.RLIMIT_AS, 512 * 1024 * 1024))
        for name, resource_id, ceiling in limits:
            soft, hard = resource.getrlimit(resource_id)
            target = ceiling if hard == resource.RLIM_INFINITY else min(ceiling, hard)
            target = target if soft == resource.RLIM_INFINITY else min(target, soft)
            resource.setrlimit(resource_id, (target, target))
            verified, verified_hard = resource.getrlimit(resource_id)
            if (
                verified == resource.RLIM_INFINITY
                or verified_hard == resource.RLIM_INFINITY
                or verified > ceiling
                or verified_hard > ceiling
            ):
                raise ValueError("resource posture")
            applied[name] = True
    except (ImportError, AttributeError, OSError, ValueError):
        raise ModerationExecutionError("worker_unavailable") from None
    return applied


def main() -> int:
    """Read exactly one bounded frame and EOF; emit protocol only, never content logs."""
    operation = None
    posture = None
    response_cap = ExecutionBounds().max_response_bytes
    try:
        posture = _apply_resource_limits()
        header = sys.stdin.buffer.read(4)
        if len(header) != 4:
            raise ModerationExecutionError("invalid_response")
        length = struct.unpack(">I", header)[0]
        if length + 4 > ExecutionBounds().max_request_bytes:
            raise ModerationExecutionError("size_exceeded")
        payload = sys.stdin.buffer.read(length + 1)
        request = decode_request(header + payload)
        response_cap = request.bounds.max_response_bytes
        operation = request.operation
        response = replace(execute_request(request), resource_limits=posture)
        if len(encode_response(response)) > request.bounds.max_response_bytes:
            raise ModerationExecutionError("size_exceeded")
    except ModerationExecutionError as error:
        response = WorkerResponse(operation, reason=error.reason, resource_limits=posture)
    except Exception:  # noqa: BLE001 - stderr must not disclose policy, text or tracebacks.
        response = WorkerResponse(operation, reason="worker_crashed", resource_limits=posture)
    encoded = encode_response(response)
    if len(encoded) > response_cap:
        # A tightened cap may not even fit the sanitized failure envelope.
        return 1
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
