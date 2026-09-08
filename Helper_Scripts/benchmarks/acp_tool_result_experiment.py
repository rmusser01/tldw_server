"""Replay the ACP result policy; default worker is a labeled fixture oracle.

Run from the repository root with ``python -m Helper_Scripts.benchmarks.acp_tool_result_experiment``.
This measures result selection and exact recovery, not completed-agent quality.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import time
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_llm_caller import LLMCaller, LLMResponse
from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_result_context import (
    ToolResultContext,
    ToolResultPolicy,
)


class FixtureOracle(LLMCaller):
    """Select ground-truth segments to exercise machinery, not simulate quality."""

    def __init__(self, expected_quote: str) -> None:
        self.expected_quote = expected_quote

    async def call(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        payload = json.loads(messages[-1]["content"])
        segments = payload["segments"]
        text = "".join(segment["text"] for segment in segments)
        start = text.index(self.expected_quote)
        end = start + len(self.expected_quote)
        offset, selected = 0, []
        for segment in segments:
            next_offset = offset + len(segment["text"])
            if offset < end and next_offset > start:
                selected.append(segment["id"])
            offset = next_offset
        return LLMResponse(text=json.dumps({"segment_ids": selected}))


def sample_cases() -> list[dict[str, str]]:
    """Small fixed fixtures with independently specified expected evidence."""
    return [
        {
            "id": "late-answer",
            "question": "Which port serves metrics?",
            "text": "Unrelated build and deployment details.\n" * 300 + "Metrics listen on port 9191.\n",
            "expected_quote": "Metrics listen on port 9191.",
        },
        {
            "id": "unicode-transcript",
            "question": "What decision was made about the launch?",
            "text": "[00:00] 背景の説明と議論。\n" * 300 + "[42:10] Decision: postpone the launch until Friday.\n",
            "expected_quote": "Decision: postpone the launch until Friday.",
        },
        {
            "id": "scripted-recovery",
            "question": "Describe zephyr.",
            "text": "Unrelated background. " * 700 + "Launch code is amber.",
            "expected_quote": "Launch code is amber.",
        },
    ]


async def compare_case(case: dict[str, str], worker: LLMCaller | None = None) -> list[dict[str, Any]]:
    """Compare all three policies, reporting actual sizes and scripted recovery."""
    text, question, expected = case["text"], case["question"], case["expected_quote"]
    if not expected or expected not in text:
        raise ValueError("expected_quote must be a nonempty exact quote from the source")
    records = []
    for mode in ("off", "excerpt", "worker"):
        caller = worker if worker is not None else FixtureOracle(expected)
        context = ToolResultContext(ToolResultPolicy(mode=mode), worker=caller if mode == "worker" else None)
        started = time.perf_counter()
        result = await context.prepare(
            tool_name="fixture_search",
            arguments={"query": question},
            text=text,
            question=question,
            cancel_event=asyncio.Event(),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        evidence_present = expected in result.output
        recovered = evidence_present
        reread_count, reread_bytes = 0, 0
        source_id = result.metadata.get("source_id")
        if not recovered and source_id:
            position = text.index(expected)
            target_end = position + len(expected)
            recovered_text = ""
            while position < target_end:
                reread = context.read(source_id, offset=position, limit=min(512, target_end - position))
                reread_count += 1
                reread_bytes += len(reread.output.encode("utf-8"))
                ranges = reread.metadata["ranges"]
                if not ranges:
                    break
                start, end = ranges[0]
                label = f"\n[{start}:{end}]\n"
                _prefix, marker, body = reread.output.partition(label)
                snippet = body[: end - start]
                if not marker or start != position or not start < end <= target_end or snippet != text[start:end]:
                    break
                recovered_text += snippet
                position = end
            recovered = recovered_text == expected
        records.append(
            {
                "case_id": case["id"],
                "mode": mode,
                "outcome": result.metadata["outcome"],
                "worker_kind": ("injected" if worker is not None else "fixture_oracle") if mode == "worker" else None,
                "input_bytes": len(text.encode("utf-8")),
                "output_bytes": len(result.output.encode("utf-8")),
                "policy_latency_ms": elapsed_ms,
                "evidence_present": evidence_present,
                "evidence_recovered": recovered,
                "scripted_reread_count": reread_count,
                "scripted_reread_bytes": reread_bytes,
                "worker_usage": result.metadata.get("worker_usage"),
                "worker_called": result.metadata.get("worker_called", False),
                "worker_request_bytes": result.metadata.get("worker_request_bytes"),
                "worker_response_bytes": result.metadata.get("worker_response_bytes"),
                "worker_latency_ms": result.metadata.get("worker_latency_ms"),
                "main_model_usage": None,
            }
        )
    return records


async def _run(cases: list[dict[str, str]], worker: LLMCaller | None) -> dict[str, Any]:
    results = []
    for case in cases:
        results.extend(await compare_case(case, worker))
    return {
        "scope": "tool_result_policy_replay",
        "worker_kind": "injected" if worker is not None else "fixture_oracle",
        "measures_task_success": False,
        "results": results,
        "limitations": (
            "Scripted evidence recovery uses known source offsets. No main model is invoked. "
            "Missing usage is unknown. The fixture oracle uses ground truth and is not a model quality benchmark. "
            "Use representative full agent runs to measure total cost, latency, task success, and extra reads."
        ),
    }


def main(argv: list[str] | None = None) -> None:
    """Print JSON; real worker calls require an explicit trusted caller factory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="JSONL with id, question, text, expected_quote")
    parser.add_argument("--worker-factory", help="Trusted local module:factory returning a configured LLMCaller")
    args = parser.parse_args(argv)
    cases = sample_cases()
    if args.dataset:
        cases = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    worker = None
    if args.worker_factory:
        module_name, separator, factory_name = args.worker_factory.partition(":")
        if not separator or not module_name or not factory_name:
            parser.error("--worker-factory must be module:factory")
        worker = getattr(importlib.import_module(module_name), factory_name)()
        if not isinstance(worker, LLMCaller):
            parser.error("worker factory must return an LLMCaller")
    print(json.dumps(asyncio.run(_run(cases, worker)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
