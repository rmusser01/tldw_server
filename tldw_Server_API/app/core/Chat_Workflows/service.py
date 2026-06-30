from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Chat_Workflows.question_renderer import (
    ChatWorkflowQuestionRenderer,
)
from tldw_Server_API.app.core.DB_Management.ChatWorkflows_DB import (
    ChatWorkflowsDatabase,
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_loads(value: str | None, *, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


class ChatWorkflowConflictError(ValueError):
    """Raised when a duplicate answer conflicts with the stored workflow state."""


class ChatWorkflowService:
    """Orchestrates chat workflow runs on top of the persistence adapter."""

    def __init__(
        self,
        *,
        db: ChatWorkflowsDatabase,
        question_renderer: ChatWorkflowQuestionRenderer | None,
        dialogue_orchestrator: Any | None = None,
    ) -> None:
        self.db = db
        self.question_renderer = question_renderer
        self.dialogue_orchestrator = dialogue_orchestrator

    async def _db_call_async(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Run a synchronous DB adapter method on a worker thread."""
        method = getattr(self.db, method_name)
        return await asyncio.to_thread(method, *args, **kwargs)

    def start_run(
        self,
        *,
        tenant_id: str,
        user_id: str,
        template: dict[str, Any],
        source_mode: str,
        selected_context_refs: list[dict[str, Any]] | list[Any],
        question_renderer_model: str | None = None,
    ) -> dict[str, Any]:
        template_snapshot = self._normalize_template(template)
        if not template_snapshot["steps"]:
            raise ValueError("template must contain at least one step")
        template_id = template.get("id")
        if template_id is not None and self.db.get_template(int(template_id)) is None:
            template_id = None

        run_id = self.db.create_run(
            tenant_id=tenant_id,
            user_id=user_id,
            template_id=template_id,
            template_version=int(template_snapshot.get("version", 1)),
            source_mode=source_mode,
            status="active",
            template_snapshot=template_snapshot,
            selected_context_refs=selected_context_refs,
            resolved_context_snapshot=[],
            question_renderer_model=question_renderer_model,
        )
        self.db.append_event(
            run_id,
            "run_started",
            {
                "source_mode": source_mode,
                "step_count": len(template_snapshot["steps"]),
            },
        )
        return self._require_run(run_id)

    def generate_draft(
        self,
        *,
        goal: str,
        base_question: str | None = None,
        desired_step_count: int = 4,
    ) -> dict[str, Any]:
        normalized_goal = goal.strip()
        if not normalized_goal:
            raise ValueError("goal must not be empty")

        seed_questions = [
            base_question or f"What outcome do you want from {normalized_goal}?",
            f"What constraints should shape {normalized_goal}?",
            f"What context or assets already exist for {normalized_goal}?",
            f"What risks or unknowns could block {normalized_goal}?",
        ]
        while len(seed_questions) < desired_step_count:
            seed_questions.append(f"What else should be clarified about {normalized_goal}?")

        steps: list[dict[str, Any]] = []
        for step_index in range(desired_step_count):
            steps.append(
                {
                    "id": f"step-{step_index + 1}",
                    "step_index": step_index,
                    "label": f"Step {step_index + 1}",
                    "base_question": seed_questions[step_index],
                    "question_mode": "stock",
                    "context_refs": [],
                }
            )

        return {
            "title": normalized_goal,
            "description": f"Generated workflow for {normalized_goal}",
            "version": 1,
            "steps": steps,
        }

    async def get_current_step(self, run_id: str) -> dict[str, Any] | None:
        run = await self._require_run_async(run_id)
        template_snapshot = self._get_template_snapshot(run)
        current_step_index = int(run["current_step_index"])
        steps = template_snapshot["steps"]
        if current_step_index >= len(steps):
            return None

        step = steps[current_step_index]
        if self._get_step_type(step) == "dialogue_round_step":
            current_prompt = self._get_dialogue_prompt(step=step, run=run)
            return {
                **step,
                "run_id": run_id,
                "step_index": current_step_index,
                "displayed_question": current_prompt,
                "current_prompt": current_prompt,
                "current_round_index": int(run.get("active_round_index", 0)),
                "rounds": await self._db_call_async("list_rounds", run_id, current_step_index),
            }

        cached_question = await self._get_rendered_question_cache_async(
            run_id,
            current_step_index,
        )
        if cached_question is None:
            cached_question = await self._render_question_async(
                run_id=run_id,
                step_index=current_step_index,
                step=step,
                prior_answers=await self._db_call_async("list_answers", run_id),
                context_snapshot=_json_loads(
                    run.get("resolved_context_snapshot_json"),
                    default=[],
                ),
                model=run.get("question_renderer_model"),
            )
            await self._db_call_async(
                "append_event",
                run_id,
                "question_rendered",
                {
                    "step_index": current_step_index,
                    "step_id": step["id"],
                    **cached_question,
                },
            )

        return {
            **step,
            **cached_question,
            "run_id": run_id,
            "step_index": current_step_index,
        }

    async def record_answer(
        self,
        *,
        run_id: str,
        step_index: int,
        answer_text: str,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if idempotency_key is not None:
            existing_by_key = await self._db_call_async(
                "get_answer_by_idempotency_key",
                run_id,
                idempotency_key,
            )
            if existing_by_key is not None:
                self._validate_replayed_answer(
                    existing_answer=existing_by_key,
                    step_index=step_index,
                    answer_text=answer_text,
                )
                return await self._build_run_result(run_id)

        run = await self._require_run_async(run_id)
        current_step_index = int(run["current_step_index"])
        if step_index != current_step_index:
            existing_answer = await self._db_call_async("get_answer", run_id, step_index)
            if existing_answer is not None and existing_answer["answer_text"] == answer_text:
                return await self._build_run_result(run_id)
            raise ValueError(f"invalid step submission: expected step submission for index {current_step_index}")
        if run.get("status") != "active":
            existing_answer = await self._db_call_async("get_answer", run_id, step_index)
            if existing_answer is not None and existing_answer["answer_text"] == answer_text:
                return await self._build_run_result(run_id)
            raise ValueError("run is not active")

        template_snapshot = self._get_template_snapshot(run)
        steps = template_snapshot["steps"]
        if current_step_index >= len(steps):
            raise ValueError("run is already complete")

        step = steps[current_step_index]
        if self._get_step_type(step) != "question_step":
            raise ValueError("current step requires dialogue round responses")
        rendered_question = await self._get_rendered_question_cache_async(
            run_id,
            current_step_index,
        )
        if rendered_question is None:
            rendered_question = await self._render_question_async(
                run_id=run_id,
                step_index=current_step_index,
                step=step,
                prior_answers=await self._db_call_async("list_answers", run_id),
                context_snapshot=_json_loads(
                    run.get("resolved_context_snapshot_json"),
                    default=[],
                ),
                model=run.get("question_renderer_model"),
            )
            await self._db_call_async(
                "append_event",
                run_id,
                "question_rendered",
                {
                    "step_index": current_step_index,
                    "step_id": step["id"],
                    **rendered_question,
                },
            )

        next_step_index = current_step_index + 1
        is_final_step = next_step_index >= len(steps)
        transition = await self._db_call_async(
            "record_answer_transition",
            run_id=run_id,
            expected_step_index=current_step_index,
            next_step_index=next_step_index,
            next_status="completed" if is_final_step else "active",
            completed_at=_utcnow_iso() if is_final_step else None,
            step_id=step["id"],
            displayed_question=rendered_question["displayed_question"],
            answer_text=answer_text,
            question_generation_meta=rendered_question.get("question_generation_meta", {}),
            idempotency_key=idempotency_key,
        )

        if transition["outcome"] == "replayed":
            return await self._build_run_result(run_id)
        if transition["outcome"] == "stale":
            raise ValueError(
                f"invalid step submission: expected step submission for index "
                f"{int(transition['run']['current_step_index'])}"
            )
        if transition["outcome"] == "conflict":
            raise ChatWorkflowConflictError(
                "idempotency key has already been used for a different answer"
                if transition.get("reason") == "idempotency_key_reused"
                else "step already contains a different answer"
            )

        if is_final_step:
            await self._db_call_async(
                "append_event",
                run_id,
                "run_completed",
                {
                    "final_step_index": current_step_index,
                    "answer_count": len(await self._db_call_async("list_answers", run_id)),
                },
            )
        else:
            await self._db_call_async(
                "append_event",
                run_id,
                "step_answered",
                {
                    "answered_step_index": current_step_index,
                    "next_step_index": next_step_index,
                },
            )

        return await self._build_run_result(run_id)

    async def respond_to_round(
        self,
        *,
        run_id: str,
        round_index: int,
        user_message: str,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Persist a dialogue round and advance the run based on moderator control output."""
        normalized_user_message = str(user_message).strip()
        if not normalized_user_message:
            raise ValueError("user_message must not be empty")

        run = await self._require_run_async(run_id)
        if idempotency_key is not None:
            existing_by_key = await self._db_call_async(
                "get_round_by_idempotency_key",
                run_id,
                idempotency_key,
            )
            if existing_by_key is not None:
                replay_step_index = int(run["current_step_index"]) if run.get("status") == "active" else None
                self._validate_replayed_round(
                    existing_round=existing_by_key,
                    run_id=run_id,
                    step_index=replay_step_index,
                    round_index=round_index,
                    user_message=normalized_user_message,
                )
                if str(existing_by_key.get("status") or "").strip().lower() != "failed":
                    return await self._build_run_result(run_id)

        if run.get("status") != "active":
            raise ValueError("run is not active")

        template_snapshot = self._get_template_snapshot(run)
        current_step_index = int(run["current_step_index"])
        steps = template_snapshot["steps"]
        if current_step_index >= len(steps):
            raise ValueError("run is already complete")

        step = steps[current_step_index]
        if self._get_step_type(step) != "dialogue_round_step":
            raise ValueError("current step does not accept dialogue rounds")
        if round_index != int(run.get("active_round_index", 0)):
            raise ValueError(
                f"invalid round submission: expected round submission for index "
                f"{int(run.get('active_round_index', 0))}"
            )
        if self.dialogue_orchestrator is None:
            raise ValueError("dialogue orchestration is not configured")

        round_claim = await self._db_call_async(
            "begin_dialogue_round",
            run_id=run_id,
            step_index=current_step_index,
            round_index=round_index,
            user_message=normalized_user_message,
            idempotency_key=idempotency_key,
        )
        if round_claim["outcome"] == "replayed":
            return await self._build_run_result(run_id)
        if round_claim["outcome"] == "stale":
            raise ValueError(
                f"invalid round submission: expected round submission for index "
                f"{int(round_claim['run']['active_round_index'])}"
            )
        if round_claim["outcome"] == "conflict":
            raise ChatWorkflowConflictError(
                "idempotency key has already been used for a different dialogue round"
                if idempotency_key is not None
                else "round already contains a different response"
            )

        rounds = await self._db_call_async("list_rounds", run_id, current_step_index)
        prior_rounds = [
            row for row in rounds if int(row.get("round_index", -1)) < round_index and row.get("status") == "completed"
        ]
        current_prompt = self._get_dialogue_prompt(step=step, run=run)

        try:
            round_result = await self.dialogue_orchestrator.run_round(
                run_id=run_id,
                step_index=current_step_index,
                round_index=round_index,
                step=step,
                dialogue_config=self._get_dialogue_config(step),
                current_prompt=current_prompt,
                user_message=normalized_user_message,
                prior_rounds=prior_rounds,
                selected_context_refs=_json_loads(
                    run.get("selected_context_refs_json"),
                    default=[],
                ),
                resolved_context_snapshot=_json_loads(
                    run.get("resolved_context_snapshot_json"),
                    default=[],
                ),
                question_renderer_model=run.get("question_renderer_model"),
            )
            normalized_round_result = self._normalize_dialogue_round_result(round_result)
        except Exception:
            await self._db_call_async(
                "fail_dialogue_round",
                run_id=run_id,
                step_index=current_step_index,
                round_index=round_index,
            )
            raise

        should_finish = self._dialogue_round_should_finish(
            step=step,
            round_index=round_index,
            moderator_decision=normalized_round_result["moderator_decision"],
        )
        if should_finish:
            next_step_index = current_step_index + 1
            next_round_index = 0
            next_status = "completed" if next_step_index >= len(steps) else "active"
            step_runtime_state: dict[str, Any] = {}
            completed_at = _utcnow_iso() if next_status == "completed" else None
        else:
            next_step_index = current_step_index
            next_round_index = round_index + 1
            next_status = "active"
            step_runtime_state = {
                "current_prompt": normalized_round_result["next_user_prompt"] or current_prompt,
            }
            completed_at = None

        finalize_result = await self._db_call_async(
            "complete_dialogue_round",
            run_id=run_id,
            step_index=current_step_index,
            round_index=round_index,
            debate_llm_message=normalized_round_result["debate_llm_message"],
            moderator_decision="finish" if should_finish else normalized_round_result["moderator_decision"],
            moderator_summary=normalized_round_result["moderator_summary"],
            next_user_prompt=(None if should_finish else step_runtime_state.get("current_prompt")),
            next_step_index=next_step_index,
            next_round_index=next_round_index,
            next_status=next_status,
            step_runtime_state_json=step_runtime_state,
            completed_at=completed_at,
        )
        if finalize_result["outcome"] == "stale":
            await self._db_call_async(
                "fail_dialogue_round",
                run_id=run_id,
                step_index=current_step_index,
                round_index=round_index,
            )
            raise ValueError("workflow state changed during round execution")

        event_type = "run_completed" if next_status == "completed" else "dialogue_round_completed"
        await self._db_call_async(
            "append_event",
            run_id,
            event_type,
            {
                "step_index": current_step_index,
                "round_index": round_index,
                "moderator_decision": "finish" if should_finish else normalized_round_result["moderator_decision"],
                "advanced_to_step_index": next_step_index,
                "next_round_index": next_round_index,
            },
        )
        return await self._build_run_result(run_id)

    def _normalize_template(self, template: dict[str, Any]) -> dict[str, Any]:
        normalized_steps: list[dict[str, Any]] = []
        for index, step in enumerate(template.get("steps", [])):
            step_type = self._get_step_type(step)
            context_refs = step.get("context_refs")
            if context_refs is None:
                context_refs = _json_loads(step.get("context_refs_json"), default=[])
            normalized_steps.append(
                {
                    "id": str(step.get("id") or step.get("step_id") or f"step-{index + 1}"),
                    "step_index": int(step.get("step_index", index)),
                    "step_type": step_type,
                    "label": step.get("label"),
                    "base_question": str(step["base_question"]).strip(),
                    "question_mode": str(step.get("question_mode", "stock")).strip().lower().replace("-", "_"),
                    "phrasing_instructions": step.get("phrasing_instructions"),
                    "context_refs": list(context_refs if isinstance(context_refs, list) else []),
                    "dialogue_config": (
                        self._normalize_dialogue_config(
                            step.get("dialogue_config") or _json_loads(step.get("dialogue_config_json"), default=None)
                        )
                        if step_type == "dialogue_round_step"
                        else None
                    ),
                }
            )
        return {
            "id": template.get("id"),
            "title": template.get("title") or "Untitled workflow",
            "description": template.get("description"),
            "version": int(template.get("version", 1)),
            "steps": normalized_steps,
        }

    def _require_run(self, run_id: str) -> dict[str, Any]:
        run = self.db.get_run(run_id)
        if run is None:
            raise ValueError(f"run not found: {run_id}")
        return run

    async def _require_run_async(self, run_id: str) -> dict[str, Any]:
        run = await self._db_call_async("get_run", run_id)
        if run is None:
            raise ValueError(f"run not found: {run_id}")
        return run

    async def _build_run_result(self, run_id: str) -> dict[str, Any]:
        run = await self._require_run_async(run_id)
        result = dict(run)
        result["answers"] = await self._db_call_async("list_answers", run_id)
        result["selected_context_refs"] = _json_loads(
            run.get("selected_context_refs_json"),
            default=[],
        )
        result["current_step_kind"] = None
        result["current_prompt"] = None
        result["current_round_index"] = None
        result["rounds"] = []
        result["current_question"] = None
        if result["status"] == "active":
            current_step = await self.get_current_step(run_id)
            result["current_step"] = current_step
            if current_step is not None:
                step_type = self._get_step_type(current_step)
                result["current_step_kind"] = step_type
                result["current_prompt"] = current_step.get("current_prompt") or current_step.get("displayed_question")
                result["current_question"] = current_step.get("displayed_question")
                if step_type == "dialogue_round_step":
                    result["current_round_index"] = int(run.get("active_round_index", 0))
                    result["rounds"] = current_step.get("rounds", [])
        return result

    def _validate_replayed_answer(
        self,
        *,
        existing_answer: dict[str, Any],
        step_index: int,
        answer_text: str,
    ) -> None:
        if int(existing_answer["step_index"]) != step_index:
            raise ChatWorkflowConflictError("idempotency key has already been used for a different step")
        if existing_answer["answer_text"] != answer_text:
            raise ChatWorkflowConflictError("idempotency key has already been used for a different answer")

    def _validate_replayed_round(
        self,
        *,
        existing_round: dict[str, Any],
        run_id: str,
        step_index: int | None,
        round_index: int,
        user_message: str,
    ) -> None:
        """Validate that a stored dialogue round matches the submitted replay request."""
        if existing_round["run_id"] != run_id:
            raise ChatWorkflowConflictError("idempotency key has already been used for a different dialogue round")
        if step_index is not None and int(existing_round["step_index"]) != step_index:
            raise ChatWorkflowConflictError("idempotency key has already been used for a different dialogue round")
        if int(existing_round["round_index"]) != round_index:
            raise ChatWorkflowConflictError("idempotency key has already been used for a different dialogue round")
        if existing_round["user_message"] != user_message:
            raise ChatWorkflowConflictError("idempotency key has already been used for a different dialogue round")

    def _get_template_snapshot(self, run: dict[str, Any]) -> dict[str, Any]:
        snapshot = _json_loads(run.get("template_snapshot_json"), default={})
        steps = snapshot.get("steps", [])
        if not isinstance(steps, list):
            raise ValueError("run template snapshot is invalid")
        return snapshot

    def _get_step_type(self, step: dict[str, Any]) -> str:
        """Return the normalized workflow step type."""
        return str(step.get("step_type", "question_step")).strip().lower().replace("-", "_")

    def _get_dialogue_config(self, step: dict[str, Any]) -> dict[str, Any]:
        """Return the normalized dialogue configuration for a step."""
        config = step.get("dialogue_config") or {}
        if not isinstance(config, dict):
            raise ValueError("dialogue step configuration is invalid")
        return config

    def _normalize_dialogue_config(self, config: Any) -> dict[str, Any]:
        """Normalize a dialogue config payload for template snapshots."""
        if config is None:
            return {}
        if not isinstance(config, dict):
            raise ValueError("dialogue_config must be an object")
        normalized = dict(config)
        opening_prompt_mode = normalized.get("opening_prompt_mode", "base_question")
        if isinstance(opening_prompt_mode, str):
            normalized["opening_prompt_mode"] = opening_prompt_mode.strip().lower().replace("-", "_")
        return normalized

    def _get_dialogue_opening_prompt(self, step: dict[str, Any]) -> str:
        """Resolve the first prompt shown for a dialogue step."""
        dialogue_config = self._get_dialogue_config(step)
        if dialogue_config.get("opening_prompt_mode") == "custom_prompt":
            custom_prompt = str(dialogue_config.get("opening_prompt_text") or "").strip()
            if custom_prompt:
                return custom_prompt
        return str(step.get("base_question") or "").strip()

    def _get_dialogue_prompt(self, *, step: dict[str, Any], run: dict[str, Any]) -> str:
        """Return the prompt currently shown to the user for a dialogue step."""
        runtime_state = _json_loads(run.get("step_runtime_state_json"), default={})
        if isinstance(runtime_state, dict):
            current_prompt = str(runtime_state.get("current_prompt") or "").strip()
            if current_prompt:
                return current_prompt
        return self._get_dialogue_opening_prompt(step)

    def _normalize_dialogue_round_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize orchestrator control output."""
        debate_llm_message = str(result.get("debate_llm_message") or "").strip()
        if not debate_llm_message:
            raise ValueError("dialogue orchestrator returned an empty debate_llm_message")
        moderator_decision = str(result.get("moderator_decision") or "").strip().lower()
        if moderator_decision not in {"continue", "finish"}:
            raise ValueError("dialogue orchestrator returned an invalid moderator_decision")
        moderator_summary = str(result.get("moderator_summary") or "").strip() or None
        next_user_prompt = str(result.get("next_user_prompt") or "").strip() or None
        return {
            "debate_llm_message": debate_llm_message,
            "moderator_decision": moderator_decision,
            "moderator_summary": moderator_summary,
            "next_user_prompt": next_user_prompt,
        }

    def _dialogue_round_should_finish(
        self,
        *,
        step: dict[str, Any],
        round_index: int,
        moderator_decision: str,
    ) -> bool:
        """Decide whether a dialogue step should advance after this round."""
        if moderator_decision == "finish":
            return True
        max_rounds = int(self._get_dialogue_config(step).get("max_rounds", 1))
        return round_index + 1 >= max_rounds

    async def _get_rendered_question_cache_async(
        self,
        run_id: str,
        step_index: int,
    ) -> dict[str, Any] | None:
        events = await self._db_call_async("list_events", run_id)
        for event in reversed(events):
            if event["event_type"] != "question_rendered":
                continue
            payload = _json_loads(event["payload_json"], default={})
            if int(payload.get("step_index", -1)) != step_index:
                continue
            return {
                "displayed_question": payload["displayed_question"],
                "question_generation_meta": payload.get("question_generation_meta", {}),
                "fallback_used": bool(payload.get("fallback_used", False)),
            }
        return None

    async def _render_question_async(
        self,
        *,
        run_id: str,
        step_index: int,
        step: dict[str, Any],
        prior_answers: list[dict[str, Any]],
        context_snapshot: list[dict[str, Any]] | list[Any],
        model: str | None = None,
    ) -> dict[str, Any]:
        base_question = str(step.get("base_question") or "").strip()
        step_model = str(step.get("model") or "").strip() or None
        effective_model = model if model is not None else step_model
        stock_result = {
            "displayed_question": base_question,
            "question_generation_meta": {"mode": "stock"},
            "fallback_used": False,
        }
        if step.get("question_mode") != "llm_phrased":
            return stock_result
        if self.question_renderer is None:
            return self._build_question_fallback_result(
                base_question,
                reason="renderer_unavailable",
                model=effective_model,
            )

        try:
            rendered = await self.question_renderer.render_question(
                base_question=base_question,
                phrasing_instructions=step.get("phrasing_instructions"),
                prior_answers=prior_answers,
                context_snapshot=context_snapshot,
                model=effective_model,
            )
            displayed_question = str(rendered.get("displayed_question") or "").strip()
            if not displayed_question:
                raise ValueError("renderer returned an empty displayed_question")
            return {
                "displayed_question": displayed_question,
                "question_generation_meta": rendered.get("question_generation_meta", {}),
                "fallback_used": bool(rendered.get("fallback_used", False)),
            }
        except Exception as exc:  # noqa: BLE001
            error_type = type(exc).__name__
            logger.warning(
                "Falling back to base question for chat workflow step "
                "run_id={} step_index={} step_id={} model={} error_type={}",
                run_id,
                step_index,
                step.get("id"),
                effective_model,
                error_type,
            )
            return self._build_question_fallback_result(
                base_question,
                reason="renderer_error",
                model=effective_model,
                error_type=error_type,
            )

    def _build_question_fallback_result(
        self,
        base_question: str,
        *,
        reason: str,
        model: str | None = None,
        error_type: str | None = None,
    ) -> dict[str, Any]:
        """Build the sanitized fallback payload returned when question rendering is unavailable."""
        meta: dict[str, Any] = {
            "mode": "fallback",
            "reason": reason,
            "model": model,
        }
        if error_type is not None:
            meta["error_type"] = error_type
        return {
            "displayed_question": base_question,
            "question_generation_meta": meta,
            "fallback_used": True,
        }

    def _render_question_sync(
        self,
        *,
        step: dict[str, Any],
        prior_answers: list[dict[str, Any]],
        context_snapshot: list[dict[str, Any]] | list[Any],
        model: str | None = None,
    ) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self._render_question_async(
                    run_id="sync-preview",
                    step_index=int(step.get("step_index", 0)),
                    step=step,
                    prior_answers=prior_answers,
                    context_snapshot=context_snapshot,
                    model=model,
                )
            )

        base_question = str(step.get("base_question") or "").strip()
        return {
            "displayed_question": base_question,
            "question_generation_meta": {"mode": "sync_fallback"},
            "fallback_used": True,
        }
