"""research code moved from unified_pipeline.py (TASK-13950.24)."""

import contextlib
import time
from typing import TYPE_CHECKING, Any, Optional

from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as up

from .run_config import RagRunConfig

if TYPE_CHECKING:
    from .run_state import PipelineState


async def run_research_loop(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Agentic research loop."""
    if (
        rcfg.research.enable_research_loop
        and up.research_loop is not None
        and state._classification is not None
        and not state._classification.skip_search
    ):
        try:
            _research_start = time.time()
            _research_mode = rcfg.request.search_depth_mode or "balanced"
            _db_ctx = {
                "user_id": rcfg.scope.user_id or "0",
                "media_db_path": state.media_db_path,
                "notes_db_path": rcfg.runtime.notes_db_path,
                "character_db_path": rcfg.runtime.character_db_path,
                "kanban_db_path": rcfg.runtime.kanban_db_path,
            }
            if rcfg.runtime.media_db is not None:
                _db_ctx["media_db"] = rcfg.runtime.media_db
            if rcfg.runtime.chacha_db is not None:
                _db_ctx["chacha_db"] = rcfg.runtime.chacha_db

            def _positive_int_or_none(value: Any) -> Optional[int]:
                try:
                    ivalue = int(value)
                except (TypeError, ValueError):
                    return None
                return ivalue if ivalue > 0 else None

            _mode_specific_overrides = {
                "speed": _positive_int_or_none(rcfg.research.research_max_iterations_speed),
                "balanced": _positive_int_or_none(rcfg.research.research_max_iterations_balanced),
                "quality": _positive_int_or_none(rcfg.research.research_max_iterations_quality),
            }
            _effective_max_iterations = _positive_int_or_none(rcfg.research.research_max_iterations)
            if _effective_max_iterations is None:
                _effective_max_iterations = _mode_specific_overrides.get(_research_mode)
            if _effective_max_iterations is None:
                with contextlib.suppress(TypeError, ValueError):
                    import os as _os

                    _env_key = f"SEARCH_MAX_ITERATIONS_{_research_mode.upper()}"
                    _effective_max_iterations = _positive_int_or_none(_os.getenv(_env_key))

            _progress_cb = rcfg.runtime.research_progress_callback if rcfg.research.enable_research_progress else None
            _registry = None
            if up.create_default_registry is not None:
                with contextlib.suppress(TypeError, ValueError):
                    _registry = up.create_default_registry(
                        discussion_platforms=rcfg.research.discussion_platforms,
                        enable_url_scraping=bool(rcfg.research.search_url_scraping),
                        enable_image_search=bool(rcfg.generation.enable_image_search),
                        enable_video_search=bool(rcfg.generation.enable_video_search),
                        on_progress=_progress_cb,
                        **(
                            {"credential_runtime": rcfg.runtime.credential_runtime}
                            if rcfg.runtime.credential_runtime is not None
                            else {}
                        ),
                    )

            _research_output = await up.research_loop(
                query=state.query,
                classification=state._classification,
                mode=_research_mode,
                llm_provider=rcfg.research.classifier_provider or rcfg.generation.generation_provider or "openai",
                llm_model=rcfg.research.classifier_model or rcfg.generation.generation_model,
                max_iterations=_effective_max_iterations,
                on_progress=_progress_cb,
                registry=_registry,
                db_context=_db_ctx,
                discussion_platforms=rcfg.research.discussion_platforms,
                enable_url_scraping=bool(rcfg.research.search_url_scraping),
                enable_action_dedup=bool(rcfg.research.enable_research_action_dedup),
                enable_image_search=bool(rcfg.generation.enable_image_search),
                enable_video_search=bool(rcfg.generation.enable_video_search),
                **(
                    {"credential_runtime": rcfg.runtime.credential_runtime}
                    if rcfg.runtime.credential_runtime is not None
                    else {}
                ),
            )

            # Convert research results to Document objects
            from ..types import DataSource as _DS, Document as _Doc  # noqa: I001

            _research_docs = []
            for r_item in _research_output.all_results:
                _research_docs.append(
                    _Doc(
                        id=str(r_item.get("id", r_item.get("url", ""))),
                        content=str(r_item.get("content", "")),
                        metadata={
                            "title": r_item.get("title", ""),
                            "url": r_item.get("url", ""),
                            "source_type": r_item.get("source", "research"),
                            "platform": r_item.get("platform", ""),
                        },
                        source=(
                            _DS.WEB_CONTENT
                            if r_item.get("source") in ("web", "discussion", "academic", "scraped_url")
                            else _DS.MEDIA_DB
                        ),
                        score=float(r_item.get("score", 0.5)),
                    )
                )

            state.result.documents = _research_docs
            state.result.timings["research_loop"] = time.time() - _research_start
            state.result.metadata["research"] = {
                "total_iterations": _research_output.total_iterations,
                "total_results": _research_output.total_results,
                "completed": _research_output.completed,
                "final_reasoning": _research_output.final_reasoning,
                "max_iterations_requested": _effective_max_iterations,
                "url_scraping_enabled": bool(rcfg.research.search_url_scraping),
                "action_dedup": _research_output.metadata.get("action_dedup", {}),
                "discussion_platforms": rcfg.research.discussion_platforms or ["reddit", "stackoverflow", "hackernews"],
                "url_dedup": _research_output.metadata.get("url_dedup", {}),
                **(
                    dict(_research_output.metadata.get("provider_stage") or {})
                    if isinstance(
                        _research_output.metadata.get("provider_stage"),
                        dict,
                    )
                    else {}
                ),
                "steps": [
                    {
                        "iteration": s.iteration,
                        "action": s.action_name,
                        "reasoning": s.reasoning,
                        "result_count": s.output.result_count if s.output else 0,
                        "duration_sec": s.duration_sec,
                    }
                    for s in _research_output.steps
                ],
            }

            # Collect media search results from research steps
            if rcfg.generation.enable_image_search or rcfg.generation.enable_video_search:
                _images = []
                _videos = []
                _media_provider_stage: dict[str, Any] = {}
                for _step in _research_output.steps:
                    if not _step.output or not _step.output.success:
                        continue
                    _step_metadata = (
                        _step.output.metadata
                        if isinstance(_step.output.metadata, dict)
                        else {}
                    )
                    _step_type = _step_metadata.get("type", "")
                    if _step_type not in {"images", "videos"}:
                        continue
                    if _step.output.results:
                        if _step_type == "images":
                            _images.extend(_step.output.results)
                        else:
                            _videos.extend(_step.output.results)
                    _verification_available = _step_metadata.get(
                        "verification_available"
                    )
                    if isinstance(_verification_available, bool):
                        _bounded_stage: dict[str, bool | str] = {
                            "verification_available": _verification_available
                        }
                        if (
                            _verification_available is False
                            and _step_metadata.get("failure_code")
                            == "provider_unavailable"
                        ):
                            _bounded_stage["failure_code"] = "provider_unavailable"
                        if (
                            _verification_available is False
                            or _media_provider_stage.get(
                                "verification_available"
                            )
                            is not False
                        ):
                            _media_provider_stage = _bounded_stage
                if _images:
                    state.result.metadata["images"] = _images
                if _videos:
                    state.result.metadata["videos"] = _videos
                if _media_provider_stage:
                    state.result.metadata["research"]["media_provider_stage"] = (
                        _media_provider_stage
                    )

            up.logger.info(
                f"Research loop completed: {_research_output.total_iterations} iterations, "
                f"{_research_output.total_results} results in "
                f"{_research_output.total_duration_sec:.2f}s"
            )

            # Skip cache/retrieval after a successful research loop so the
            # assembled research evidence remains the source of truth.
            _research_provider_stage = _research_output.metadata.get("provider_stage")
            _research_available = not (
                isinstance(_research_provider_stage, dict)
                and _research_provider_stage.get("verification_available") is False
            )
            if _research_available:
                state._skip_retrieval_stack = True
                state._skip_retrieval_reason = "research_loop"
                state.result.metadata["research_retrieval_bypassed"] = True
            elif up._should_restore_classification_local_retrieval(
                skip_retrieval_stack=state._skip_retrieval_stack,
                skip_local_retrieval=state._skip_local_retrieval,
                classification_local_retrieval=state.result.metadata.get(
                    "classification_local_retrieval"
                ),
            ):
                state._skip_local_retrieval = False
                state.result.metadata["classification_local_retrieval"] = (
                    "fallback_after_research_unavailable"
                )

        except Exception as _res_exc:  # noqa: BLE001 - moved from unified_pipeline.py (TASK-13950.24)
            up.logger.warning(f"Research loop failed, falling back to standard pipeline: {_res_exc!r}")
            state.result.errors.append(f"Research loop failed: {_res_exc}")
