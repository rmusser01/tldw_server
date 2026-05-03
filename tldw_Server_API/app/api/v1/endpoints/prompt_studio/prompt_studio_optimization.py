"""
Prompt Studio Optimizations API

Creates and manages optimization jobs that iterate on prompts using
defined strategies (e.g., iterative refinement, hyperparameter tuning,
genetic algorithms). Integrates with the job queue and background
workers to run safely and asynchronously.

Key responsibilities
- Create optimization jobs against a prompt and test cases
- List/get/cancel optimizations
- Enumerate available optimization strategies
- Compare multiple strategies by spawning multiple jobs

Security
- Read operations require project access
- Write operations require project write access
- Rate limits applied to optimization creation and comparisons
"""

import contextlib
import json
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Path, Query, Request, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps import (
    PromptStudioDatabase,
    SecurityConfig,
    check_rate_limit,
    get_prompt_studio_db,
    get_prompt_studio_user,
    get_security_config,
    require_project_access,
    require_project_write_access,
)
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import (
    build_page_pagination_meta,
    resolve_page_pagination_metadata,
)
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http

# Local imports
from tldw_Server_API.app.api.v1.schemas.prompt_studio_base import (
    PageListResponse,
    PageStandardResponse,
    StandardResponse,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_optimization import (
    OptimizationCreate,
    OptimizationResponse,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_optimization_requests import (
    CompareStrategiesRequest,
    OptimizationSimpleCreateRequest,
)
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import DatabaseError
from tldw_Server_API.app.core.Logging.log_context import (
    ensure_request_id,
    ensure_traceparent,
    get_ps_logger,
    log_context,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.job_types import JobType
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.jobs_adapter import PromptStudioJobsAdapter
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.monitoring import prompt_studio_metrics
from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.app.core.Utils.pydantic_compat import model_dump_compat

_OPTIMIZATION_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    json.JSONDecodeError,
    DatabaseError,
)

########################################################################################################################
# Router Setup

router = APIRouter(
    prefix="/api/v1/prompt-studio/optimizations",
    tags=["prompt-studio"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not found"},
        429: {"description": "Rate limit exceeded"}
    }
)

def _get_optimization_request_logger(
    request: Request | None,
    *,
    optimization_id: int,
):
    request_id = ensure_request_id(request) if request is not None else None
    traceparent = ensure_traceparent(request) if request is not None else ""
    return get_ps_logger(
        ps_component="endpoint",
        ps_job_kind="optimization",
        optimization_id=optimization_id,
        request_id=request_id,
        traceparent=traceparent,
    )

########################################################################################################################
# Optimization CRUD Endpoints

# --- Strategy helpers ---
_OPTIMIZER_SYNONYMS = {
    "hill_climb": "hill_climbing",
}

_VALIDATION_REQUIRED = {
    # For these strategies, we validate additional fields
    "grid_search": ("models_to_test",),
    "bayesian": ("models_to_test",),
    # bootstrap may require bootstrap_config, but we keep this optional for now
}

def _normalize_optimizer_type(opt_type: str) -> str:
    t = (opt_type or "").strip().lower()
    return _OPTIMIZER_SYNONYMS.get(t, t)

def _validate_strategy_config(optimizer_type: str, cfg: dict[str, Any]) -> None:
    """Light validation for specific strategies.

    Keeps existing behavior for common strategies (iterative, mipro, random_search, hill_climbing,
    beam_search, greedy, anneal, genetic). For grid_search/bayesian, require non-empty models_to_test.
    """
    ot = _normalize_optimizer_type(optimizer_type)
    required = _VALIDATION_REQUIRED.get(ot, ())
    for field in required:
        value = cfg.get(field)
        if not value or (isinstance(value, list) and len(value) == 0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"optimizer_type='{optimizer_type}' requires non-empty '{field}'",
            )

    # Optional, strategy-specific validations (only apply if provided)
    params: dict[str, Any] = {}
    try:
        raw = cfg.get("strategy_params")
        if isinstance(raw, dict):
            params = raw
    except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS:
        params = {}

    def _get(name: str) -> Any:
        # Look both at top-level and strategy_params
        return cfg.get(name, params.get(name))

    # beam_search: validate beam_width if provided
    if ot == "beam_search":
        bw = _get("beam_width")
        if bw is not None:
            try:
                bw_int = int(bw)
            except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS:
                raise HTTPException(status_code=400, detail="beam_width must be an integer >= 2") from None
            if bw_int < 2:
                raise HTTPException(status_code=400, detail="beam_width must be >= 2")
        # Optional pruning threshold within [0,1]
        pt = _get("prune_threshold")
        if pt is not None:
            try:
                pt_f = float(pt)
            except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS:
                raise HTTPException(status_code=400, detail="prune_threshold must be a float between 0 and 1") from None
            if not (0.0 <= pt_f <= 1.0):
                raise HTTPException(status_code=400, detail="prune_threshold must be in [0, 1]")
        # Optional max_candidates >= beam_width if both provided
        mc = _get("max_candidates")
        if mc is not None:
            try:
                mc_i = int(mc)
            except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS:
                raise HTTPException(status_code=400, detail="max_candidates must be an integer >= 2") from None
            if mc_i < 2:
                raise HTTPException(status_code=400, detail="max_candidates must be >= 2")
            if bw is not None and int(bw) > mc_i:
                raise HTTPException(status_code=400, detail="max_candidates must be >= beam_width")
        # Optional diversity_rate in [0,1]
        dr = _get("diversity_rate")
        if dr is not None:
            try:
                dr_f = float(dr)
            except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS:
                raise HTTPException(status_code=400, detail="diversity_rate must be a float between 0 and 1") from None
            if not (0.0 <= dr_f <= 1.0):
                raise HTTPException(status_code=400, detail="diversity_rate must be in [0, 1]")
        # Optional length_penalty (>= 0, typical range [0,2])
        lp = _get("length_penalty")
        if lp is not None:
            try:
                lp_f = float(lp)
            except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS:
                raise HTTPException(status_code=400, detail="length_penalty must be a non-negative number") from None
            if not (0.0 <= lp_f <= 2.0):
                raise HTTPException(status_code=400, detail="length_penalty must be in [0, 2]")
        # Optional candidate_reranker policy
        crp = _get("candidate_reranker")
        if crp is not None:
            allowed_crp = {"none", "score", "diversity", "hybrid"}
            if str(crp).lower() not in allowed_crp:
                raise HTTPException(status_code=400, detail=f"candidate_reranker must be one of {sorted(allowed_crp)}")

    # anneal (simulated annealing): validate cooling_rate and initial_temp if provided
    if ot in {"anneal", "simulated_annealing"}:
        cr = _get("cooling_rate")
        if cr is not None:
            try:
                cr_f = float(cr)
            except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS:
                raise HTTPException(status_code=400, detail="cooling_rate must be a float between 0 and 1") from None
            if not (0.0 < cr_f <= 1.0):
                raise HTTPException(status_code=400, detail="cooling_rate must be in (0, 1]")
        it = _get("initial_temp")
        if it is not None:
            try:
                it_f = float(it)
            except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS:
                raise HTTPException(status_code=400, detail="initial_temp must be a positive number") from None
            if it_f <= 0:
                raise HTTPException(status_code=400, detail="initial_temp must be > 0")
        # Optional schedule type
        sched = _get("schedule")
        if sched is not None:
            allowed = {"exponential", "linear", "cosine"}
            if str(sched).lower() not in allowed:
                raise HTTPException(status_code=400, detail=f"schedule must be one of {sorted(allowed)}")
        # Optional min_temp <= initial_temp and >= 0
        mt = _get("min_temp")
        if mt is not None:
            try:
                mt_f = float(mt)
            except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS:
                raise HTTPException(status_code=400, detail="min_temp must be a non-negative number") from None
            if mt_f < 0:
                raise HTTPException(status_code=400, detail="min_temp must be >= 0")
            if _get("initial_temp") is not None and float(_get("initial_temp")) < mt_f:
                raise HTTPException(status_code=400, detail="min_temp must be <= initial_temp")
        # Optional step schedule knobs
        step_size = _get("step_size")
        if step_size is not None:
            try:
                ss_f = float(step_size)
            except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS:
                raise HTTPException(status_code=400, detail="step_size must be a positive number") from None
            if ss_f <= 0:
                raise HTTPException(status_code=400, detail="step_size must be > 0")
        epochs = _get("epochs")
        if epochs is not None:
            try:
                ep_i = int(epochs)
            except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS:
                raise HTTPException(status_code=400, detail="epochs must be a positive integer") from None
            if ep_i < 1:
                raise HTTPException(status_code=400, detail="epochs must be >= 1")
        # If linear schedule and we have initial/min temps, epochs and step_size, ensure consistency
        try:
            if str(_get("schedule")).lower() == "linear" and all(v is not None for v in (it, mt, step_size, epochs)):
                if float(it) - float(mt) < float(step_size) * int(epochs):
                    raise HTTPException(status_code=400, detail="linear schedule: step_size * epochs must not exceed (initial_temp - min_temp)")
        except HTTPException:
            # Bubble up expected validation error
            raise
        except (TypeError, ValueError):
            # Ignore only type conversion issues; other checks above handle them
            pass

    # genetic: validate population_size and mutation_rate if provided
    if ot == "genetic":
        ps = _get("population_size")
        if ps is not None:
            try:
                ps_i = int(ps)
            except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS:
                raise HTTPException(status_code=400, detail="population_size must be an integer >= 2") from None
            if ps_i < 2:
                raise HTTPException(status_code=400, detail="population_size must be >= 2")
        mr = _get("mutation_rate")
        if mr is not None:
            try:
                mr_f = float(mr)
            except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS:
                raise HTTPException(status_code=400, detail="mutation_rate must be a float between 0 and 1") from None
            if not (0.0 <= mr_f <= 1.0):
                raise HTTPException(status_code=400, detail="mutation_rate must be in [0, 1]")
        # Optional crossover_rate in [0,1]
        cr = _get("crossover_rate")
        if cr is not None:
            try:
                cr_f = float(cr)
            except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS:
                raise HTTPException(status_code=400, detail="crossover_rate must be a float between 0 and 1") from None
            if not (0.0 <= cr_f <= 1.0):
                raise HTTPException(status_code=400, detail="crossover_rate must be in [0, 1]")
        # Optional elitism >= 0
        el = _get("elitism")
        if el is not None:
            try:
                el_i = int(el)
            except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS:
                raise HTTPException(status_code=400, detail="elitism must be a non-negative integer") from None
            if el_i < 0:
                raise HTTPException(status_code=400, detail="elitism must be >= 0")
        # Optional selection policy
        sel = _get("selection")
        if sel is not None:
            allowed_sel = {"tournament", "roulette", "rank"}
            if str(sel).lower() not in allowed_sel:
                raise HTTPException(status_code=400, detail=f"selection must be one of {sorted(allowed_sel)}")
        # Optional crossover operator enum
        xo = _get("crossover_operator")
        if xo is not None:
            allowed_xo = {"one_point", "two_point", "uniform"}
            if str(xo).lower() not in allowed_xo:
                raise HTTPException(status_code=400, detail=f"crossover_operator must be one of {sorted(allowed_xo)}")

    # hyperparameter tuning (optional checks)
    if ot in {"hyperparameter", "hyperparam", "hparam"}:
        sm = _get("search_method")
        if sm is not None:
            allowed_sm = {"bayesian", "grid", "random"}
            if str(sm).lower() not in allowed_sm:
                raise HTTPException(status_code=400, detail=f"search_method must be one of {sorted(allowed_sm)}")
        pto = _get("params_to_optimize")
        if pto is not None:
            if not isinstance(pto, list) or len(pto) == 0 or not all(isinstance(x, str) and x for x in pto):
                raise HTTPException(status_code=400, detail="params_to_optimize must be a non-empty list of strings")
        max_trials = _get("max_trials")
        if max_trials is not None:
            try:
                mt_i = int(max_trials)
            except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS:
                raise HTTPException(status_code=400, detail="max_trials must be a positive integer") from None
            if mt_i < 1:
                raise HTTPException(status_code=400, detail="max_trials must be >= 1")
        # Optional bounds for common params (max_tokens_range)
        mtr = _get("max_tokens_range")
        if mtr is not None:
            if not isinstance(mtr, (list, tuple)) or len(mtr) != 2:
                raise HTTPException(status_code=400, detail="max_tokens_range must be [min, max]")
            try:
                mn, mx = int(mtr[0]), int(mtr[1])
            except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS:
                raise HTTPException(status_code=400, detail="max_tokens_range must contain integers") from None
            if not (1 <= mn < mx <= 100000):
                raise HTTPException(status_code=400, detail="max_tokens_range must satisfy 1 <= min < max <= 100000")

    # random_search optional checks
    if ot == "random_search":
        mt = _get("max_trials")
        if mt is not None:
            try:
                mt_i = int(mt)
            except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS:
                raise HTTPException(status_code=400, detail="max_trials must be a positive integer") from None
            if mt_i < 1:
                raise HTTPException(status_code=400, detail="max_trials must be >= 1")
        # Optional bounds for common params (max_tokens_range)
        mtr = _get("max_tokens_range")
        if mtr is not None:
            if not isinstance(mtr, (list, tuple)) or len(mtr) != 2:
                raise HTTPException(status_code=400, detail="max_tokens_range must be [min, max]")
            try:
                mn, mx = int(mtr[0]), int(mtr[1])
            except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS:
                raise HTTPException(status_code=400, detail="max_tokens_range must contain integers") from None
            if not (1 <= mn < mx <= 100000):
                raise HTTPException(status_code=400, detail="max_tokens_range must satisfy 1 <= min < max <= 100000")

    # mcts: validate tree-search configuration knobs if provided
    if ot == "mcts":
        # Feature gate: default off; canary enable in dev via env
        import os as _os
        def _is_dev_env() -> bool:
            env = (_os.getenv("ENVIRONMENT") or _os.getenv("APP_ENV") or _os.getenv("ENV") or "dev").lower()
            return env in {"dev", "development", "local", "debug"}

        def _flag(name: str, default: str = "false") -> bool:
            return is_truthy(_os.getenv(name, default))

        _mcts_enabled = _flag("PROMPT_STUDIO_ENABLE_MCTS", "false") or (
            _flag("PROMPT_STUDIO_ENABLE_MCTS_CANARY", "true") and _is_dev_env()
        )
        if not _mcts_enabled:
            raise HTTPException(status_code=400, detail="MCTS strategy is disabled. Enable via PROMPT_STUDIO_ENABLE_MCTS or canary in dev.")
        def _as_int(name: str, value: Any, *, ge: int = None, le: int = None) -> int:
            try:
                iv = int(value)
            except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS:
                raise HTTPException(status_code=400, detail=f"{name} must be an integer") from None
            if ge is not None and iv < ge:
                raise HTTPException(status_code=400, detail=f"{name} must be >= {ge}")
            if le is not None and iv > le:
                raise HTTPException(status_code=400, detail=f"{name} must be <= {le}")
            return iv

        def _as_float(name: str, value: Any, *, gt: float = None, ge_f: float = None, le_f: float = None) -> float:
            try:
                fv = float(value)
            except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS:
                raise HTTPException(status_code=400, detail=f"{name} must be a number") from None
            if gt is not None and not (fv > gt):
                raise HTTPException(status_code=400, detail=f"{name} must be > {gt}")
            if ge_f is not None and fv < ge_f:
                raise HTTPException(status_code=400, detail=f"{name} must be >= {ge_f}")
            if le_f is not None and fv > le_f:
                raise HTTPException(status_code=400, detail=f"{name} must be <= {le_f}")
            return fv

        def _as_bool(name: str, value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and int(value) in {0, 1}:
                return bool(int(value))
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "1", "yes", "y", "on"}:
                    return True
                if normalized in {"false", "0", "no", "n", "off"}:
                    return False
            raise HTTPException(
                status_code=400,
                detail=f"{name} must be a boolean (true/false)",
            )

        sims = _get("mcts_simulations")
        if sims is not None:
            _as_int("mcts_simulations", sims, ge=1, le=200)

        depth = _get("mcts_max_depth")
        if depth is not None:
            _as_int("mcts_max_depth", depth, ge=1, le=10)

        c_ucb = _get("mcts_exploration_c")
        if c_ucb is not None:
            _as_float("mcts_exploration_c", c_ucb, ge_f=0.05, le_f=5.0)

        k = _get("prompt_candidates_per_node")
        if k is not None:
            _as_int("prompt_candidates_per_node", k, ge=1, le=10)

        sbin = _get("score_dedup_bin")
        if sbin is not None:
            _as_float("score_dedup_bin", sbin, ge_f=0.01, le_f=0.5)

        min_quality = _get("min_quality")
        if min_quality is not None:
            _as_float("min_quality", min_quality, ge_f=0.0, le_f=1.0)

        feedback_enabled = _get("feedback_enabled")
        if feedback_enabled is not None:
            _as_bool("feedback_enabled", feedback_enabled)

        thr = _get("feedback_threshold")
        if thr is not None:
            _as_float("feedback_threshold", thr, ge_f=0.0, le_f=10.0)

        retries = _get("feedback_max_retries")
        if retries is not None:
            _as_int("feedback_max_retries", retries, ge=0, le=10)

        budget = _get("token_budget")
        if budget is not None:
            _as_int("token_budget", budget, ge=1, le=1_000_000)

        noimp = _get("early_stop_no_improve")
        if noimp is not None:
            _as_int("early_stop_no_improve", noimp, ge=1, le=50)

        ws_every = _get("ws_throttle_every")
        if ws_every is not None:
            _as_int("ws_throttle_every", ws_every, ge=1, le=200)

        trace_top_k = _get("trace_top_k")
        if trace_top_k is not None:
            _as_int("trace_top_k", trace_top_k, ge=1, le=20)

        scorer_model = _get("scorer_model")
        if scorer_model is not None and (not isinstance(scorer_model, str) or not scorer_model.strip()):
            raise HTTPException(status_code=400, detail="scorer_model must be a non-empty string if provided")

        rollout_model = _get("rollout_model")
        if rollout_model is not None:
            if not isinstance(rollout_model, str) or not rollout_model.strip():
                raise HTTPException(status_code=400, detail="rollout_model must be a non-empty string if provided")
            raise HTTPException(
                status_code=400,
                detail="rollout_model is currently unsupported for optimizer_type='mcts'",
            )

# Compatibility: base POST returns job info directly
@router.post("")
async def create_optimization_simple(
    payload: OptimizationSimpleCreateRequest,
    request: Request,  # type: ignore[assignment]
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: dict = Depends(get_prompt_studio_user),
) -> dict[str, Any]:
    # Minimal creation: create a job with provided payload
    prompt_id = int(payload.prompt_id or payload.initial_prompt_id or 0)
    # Correlate job with request_id if available
    req_id = ensure_request_id(request) if request is not None else None
    tp = ensure_traceparent(request) if request is not None else ""
    prompt_row = db.get_prompt_with_project(prompt_id, include_deleted=False)
    project_id = payload.project_id or (prompt_row.get("project_id") if prompt_row else None)
    if not project_id:
        raise HTTPException(status_code=404, detail=f"Prompt {prompt_id} not found")
    optimizer_type = _normalize_optimizer_type(
        payload.strategy or (payload.config or {}).get("optimizer_type") or "iterative"
    )
    combined_config: dict[str, Any] = dict(payload.config or {})
    max_iters = combined_config.get("max_iterations", 10)
    optimization_record = db.create_optimization(
        project_id=project_id,
        name=None,
        initial_prompt_id=prompt_id,
        optimizer_type=optimizer_type,
        optimization_config=combined_config,
        max_iterations=max_iters,
        status="pending",
        client_id=db.client_id,
    )
    adapter = PromptStudioJobsAdapter()
    job_payload: dict[str, Any] = {
        "optimization_id": optimization_record["id"],
        "optimizer_type": optimizer_type,
        "optimization_config": combined_config,
        "initial_prompt_id": prompt_id,
        "project_id": project_id,
        "created_by": user_context.get("user_id"),
        "submitted_at": datetime.utcnow().isoformat(),
    }
    if req_id:
        job_payload["request_id"] = req_id
    job = adapter.create_job(
        user_id=user_context.get("user_id"),
        job_type=JobType.OPTIMIZATION.value,
        entity_id=optimization_record["id"],
        payload=job_payload,
        project_id=project_id,
        priority=5,
        max_retries=3,
        request_id=req_id,
        trace_id=tp,
    )
    with log_context(request_id=req_id, traceparent=tp, ps_component="endpoint", ps_job_kind="optimization"):
        logger.info("Created optimization job via simple endpoint: job_id={}", job.get("id"))
    job_id = str(job.get("uuid") or job.get("id"))
    return {"id": job_id, "status": job.get("status", "pending")}

async def _rl_optimizations(
    user_context: dict = Depends(get_prompt_studio_user),
    security_config: SecurityConfig = Depends(get_security_config),
) -> bool:
    return await check_rate_limit("optimization", user_context=user_context, security_config=security_config)

@router.post(
    "/create",
    response_model=StandardResponse,
    status_code=status.HTTP_201_CREATED,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "iterative": {
                            "summary": "Create optimization job",
                            "value": {
                                "project_id": 1,
                                "initial_prompt_id": 12,
                                "optimization_config": {
                                    "optimizer_type": "iterative",
                                    "max_iterations": 20,
                                    "target_metric": "accuracy",
                                    "early_stopping": True
                                },
                                "test_case_ids": [1, 2, 3],
                                "name": "Refine Summarizer"
                            }
                        },
                        "mcts": {
                            "summary": "Create MCTS sequence optimization job",
                            "value": {
                                "project_id": 1,
                                "initial_prompt_id": 12,
                                "optimization_config": {
                                    "optimizer_type": "mcts",
                                    "max_iterations": 20,
                                    "target_metric": "accuracy",
                                    "strategy_params": {
                                        "mcts_simulations": 20,
                                        "mcts_max_depth": 4,
                                        "mcts_exploration_c": 1.4,
                                        "prompt_candidates_per_node": 3,
                                        "score_dedup_bin": 0.1,
                                        "early_stop_no_improve": 5,
                                        "token_budget": 50000,  # nosec B105 - OpenAPI example value, not a secret
                                    }
                                },
                                "test_case_ids": [1, 2, 3],
                                "name": "MCTS Sequence Optimization"
                            }
                        },
                        "mcts_with_program_evaluator": {
                            "summary": "MCTS with Program Evaluator (feature flag)",
                            "description": "Enable code execution evaluator via PROMPT_STUDIO_ENABLE_CODE_EVAL=true and use test cases with runner=\"python\". See Prompt Studio README for safety and runner spec.",
                            "value": {
                                "project_id": 1,
                                "initial_prompt_id": 12,
                                "optimization_config": {
                                    "optimizer_type": "mcts",
                                    "max_iterations": 10,
                                    "target_metric": "accuracy",
                                    "strategy_params": {
                                        "mcts_simulations": 10,
                                        "mcts_max_depth": 3,
                                        "mcts_exploration_c": 1.4,
                                        "prompt_candidates_per_node": 2,
                                        "token_budget": 20000,  # nosec B105 - OpenAPI example value, not a secret
                                        "ws_throttle_every": 2
                                    }
                                },
                                "test_case_ids": [101, 102],
                                "name": "MCTS + Program Evaluator"
                            }
                        }
                    }
                }
            }
        },
        "responses": {
            "201": {
                "description": "Optimization created",
                "content": {
                    "application/json": {
                        "examples": {
                            "created": {
                                "summary": "Created optimization",
                                "value": {"success": True, "data": {"id": 701, "status": "pending", "job_id": 9001}}
                            }
                        }
                    }
                }
            }
        }
    }
)
async def create_optimization(
    optimization_data: OptimizationCreate,
    request: Request,  # type: ignore[assignment]
    _: bool = Depends(_rl_optimizations),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    security_config: SecurityConfig = Depends(get_security_config),
    user_context: dict = Depends(get_prompt_studio_user),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
) -> StandardResponse:
    """
    Create and start a new optimization.

    Args:
        optimization_data: Optimization configuration
        db: Database instance
        security_config: Security configuration
        user_context: Current user context

    Returns:
        Created optimization details
    """
    try:
        prompt_row = db.get_prompt_with_project(
            optimization_data.initial_prompt_id,
            include_deleted=False,
        )
        if not prompt_row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt {optimization_data.initial_prompt_id} not found",
            )

        project_id = prompt_row["project_id"]
        await require_project_write_access(project_id, user_context=user_context, db=db)

        opt_cfg = optimization_data.optimization_config
        optimizer_type = opt_cfg.optimizer_type
        max_iters = opt_cfg.max_iterations

        if hasattr(opt_cfg, "model_dump_json"):
            try:
                combined_config: dict[str, Any] = json.loads(opt_cfg.model_dump_json())
            except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS:
                combined_config = model_dump_compat(opt_cfg)
        else:
            combined_config = model_dump_compat(opt_cfg)
        if optimization_data.bootstrap_config is not None:
            bootstrap_cfg = optimization_data.bootstrap_config
            if hasattr(bootstrap_cfg, "model_dump_json"):
                try:
                    combined_config["bootstrap_config"] = json.loads(bootstrap_cfg.model_dump_json())
                except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS:
                    combined_config["bootstrap_config"] = model_dump_compat(bootstrap_cfg)
            else:
                combined_config["bootstrap_config"] = model_dump_compat(bootstrap_cfg)

        bootstrap_samples = (
            getattr(optimization_data.bootstrap_config, "num_samples", None)
            if optimization_data.bootstrap_config is not None
            else None
        )

        # Idempotency: check existing optimization mapping for key
        user_id_str = str(user_context.get("user_id", "anonymous"))
        if idempotency_key:
            try:
                existing_id = db.lookup_idempotency("optimization", idempotency_key, user_id_str)
                if existing_id:
                    with contextlib.suppress(_OPTIMIZATION_NONCRITICAL_EXCEPTIONS):
                        prompt_studio_metrics.metrics_manager.increment(
                            "prompt_studio.idempotency.hit_total", labels={"entity_type": "optimization"}
                        )
                    existing_opt = db.get_optimization(existing_id)
                    if existing_opt and int(existing_opt.get("project_id") or -1) == int(project_id):
                        return StandardResponse(success=True, data={"optimization": existing_opt, "job_id": None})
                    if existing_opt:
                        logger.warning(
                            "Ignoring idempotency hit with mismatched project for key {} (user {}, requested project {})",
                            idempotency_key,
                            user_id_str,
                            project_id,
                        )
            except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS:
                pass

        # Per-strategy validation (lightweight)
        try:
            _validate_strategy_config(optimizer_type, combined_config)
        except HTTPException:
            raise
        except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS as _e:  # pragma: no cover - safety
            raise HTTPException(status_code=400, detail=str(_e)) from _e

        optimization_record = db.create_optimization(
            project_id=project_id,
            name=optimization_data.name,
            initial_prompt_id=optimization_data.initial_prompt_id,
            optimizer_type=optimizer_type,
            optimization_config=combined_config,
            max_iterations=max_iters,
            bootstrap_samples=bootstrap_samples,
            status="pending",
            client_id=db.client_id,
        )
        if optimization_data.test_case_ids is not None:
            optimization_record = db.update_optimization(
                optimization_record["id"],
                {"test_case_ids": optimization_data.test_case_ids},
            )

        # Record idempotency mapping
        if idempotency_key and optimization_record.get("id"):
            try:
                db.record_idempotency("optimization", idempotency_key, int(optimization_record["id"]), user_id_str)
                with contextlib.suppress(_OPTIMIZATION_NONCRITICAL_EXCEPTIONS):
                    prompt_studio_metrics.metrics_manager.increment(
                        "prompt_studio.idempotency.miss_total", labels={"entity_type": "optimization"}
                    )
            except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS:
                pass

        req_id = ensure_request_id(request) if request is not None else None
        tp = ensure_traceparent(request) if request is not None else ""
        job_payload: dict[str, Any] = {
            "optimization_id": optimization_record["id"],
            "optimizer_type": optimizer_type,
            "test_case_ids": optimization_data.test_case_ids or [],
            "optimization_config": combined_config,
            "initial_prompt_id": optimization_data.initial_prompt_id,
            "project_id": project_id,
            "created_by": user_context.get("user_id"),
            "submitted_at": datetime.utcnow().isoformat(),
        }
        if req_id:
            job_payload["request_id"] = req_id

        adapter = PromptStudioJobsAdapter()
        job = adapter.create_job(
            user_id=user_context.get("user_id"),
            job_type=JobType.OPTIMIZATION.value,
            entity_id=optimization_record["id"],
            payload=job_payload,
            project_id=project_id,
            priority=5,
            max_retries=3,
            request_id=req_id,
            trace_id=tp,
        )
        with log_context(request_id=req_id, traceparent=tp, ps_component="endpoint", ps_job_kind="optimization"):
            logger.info(
                'User {} created optimization {}',
                user_context.get("user_id"),
                optimization_record.get("id"),
            )

        response_payload = {
            "optimization": optimization_record,
            "job_id": str(job.get("uuid") or job.get("id")),
        }

        return StandardResponse(success=True, data=response_payload)

    except DatabaseError as exc:
        logger.error("Database error creating optimization")
        raise map_db_error_to_http(exc, default_detail="Failed to create optimization") from exc
    except HTTPException:
        raise
    except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS as exc:  # pragma: no cover - safety
        logger.error("Unexpected error creating optimization")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create optimization",
        ) from exc

@router.get(
    "/list/{project_id}",
    response_model=PageListResponse,
    openapi_extra={
        "responses": {
            "200": {
                "description": "Optimizations",
                "content": {
                    "application/json": {
                        "examples": {
                            "list": {
                                "summary": "Optimization list",
                                "value": {
                                    "success": True,
                                    "data": [
                                        {"id": 701, "name": "Refine Summarizer", "status": "pending"}
                                    ],
                                    "metadata": {
                                        "page": 1,
                                        "per_page": 20,
                                        "total": 1,
                                        "total_pages": 1
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
)
async def list_optimizations(
    project_id: int = Path(..., description="Project ID"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    _: bool = Depends(require_project_access),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db)
) -> PageListResponse:
    """
    List optimizations for a project.

    Args:
        project_id: Project ID
        page: Page number
        per_page: Items per page
        status: Optional status filter
        db: Database instance

    Returns:
        Paginated list of optimizations
    """
    try:
        result = db.list_optimizations(
            project_id=project_id,
            status=status_filter,
            page=page,
            per_page=per_page,
        )

        optimizations = [
            OptimizationResponse(**record)
            for record in result.get("optimizations", [])
        ]
        metadata = resolve_page_pagination_metadata(
            result.get("pagination"),
            page=page,
            per_page=per_page,
            item_count=len(optimizations),
        )

        return PageListResponse(
            success=True,
            data=optimizations,
            metadata=metadata,
            pagination=build_page_pagination_meta(
                page=metadata["page"],
                per_page=metadata["per_page"],
                total=metadata["total"],
                total_pages=metadata["total_pages"],
            ),
        )

    except DatabaseError as exc:
        logger.error("Database error listing optimizations")
        raise map_db_error_to_http(exc, default_detail="Failed to list optimizations") from exc
    except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Unexpected error listing optimizations")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list optimizations",
        ) from exc

@router.get("/get/{optimization_id}", response_model=StandardResponse, openapi_extra={
    "responses": {"200": {"description": "Optimization details", "content": {"application/json": {"examples": {"get": {"summary": "Optimization", "value": {"success": True, "data": {"id": 701, "optimizer_type": "iterative", "status": "running"}}}}}}}}
})
async def get_optimization(
    optimization_id: int = Path(..., description="Optimization ID"),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Get optimization details.

    Args:
        optimization_id: Optimization ID
        db: Database instance

    Returns:
        Optimization details
    """
    try:
        optimization = db.get_optimization(optimization_id)
        if not optimization or optimization.get("deleted"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Optimization {optimization_id} not found",
            )

        await require_project_access(
            optimization.get("project_id"),
            user_context=user_context,
            db=db,
        )

        return StandardResponse(
            success=True,
            data=OptimizationResponse(**optimization),
        )

    except DatabaseError as exc:
        logger.error("Database error fetching optimization")
        raise map_db_error_to_http(exc, default_detail="Failed to get optimization") from exc
    except HTTPException:
        raise
    except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Unexpected error getting optimization")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get optimization",
        ) from exc

# Compatibility: GET job status by job_id returning direct job data
@router.get("/{job_id}")
async def get_optimization_job_status(
    job_id: str,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: dict = Depends(get_prompt_studio_user),
) -> dict[str, Any]:
    adapter = PromptStudioJobsAdapter()
    job = adapter.get_job(
        job_id,
        db=db,
        user_id=user_context.get("user_id"),
        job_type=JobType.OPTIMIZATION.value,
    )
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.post("/cancel/{optimization_id}", response_model=StandardResponse, openapi_extra={
    "responses": {"200": {"description": "Cancelled", "content": {"application/json": {"examples": {"cancelled": {"value": {"success": True, "data": {"message": "Optimization cancelled"}}}}}}}, "400": {"description": "Invalid state"}, "404": {"description": "Not found"}}
})
async def cancel_optimization(
    request: Request,
    optimization_id: int = Path(..., description="Optimization ID"),
    reason: str = Body(None, description="Cancellation reason"),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: dict = Depends(get_prompt_studio_user),
) -> StandardResponse:
    """
    Cancel a running optimization.

    Args:
        optimization_id: Optimization ID
        reason: Optional cancellation reason
        db: Database instance
        user_context: Current user context

    Returns:
        Success response
    """
    try:
        optimization = db.get_optimization(optimization_id)
        if not optimization or optimization.get("deleted"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Optimization {optimization_id} not found",
            )

        project_id = optimization.get("project_id")
        await require_project_write_access(project_id, user_context=user_context, db=db)

        status_value = optimization.get("status")
        if status_value in {"completed", "failed", "cancelled"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel optimization with status: {status_value}",
            )

        adapter = PromptStudioJobsAdapter()
        latest_job = adapter.get_latest_job_for_entity(
            db=db,
            user_id=user_context.get("user_id"),
            job_type=JobType.OPTIMIZATION.value,
            entity_id=optimization_id,
        )
        if latest_job:
            adapter.cancel_job(
                latest_job["id"],
                user_id=user_context.get("user_id"),
                reason=reason or "User cancelled",
                job_type=JobType.OPTIMIZATION.value,
            )

        db.set_optimization_status(
            optimization_id,
            "cancelled",
            error_message=reason or "Cancelled by user",
            mark_completed=True,
        )

        _get_optimization_request_logger(request, optimization_id=optimization_id).info(
            "User %s cancelled optimization %s",
            user_context.get("user_id"),
            optimization_id,
        )

        return StandardResponse(
            success=True,
            data={"message": "Optimization cancelled"},
        )

    except DatabaseError as exc:
        _get_optimization_request_logger(request, optimization_id=optimization_id).error(
            "Database error cancelling optimization %s: %s", optimization_id, exc
        )
        raise map_db_error_to_http(exc, default_detail="Failed to cancel optimization") from exc
    except HTTPException:
        raise
    except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS as exc:
        _get_optimization_request_logger(request, optimization_id=optimization_id).error(
            "Unexpected error cancelling optimization %s: %s", optimization_id, exc
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel optimization",
        ) from exc

########################################################################################################################
# Optimization Strategy Endpoints

@router.get("/strategies", response_model=StandardResponse, openapi_extra={
    "responses": {"200": {"description": "Strategies", "content": {"application/json": {"examples": {"list": {"summary": "Available strategies", "value": {"success": True, "data": [{"name": "iterative", "display_name": "Iterative Refinement"}]}}}}}}}
})
async def get_optimization_strategies() -> StandardResponse:
    """
    Get available optimization strategies.

    Returns:
        List of available strategies with descriptions
    """
    strategies = [
        {
            "name": "mipro",
            "display_name": "MIPRO",
            "description": "Multi-Instruction Prompt Optimization - iteratively refines instructions",
            "parameters": {
                "target_metric": "Metric to optimize (accuracy, f1_score, etc.)",
                "min_improvement": "Minimum improvement to continue (0.01-0.1)"
            }
        },
        {
            "name": "bootstrap",
            "display_name": "Bootstrap Few-Shot",
            "description": "Automatically selects best examples for few-shot learning",
            "parameters": {
                "num_examples": "Number of examples to include (1-10)",
                "selection_strategy": "How to select examples (best, diverse, random)"
            }
        },
        {
            "name": "iterative",
            "display_name": "Iterative Refinement",
            "description": "Analyzes errors and iteratively refines the prompt",
            "parameters": {}
        },
        {
            "name": "hyperparameter",
            "display_name": "Hyperparameter Tuning",
            "description": "Optimizes model parameters like temperature and max_tokens",
            "parameters": {
                "params_to_optimize": "List of parameters to tune",
                "search_method": "Search method (bayesian, grid, random)"
            }
        },
        {
            "name": "genetic",
            "display_name": "Genetic Algorithm",
            "description": "Evolves prompts using genetic algorithm techniques",
            "parameters": {
                "population_size": "Population size (5-20)",
                "mutation_rate": "Mutation probability (0.05-0.2)"
            }
        },
        {
            "name": "mcts",
            "display_name": "MCTS (Canary)",
            "description": "Monte Carlo Tree Search over prompt sequences; disabled by default; enable via PROMPT_STUDIO_ENABLE_MCTS or dev canary.",
            "parameters": {
                "mcts_simulations": "Number of simulations (e.g., 5-50)",
                "mcts_max_depth": "Search depth (1-10)",
                "mcts_exploration_c": "UCT exploration constant (0.1-2.0)",
                "prompt_candidates_per_node": "Candidates per node (1-5)",
                "token_budget": "Hard token cap",  # nosec B105 - parameter label, not a credential
                "ws_throttle_every": "Throttling interval for WS iteration events"
            }
        }
    ]

    return StandardResponse(
        success=True,
        data=strategies
    )

@router.get("/history/{optimization_id}", response_model=StandardResponse,
            openapi_extra={
                "responses": {
                    "200": {
                        "description": "Optimization history and progress",
                        "content": {
                            "application/json": {
                                "examples": {
                                    "history": {
                                        "summary": "Recent job and progress",
                                        "value": {
                                            "success": True,
                                            "data": {
                                                "optimization": {"id": 701, "status": "running", "iterations_completed": 3, "max_iterations": 20},
                                                "job": {"id": 9001, "status": "processing"},
                                                "progress": {"iterations_completed": 3, "max_iterations": 20, "status": "running"},
                                                "timeline": [
                                                    {"event": "queued", "job_id": 9001, "at": "2024-09-21T10:00:00"},
                                                    {"event": "processing", "job_id": 9001, "at": "2024-09-21T10:00:05"}
                                                ]
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            })
async def get_optimization_history(
    optimization_id: int = Path(..., description="Optimization ID"),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Fetch optimization status and recent job history for UI progress.

    Returns the optimization row, latest job entry (if any), and
    lightweight progress fields.
    """
    try:
        optimization = db.get_optimization(optimization_id)
        if not optimization or optimization.get("deleted"):
            raise HTTPException(status_code=404, detail="Optimization not found")

        await require_project_access(
            optimization.get("project_id"),
            user_context=user_context,
            db=db,
        )

        adapter = PromptStudioJobsAdapter()
        job = adapter.get_latest_job_for_entity(
            db=db,
            user_id=user_context.get("user_id"),
            job_type=JobType.OPTIMIZATION.value,
            entity_id=optimization_id,
        )
        timeline_records = adapter.list_jobs_for_entity(
            db=db,
            user_id=user_context.get("user_id"),
            job_type=JobType.OPTIMIZATION.value,
            entity_id=optimization_id,
            limit=50,
            ascending=True,
        )

        timeline = [
            {
                "job_id": entry.get("id"),
                "status": entry.get("status"),
                "created_at": entry.get("created_at"),
                "started_at": entry.get("started_at"),
                "completed_at": entry.get("completed_at"),
            }
            for entry in timeline_records
        ]

        return StandardResponse(
            success=True,
            data={
                "optimization": OptimizationResponse(**optimization),
                "job": job,
                "progress": {
                    "iterations_completed": optimization.get("iterations_completed"),
                    "max_iterations": optimization.get("max_iterations"),
                    "status": optimization.get("status"),
                },
                "timeline": timeline,
            },
        )
    except DatabaseError as exc:
        logger.error("Database error fetching optimization history")
        raise map_db_error_to_http(exc, default_detail="Failed to fetch optimization history") from exc
    except HTTPException:
        raise
    except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Unexpected error fetching optimization history")
        raise HTTPException(status_code=500, detail="Failed to fetch optimization history") from exc

########################################################################################################################
# Iteration Events (persisted)

from pydantic import BaseModel, Field


class OptimizationIterationCreate(BaseModel):
    iteration_number: int = Field(..., ge=1, description="Iteration number starting at 1")
    prompt_variant: Optional[dict[str, Any]] = Field(None, description="Prompt variant used")
    metrics: Optional[dict[str, Any]] = Field(None, description="Metrics for this iteration")
    tokens_used: Optional[int] = Field(None, ge=0)
    cost: Optional[float] = Field(None, ge=0.0)
    note: Optional[str] = Field(None, max_length=1000)


@router.post("/iterations/{optimization_id}", response_model=StandardResponse,
             openapi_extra={
                 "requestBody": {
                     "content": {
                         "application/json": {
                             "examples": {
                                 "iteration": {
                                     "summary": "Record iteration",
                                     "value": {
                                         "iteration_number": 4,
                                         "metrics": {"accuracy": 0.82},
                                         "tokens_used": 1400,
                                         "cost": 0.08
                                     }
                                 }
                             }
                         }
                     }
                 },
                 "responses": {
                     "200": {
                         "description": "Iteration persisted",
                         "content": {"application/json": {"examples": {"ok": {"value": {"success": True, "data": {"id": 1001}}}}}}
                     }
                 }
             })
async def add_optimization_iteration(
    optimization_id: int,
    payload: OptimizationIterationCreate,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """Persist a single optimization iteration event."""
    try:
        optimization = db.get_optimization(optimization_id)
        if not optimization or optimization.get("deleted"):
            raise HTTPException(status_code=404, detail="Optimization not found")

        await require_project_write_access(
            optimization.get("project_id"),
            user_context=user_context,
            db=db,
        )

        record = db.record_optimization_iteration(
            optimization_id,
            iteration_number=payload.iteration_number,
            prompt_variant=payload.prompt_variant,
            metrics=payload.metrics,
            tokens_used=payload.tokens_used,
            cost=payload.cost,
            note=payload.note,
        )

        return StandardResponse(success=True, data=record)
    except DatabaseError as exc:
        logger.error("Database error recording optimization iteration")
        raise map_db_error_to_http(exc, default_detail="Failed to add iteration") from exc
    except HTTPException:
        raise
    except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Unexpected error adding optimization iteration")
        raise HTTPException(status_code=500, detail="Failed to add iteration") from exc


@router.get(
    "/iterations/{optimization_id}",
    response_model=PageStandardResponse,
    openapi_extra={
        "responses": {
            "200": {
                "description": "Iteration list",
                "content": {
                    "application/json": {
                        "examples": {
                            "list": {
                                "value": {
                                    "success": True,
                                    "data": [
                                        {"iteration_number": 1, "metrics": {"accuracy": 0.7}}
                                    ],
                                    "metadata": {
                                        "page": 1,
                                        "per_page": 50,
                                        "total": 1,
                                        "total_pages": 1
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
)
async def list_optimization_iterations(
    optimization_id: int,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: dict = Depends(get_prompt_studio_user)
) -> PageStandardResponse:
    """List persisted iterations for an optimization."""
    try:
        optimization = db.get_optimization(optimization_id)
        if not optimization or optimization.get("deleted"):
            raise HTTPException(status_code=404, detail="Optimization not found")

        await require_project_access(
            optimization.get("project_id"),
            user_context=user_context,
            db=db,
        )

        result = db.list_optimization_iterations(
            optimization_id,
            page=page,
            per_page=per_page,
        )

        metadata = resolve_page_pagination_metadata(
            result.get("pagination"),
            page=page,
            per_page=per_page,
            item_count=len(result.get("iterations", [])),
        )
        # Back-compat: wrap iterations under data.{iterations} for tests expecting this shape
        return PageStandardResponse(
            success=True,
            data={"iterations": result.get("iterations", [])},
            metadata=metadata,
            pagination=build_page_pagination_meta(
                page=metadata["page"],
                per_page=metadata["per_page"],
                total=metadata["total"],
                total_pages=metadata["total_pages"],
            ),
        )
    except DatabaseError as exc:
        logger.error("Database error listing optimization iterations")
        raise map_db_error_to_http(exc, default_detail="Failed to list iterations") from exc
    except HTTPException:
        raise
    except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Unexpected error listing optimization iterations")
        raise HTTPException(status_code=500, detail="Failed to list iterations") from exc
@router.post(
    "/compare",
    response_model=StandardResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "compare": {
                            "summary": "Compare optimization strategies",
                            "value": {
                                "prompt_id": 12,
                                "test_case_ids": [1, 2, 3],
                                "strategies": ["iterative", "bayesian"],
                                "model_configuration": {"model_name": "gpt-4o-mini", "temperature": 0.3}
                            }
                        }
                    }
                }
            }
        },
        "responses": {
            "200": {
                "description": "Comparison jobs created"
            }
        }
    }
)
async def compare_strategies(
    request: CompareStrategiesRequest,
    http_request: Request,  # type: ignore[assignment]
    _: bool = Depends(_rl_optimizations),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: dict = Depends(get_prompt_studio_user),
) -> StandardResponse:
    """
    Compare multiple optimization strategies.

    Returns:
        Comparison job details
    """
    try:
        prompt_row = db.get_prompt_with_project(request.prompt_id, include_deleted=False)
        if not prompt_row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt {request.prompt_id} not found",
            )

        project_id = prompt_row["project_id"]
        await require_project_write_access(project_id, user_context=user_context, db=db)

        req_id = ensure_request_id(http_request) if http_request is not None else None
        tp = ensure_traceparent(http_request) if http_request is not None else ""
        optimization_ids: list[int] = []
        job_ids: list[str] = []
        adapter = PromptStudioJobsAdapter()

        strategies = request.strategies or []
        for strategy in strategies:
            combined_config = {
                "optimizer_type": strategy,
                "max_iterations": 10,
                "model_configuration": request.model_configuration,
            }

            optimization_record = db.create_optimization(
                project_id=project_id,
                name=f"Compare: {strategy}",
                initial_prompt_id=request.prompt_id,
                optimizer_type=strategy,
                optimization_config=combined_config,
                max_iterations=10,
                status="pending",
                client_id=db.client_id,
            )
            optimization_record = db.update_optimization(
                optimization_record["id"],
                {"test_case_ids": request.test_case_ids or []},
            )
            optimization_ids.append(optimization_record["id"])

            job_payload = {
                "optimization_id": optimization_record["id"],
                "optimizer_type": strategy,
                "test_case_ids": request.test_case_ids or [],
                "optimization_config": combined_config,
                "initial_prompt_id": request.prompt_id,
                "project_id": project_id,
                "created_by": user_context.get("user_id"),
                "submitted_at": datetime.utcnow().isoformat(),
            }
            if req_id:
                job_payload["request_id"] = req_id
            job = adapter.create_job(
                user_id=user_context.get("user_id"),
                job_type=JobType.OPTIMIZATION.value,
                entity_id=optimization_record["id"],
                payload=job_payload,
                project_id=project_id,
                priority=5,
                max_retries=3,
                request_id=req_id,
                trace_id=tp,
            )
            job_ids.append(str(job.get("uuid") or job.get("id")))
        with log_context(request_id=req_id, traceparent=tp, ps_component="endpoint", ps_job_kind="optimization"):
            logger.info(
                'User {} created strategy comparison for prompt {}',
                user_context.get("user_id"),
                request.prompt_id,
            )

        return StandardResponse(
            success=True,
            data={
                "optimization_ids": optimization_ids,
                "job_ids": job_ids,
                "strategies": strategies,
                "message": f"Comparing {len(strategies)} optimization strategies",
            },
        )

    except DatabaseError as exc:
        logger.error("Database error comparing strategies")
        raise map_db_error_to_http(exc, default_detail="Failed to compare strategies") from exc
    except HTTPException:
        raise
    except _OPTIMIZATION_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Unexpected error comparing strategies")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compare strategies",
        ) from exc

# Backward/compatibility alias used by tests: /compare-strategies
router.add_api_route(
    "/compare-strategies",
    compare_strategies,
    methods=["POST"],
    response_model=StandardResponse,
)

########################################################################################################################
# Note: Project access checks are provided via API_Deps.prompt_studio_deps.
# Do not redeclare require_project_access here to avoid shadowing the dependency.
