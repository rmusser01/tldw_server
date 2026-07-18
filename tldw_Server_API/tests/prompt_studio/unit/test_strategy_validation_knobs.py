import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
    prompt_studio_optimization as pso,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "cfg",
    [
        {"strategy_params": {"beam_width": 3, "prune_threshold": -0.1}},
        {"strategy_params": {"beam_width": 3, "prune_threshold": 1.1}},
        {"strategy_params": {"beam_width": 3, "max_candidates": 1}},
        {"strategy_params": {"beam_width": 4, "max_candidates": 2}},
    ],
)
def test_beam_search_with_legacy_knobs_is_rejected(cfg):
    with pytest.raises(HTTPException):
        pso._validate_strategy_config("beam_search", cfg)


def test_beam_search_with_formerly_valid_knobs_is_rejected():
    cfg = {
        "strategy_params": {
            "beam_width": 3,
            "max_candidates": 5,
            "prune_threshold": 0.5,
        }
    }
    with pytest.raises(HTTPException, match="Unsupported optimization strategy"):
        pso._validate_strategy_config("beam_search", cfg)


@pytest.mark.parametrize(
    "cfg",
    [
        {"strategy_params": {"schedule": "unknown"}},
        {"strategy_params": {"initial_temp": 1.0, "min_temp": 2.0}},
    ],
)
def test_anneal_with_legacy_knobs_is_rejected(cfg):
    with pytest.raises(HTTPException):
        pso._validate_strategy_config("anneal", cfg)


def test_anneal_with_formerly_valid_knobs_is_rejected():
    cfg = {
        "strategy_params": {
            "schedule": "cosine",
            "initial_temp": 5.0,
            "min_temp": 1.0,
        }
    }
    with pytest.raises(HTTPException, match="Unsupported optimization strategy"):
        pso._validate_strategy_config("anneal", cfg)


@pytest.mark.parametrize(
    "cfg",
    [
        {"strategy_params": {"selection": "invalid"}},
        {"strategy_params": {"elitism": -1}},
        {"strategy_params": {"crossover_rate": 1.5}},
    ],
)
def test_genetic_with_legacy_knobs_is_rejected(cfg):
    with pytest.raises(HTTPException):
        pso._validate_strategy_config("genetic", cfg)


def test_genetic_with_formerly_valid_knobs_is_rejected():
    cfg = {
        "strategy_params": {
            "selection": "tournament",
            "elitism": 1,
            "crossover_rate": 0.4,
        }
    }
    with pytest.raises(HTTPException, match="Unsupported optimization strategy"):
        pso._validate_strategy_config("genetic", cfg)


@pytest.mark.parametrize(
    "cfg",
    [
        {"strategy_params": {"mcts_simulations": 500}},
        {"strategy_params": {"mcts_max_depth": 20}},
        {"strategy_params": {"mcts_exploration_c": 0.0}},
        {"strategy_params": {"prompt_candidates_per_node": 20}},
        {"strategy_params": {"token_budget": 0}},
        {"strategy_params": {"min_quality": -0.1}},
        {"strategy_params": {"min_quality": 1.1}},
        {"strategy_params": {"feedback_enabled": "definitely"}},
        {"strategy_params": {"ws_throttle_every": 0}},
        {"strategy_params": {"trace_top_k": 0}},
    ],
)
def test_mcts_knobs_reject_out_of_range_values(cfg, monkeypatch):
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_MCTS", "true")

    with pytest.raises(HTTPException):
        pso._validate_strategy_config("mcts", cfg)


def test_mcts_knobs_accept_supported_values(monkeypatch):
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_MCTS", "true")
    cfg = {
        "strategy_params": {
            "mcts_simulations": 50,
            "mcts_max_depth": 4,
            "mcts_exploration_c": 1.5,
            "prompt_candidates_per_node": 3,
            "token_budget": 1000,
            "min_quality": 0.25,
            "feedback_enabled": False,
            "ws_throttle_every": 2,
            "trace_top_k": 5,
            "scorer_model": "gpt-4o-mini",
        }
    }

    pso._validate_strategy_config("mcts", cfg)


def test_mcts_rollout_model_is_explicitly_rejected(monkeypatch):
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_MCTS", "true")
    cfg = {"strategy_params": {"rollout_model": "gpt-4o-mini"}}

    with pytest.raises(HTTPException, match="unsupported"):
        pso._validate_strategy_config("mcts", cfg)
