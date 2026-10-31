import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
    prompt_studio_optimization as pso,
)

pytestmark = pytest.mark.unit


def test_beam_search_with_formerly_valid_reranker_is_rejected():
    cfg = {
        "strategy_params": {
            "beam_width": 4,
            "max_candidates": 8,
            "diversity_rate": 0.2,
            "length_penalty": 1.5,
            "candidate_reranker": "hybrid",
        }
    }
    with pytest.raises(HTTPException, match="Unsupported optimization strategy"):
        pso._validate_strategy_config("beam_search", cfg)


@pytest.mark.parametrize(
    "cfg",
    [
        {"strategy_params": {"length_penalty": -0.1}},
        {"strategy_params": {"candidate_reranker": "unknown"}},
    ],
)
def test_beam_search_length_penalty_and_reranker_invalid(cfg):
    with pytest.raises(HTTPException):
        pso._validate_strategy_config("beam_search", cfg)


def test_anneal_with_formerly_valid_schedule_is_rejected():
    cfg = {
        "strategy_params": {
            "schedule": "linear",
            "initial_temp": 10.0,
            "min_temp": 5.0,
            "step_size": 1.0,
            "epochs": 5,
        }
    }
    with pytest.raises(HTTPException, match="Unsupported optimization strategy"):
        pso._validate_strategy_config("anneal", cfg)


def test_anneal_step_schedule_consistency_invalid():
    cfg = {
        "strategy_params": {
            "schedule": "linear",
            "initial_temp": 2.0,
            "min_temp": 1.0,
            "step_size": 1.0,
            "epochs": 2,
        }
    }
    with pytest.raises(HTTPException):
        pso._validate_strategy_config("anneal", cfg)


def test_genetic_with_formerly_valid_crossover_is_rejected():
    cfg = {"strategy_params": {"crossover_operator": "two_point"}}
    with pytest.raises(HTTPException, match="Unsupported optimization strategy"):
        pso._validate_strategy_config("genetic", cfg)


def test_genetic_crossover_operator_invalid():
    cfg = {"strategy_params": {"crossover_operator": "bad"}}
    with pytest.raises(HTTPException):
        pso._validate_strategy_config("genetic", cfg)


@pytest.mark.parametrize("strategy", ["hyperparameter", "random_search"])
def test_hp_and_random_with_formerly_valid_range_are_rejected(strategy):
    cfg = {"strategy_params": {"max_tokens_range": [64, 512]}}
    with pytest.raises(HTTPException, match="Unsupported optimization strategy"):
        pso._validate_strategy_config(strategy, cfg)


@pytest.mark.parametrize(
    "strategy,cfg",
    [
        ("hyperparameter", {"strategy_params": {"max_tokens_range": [0, 100]}}),
        ("random_search", {"strategy_params": {"max_tokens_range": [100, 100]}}),
        ("random_search", {"strategy_params": {"max_tokens_range": [512, 64]}}),
        (
            "hyperparameter",
            {"strategy_params": {"max_tokens_range": ["a", 64]}},
        ),
        ("random_search", {"strategy_params": {"max_tokens_range": [64]}}),
    ],
)
def test_hp_and_random_max_tokens_range_invalid(strategy, cfg):
    with pytest.raises(HTTPException):
        pso._validate_strategy_config(strategy, cfg)
