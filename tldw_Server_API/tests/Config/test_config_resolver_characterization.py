"""Characterisation golden for env-over-config precedence (TASK-13950.21).

Every typed section loader and config.py's precedence helpers run against spy fakes over eleven
scenarios. The golden pins each result and every env key and config option asked for, in order,
so moving the precedence logic onto ``config_sections.precedence.resolve`` cannot drift. It was
generated on dev before that refactor; a mismatch means fixing the call, never the golden.

Regenerate (only on unmodified code):
``PYTHONPATH=. python -m tldw_Server_API.tests.Config.test_config_resolver_characterization --write-golden``
"""

from __future__ import annotations

import configparser
import contextlib
import dataclasses
import json
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import pytest

from tldw_Server_API.app.core import config as config_mod
from tldw_Server_API.app.core.AuthNZ import settings as authnz_settings
from tldw_Server_API.app.core.config_sections import (
    load_audio_config,
    load_auth_config,
    load_chat_config,
    load_chunking_config,
    load_database_config,
    load_embeddings_config,
    load_jobs_config,
    load_logging_config,
    load_moderation_config,
    load_providers_config,
    load_rag_config,
    load_server_config,
    load_slides_config,
    load_stt_config,
)

pytestmark = pytest.mark.unit

GOLDEN_PATH = Path(__file__).with_name("config_resolver_golden.json")
REPO_ROOT = Path(__file__).resolve().parents[3]


class _AnyEnv(dict):
    """Every key reads ``value`` (absent when it is None); reads are recorded in first-access order."""

    def __init__(self, value: str | None) -> None:
        super().__init__()
        self.value = value
        self.asked: list[str] = []

    def _ask(self, key: str) -> str | None:
        if key not in self.asked:
            self.asked.append(key)
        return self.value

    def __getitem__(self, key: str) -> str:
        value = self._ask(key)
        if value is None:
            raise KeyError(key)
        return value

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        value = self._ask(key)
        return default if value is None else value

    def __contains__(self, key: object) -> bool:
        return self._ask(str(key)) is not None


class _AnyParser:
    """Every option reads ``value`` (absent when it is None); ``get`` raises it when it is an exception."""

    def __init__(self, value: str | None | BaseException) -> None:
        self.value = value
        self.asked: list[list[str]] = []

    def _ask(self, section: str, option: str) -> None:
        if [section, option] not in self.asked:
            self.asked.append([section, option])

    def get(self, section: str, option: str, fallback: Any = None, **_: Any) -> Any:
        self._ask(section, option)
        if isinstance(self.value, BaseException):
            raise self.value
        return fallback if self.value is None else self.value

    def has_section(self, section: str) -> bool:
        return self.value is not None

    def has_option(self, section: str, option: str) -> bool:
        self._ask(section, option)
        return self.value is not None

    def getboolean(self, section: str, option: str, fallback: Any = None, **_: Any) -> Any:
        raw = self.get(section, option, fallback=None)
        if raw is None:
            return fallback
        try:
            return configparser.ConfigParser.BOOLEAN_STATES[str(raw).lower()]
        except KeyError:
            raise ValueError(f"Not a boolean: {raw}") from None

    def __getitem__(self, section: str) -> "_AnySection":
        return _AnySection(self, section)


class _AnySection:
    """What ``parser[section]`` returns: ``get`` asks the parser."""

    def __init__(self, parser: _AnyParser, section: str) -> None:
        self.parser, self.section = parser, section

    def get(self, option: str, fallback: Any = None, **_: Any) -> Any:
        return self.parser.get(self.section, option, fallback=fallback)


SCENARIOS: dict[str, tuple[str | None, str | None | BaseException]] = {
    "none": (None, None),
    "env_blank": ("", None),
    "env_space": ("  7  ", None),
    "env_word": ("abc", None),
    "env_true": ("true", None),
    "cfg_blank": (None, ""),
    "cfg_space": (None, "  7  "),
    "cfg_false": (None, "false"),
    "both": ("8", "9"),
    "env_blank_cfg": ("", "9"),
    "cfg_raises": (None, configparser.InterpolationSyntaxError("o", "S", "bad %")),
}

def _loader(parser: Any) -> Callable[..., Any]:
    def _load(*_: Any, **__: Any) -> Any:
        return parser

    _load.cache_clear = lambda: None  # type: ignore[attr-defined]  # clear_config_cache() calls it
    return _load


def _in_runtime(call: Callable[[Any], object], *, not_pytest: bool = False) -> Callable[[Mapping[str, str], Any], object]:
    """Run ``call(parser)`` with os.environ swapped for the spy env and config.txt loading swapped for the spy parser."""

    def run(env: Mapping[str, str], parser: Any) -> object:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(os, "environ", env)
            for name in ("load_comprehensive_config", "_load_config_parser"):
                mp.setattr(config_mod, name, _loader(parser))
            mp.setattr(authnz_settings, "load_comprehensive_config", _loader(parser))
            if not_pytest:  # rg_enabled returns False under pytest before it reads config.txt
                mp.setattr(config_mod, "is_test_mode", lambda: False)
                mp.setattr(config_mod, "is_explicit_pytest_runtime", lambda: False)
                mp.delitem(sys.modules, "pytest")
            return call(parser)

    return run


_GETTERS: dict[str, Callable[[], object]] = {
    name: getattr(config_mod, name)
    for name in (
        "rag_enable_structure_index",
        "rag_strict_extractive",
        "rag_require_hard_citations",
        "rag_low_confidence_behavior",
        "rag_agentic_cache_backend",
        "rag_agentic_cache_ttl_sec",
        "implicit_feedback_enabled",
        "web_outbound_policy_mode",
        "web_browser_transport_mode",
        "rg_enabled",
        "rg_policy_store",
        "rg_policy_reload_enabled",
        "rg_policy_reload_interval_sec",
        "rg_backend",
        "rg_redis_fail_mode",
        "rg_policy_path",
        "resolve_governance_rollout_mode",
        "usage_quotas_enabled",
        "get_llamacpp_handler_settings",
        "get_ingestion_source_allowed_roots",
        "get_workspace_project_root_allowed_roots",
        "resolve_chat_run_first_rollout_mode",
        "resolve_chat_run_first_provider_allowlist",
        "resolve_chat_run_first_presentation_variant",
        "_resolve_disable_cors_value_and_source",
        "_resolve_cors_allow_credentials_value_and_source",
    )
}

GETTER_SUBJECTS: dict[str, Callable[[Mapping[str, str], Any], object]] = {
    f"getter:{name}": _in_runtime(lambda parser, fn=fn: fn()) for name, fn in _GETTERS.items()
}
GETTER_SUBJECTS["getter:rg_enabled_outside_pytest"] = _in_runtime(lambda parser: config_mod.rg_enabled(), not_pytest=True)
GETTER_SUBJECTS["getter:_load_claims_review_metrics_settings"] = _in_runtime(
    lambda parser: config_mod._load_claims_review_metrics_settings(parser)
)
GETTER_SUBJECTS["authnz:_load_overrides_from_config"] = _in_runtime(
    lambda parser: authnz_settings._load_overrides_from_config()
)

SUBJECTS: dict[str, Callable[[Mapping[str, str], Any], object]] = {
    "load_audio_config": lambda env, parser: load_audio_config(parser, env),
    "load_auth_config": lambda env, parser: load_auth_config(parser, env),
    "load_chat_config": lambda env, parser: load_chat_config(parser, env),
    "load_chunking_config": lambda env, parser: load_chunking_config(parser, env),
    "load_database_config": lambda env, parser: load_database_config(parser, env),
    "load_embeddings_config": lambda env, parser: load_embeddings_config(parser, env),
    "load_jobs_config": lambda env, parser: load_jobs_config(parser, env),
    "load_logging_config": lambda env, parser: load_logging_config(parser, env),
    "load_moderation_config": lambda env, parser: load_moderation_config(parser, env),
    "load_providers_config": lambda env, parser: load_providers_config(parser, env),
    "load_rag_config": lambda env, parser: load_rag_config(parser, env),
    "load_server_config": lambda env, parser: load_server_config(parser, env),
    "load_slides_config": lambda env, parser: load_slides_config(parser, env),
    "load_stt_config": lambda env, parser: load_stt_config(parser, env),
    "_first_nonempty_env": lambda env, parser: config_mod._first_nonempty_env(("K1", "K2"), environment=env),
    "_env_or_config_value": lambda env, parser: config_mod._env_or_config_value(
        ("K1",), parser, "S", "o", fallback="d", environment=env
    ),
    "_env_or_config_option_value": lambda env, parser: config_mod._env_or_config_option_value(
        ("K1",), parser, "S", ("o1", "o2"), fallback="d", environment=env
    ),
    "_first_config_option_value": lambda env, parser: config_mod._first_config_option_value(
        parser, "S", ("o1", "o2"), fallback="d"
    ),
    **GETTER_SUBJECTS,
}


def _plain(value: object) -> object:
    """JSON-shaped: tuples become lists, Paths strings, and the repo root ``<repo>``."""
    def portable(item: object) -> object:
        if isinstance(item, list):
            return [portable(child) for child in item]
        if isinstance(item, dict):
            return {key: portable(child) for key, child in item.items()}
        if isinstance(item, str):
            path = type(REPO_ROOT)(item)
            if path.is_absolute():
                try:
                    relative = path.relative_to(REPO_ROOT)
                except ValueError:
                    return item
                return "<repo>" + ("/" + relative.as_posix() if relative.parts else "")
        return item

    return portable(json.loads(json.dumps(value, default=str)))


@pytest.mark.parametrize("root", [
    PurePosixPath("/tmp/checkout with spaces"),  # nosec B108 - in-memory path flavor; no filesystem operation.
    PureWindowsPath("D:/a/checkout with spaces"),
    PureWindowsPath("//server/share/checkout with spaces"),
])
def test_plain_repo_paths_are_portable_without_rewriting_unrelated_values(monkeypatch, root) -> None:
    """Only actual root-relative path components lose the checkout prefix."""
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", root)
    sibling = str(root.with_name(root.name + "-other") / "policy.yaml")
    embedded = f"https://example.test/?value={root}"
    value = {
        "path": root / "Config_Files" / "policy.yaml",
        "nested": (str(root), {"path": str(root / "  7  ")}),
        "sibling": sibling,
        "embedded": embedded,
        "literal": 'literal\\value "quoted"\n',
        "scalars": [None, True, 17],
    }
    assert _plain(value) == {
        "path": "<repo>/Config_Files/policy.yaml",
        "nested": ["<repo>", {"path": "<repo>/  7  "}],
        "sibling": sibling,
        "embedded": embedded,
        "literal": 'literal\\value "quoted"\n',
        "scalars": [None, True, 17],
    }


def outcome(subject: Callable[[Mapping[str, str], Any], object], env_value: str | None, parser_value: Any) -> dict:
    env, parser = _AnyEnv(env_value), _AnyParser(parser_value)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(os, "environ", {})  # a subject that drops its env argument reads nothing, so it cannot match
        try:
            result = subject(env, parser)
        except Exception as exc:  # noqa: BLE001 - the exception type is the pinned behaviour
            value: object = {"raises": type(exc).__name__}
        else:
            value = dataclasses.asdict(result) if dataclasses.is_dataclass(result) else result  # type: ignore[arg-type]
    return {"value": _plain(value), "env": env.asked, "config": parser.asked}


def outcomes() -> dict[str, dict[str, dict]]:
    return {
        name: {scenario: outcome(subject, *values) for scenario, values in SCENARIOS.items()}
        for name, subject in SUBJECTS.items()
    }


def _dump(golden: Mapping[str, Mapping[str, dict]]) -> str:
    """One line per outcome, so a drift shows as one changed line."""
    blocks = []
    for name, scenarios in golden.items():
        rows = ",\n".join(f"    {json.dumps(s)}: {json.dumps(o)}" for s, o in scenarios.items())
        blocks.append(f"  {json.dumps(name)}: {{\n{rows}\n  }}")
    return "{\n" + ",\n".join(blocks) + "\n}\n"


def _golden() -> dict[str, dict[str, dict]]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_loaders_and_helpers_match_the_dev_golden() -> None:
    golden, actual = _golden(), outcomes()
    mismatches = [
        f"{name} / {scenario}:\n  golden {golden.get(name, {}).get(scenario)}\n  actual {got}"
        for name, scenarios in actual.items()
        for scenario, got in scenarios.items()
        if golden.get(name, {}).get(scenario) != got
    ]
    assert set(golden) == set(SUBJECTS)
    assert mismatches == [], "\n".join(mismatches)


def test_the_golden_holds_no_harness_errors() -> None:
    harness_errors = [{"raises": "AttributeError"}, {"raises": "TypeError"}]
    bad = [
        f"{name} / {scenario}"
        for name, scenarios in _golden().items()
        for scenario, result in scenarios.items()
        if result["value"] in harness_errors
    ]
    assert bad == [], f"the fakes are missing a method: {bad}"


if __name__ == "__main__":
    if sys.argv[1:] != ["--write-golden"]:
        raise SystemExit("usage: python -m tldw_Server_API.tests.Config.test_config_resolver_characterization --write-golden")
    GOLDEN_PATH.write_text(_dump(outcomes()), encoding="utf-8")
    print(f"wrote {GOLDEN_PATH}")
