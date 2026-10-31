from __future__ import annotations

from itertools import permutations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_Server_API.app.core.TTS.gateway_config import (
    canonicalize_gateway_id,
    decode_json_pointer,
    normalize_gateway_specs,
    validate_relative_gateway_path,
)

SLUG_HEAD = "abcdefghijklmnopqrstuvwxyz0123456789"
SLUG_TAIL = f"{SLUG_HEAD}-"


def _avoid_builtin_slug(slug: str) -> str:
    return "openrouter-" if slug == "openrouter" else slug


VALID_SLUGS = st.builds(
    lambda head, tail: _avoid_builtin_slug(head + tail),
    st.sampled_from(tuple(SLUG_HEAD)),
    st.text(alphabet=SLUG_TAIL, min_size=0, max_size=62),
)
INVALID_SLUGS = st.one_of(
    VALID_SLUGS.map(lambda slug: f" {slug}"),
    VALID_SLUGS.map(lambda slug: f"{slug}_"),
    st.just("a" * 64),
)
POINTER_TOKEN_LISTS = st.lists(
    st.sampled_from(("", "a", "0", "/", "~", "~0", "~1", "x/y", "a~b")),
    max_size=4,
)


@given(VALID_SLUGS)
@pytest.mark.unit
def test_valid_slugs_have_stable_canonical_ids(slug):
    canonical = canonicalize_gateway_id(slug)
    assert canonical == f"gateway:{slug}"
    assert canonicalize_gateway_id(canonical) == canonical


@given(INVALID_SLUGS)
@pytest.mark.unit
def test_invalid_slugs_are_rejected(value):
    with pytest.raises(ValueError):
        canonicalize_gateway_id(value)


@given(
    st.sampled_from(
        (
            "/absolute",
            "../parent",
            "./current",
            "//host/path",
            "https://host/path",
            "path?x=1",
            "path#fragment",
            "path\\segment",
        )
    )
)
@pytest.mark.unit
def test_forbidden_relative_path_components_are_always_rejected(value):
    with pytest.raises(ValueError):
        validate_relative_gateway_path(value, field_name="path")


@pytest.mark.parametrize("targets", tuple(permutations(("a", "b", "c", "d"))))
@pytest.mark.unit
def test_fallback_target_count_is_bounded(targets):
    gateway = {
        "enabled": True,
        "base_url": "https://speech.example/v1/",
        "speech_path": "audio/speech",
        "api_key": "secret",
        "default_model": "Model",
        "default_voice": "voice",
        "capability_defaults": {"formats": ["mp3"]},
        "fallback": {"targets": [{"backend": f"gateway:{name}", "model": "Model"} for name in targets]},
    }
    definitions = {"primary": gateway}
    definitions.update({name: {**gateway, "fallback": {}} for name in targets})
    with pytest.raises(ValueError, match="at most 3"):
        normalize_gateway_specs({}, definitions)


@given(POINTER_TOKEN_LISTS)
@pytest.mark.unit
def test_json_pointer_unescaping_round_trips_tokens(tokens):
    pointer = ""
    if tokens:
        pointer = "/" + "/".join(token.replace("~", "~0").replace("/", "~1") for token in tokens)
    assert decode_json_pointer(pointer) == tuple(tokens)
