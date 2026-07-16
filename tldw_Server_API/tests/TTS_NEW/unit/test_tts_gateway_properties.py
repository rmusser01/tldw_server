from __future__ import annotations

import re

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_Server_API.app.core.TTS.gateway_config import (
    canonicalize_gateway_id,
    decode_json_pointer,
    normalize_gateway_specs,
    validate_relative_gateway_path,
)

SLUG_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,62}\Z")


@given(st.from_regex(SLUG_RE, fullmatch=True).filter(lambda value: value != "openrouter"))
@pytest.mark.unit
def test_valid_slugs_have_stable_canonical_ids(slug):
    canonical = canonicalize_gateway_id(slug)
    assert canonical == f"gateway:{slug}"
    assert canonicalize_gateway_id(canonical) == canonical


@given(
    st.text(min_size=1).filter(
        lambda value: not SLUG_RE.fullmatch(value) and not value.startswith("gateway:")
    )
)
@pytest.mark.unit
def test_invalid_slugs_are_rejected(value):
    with pytest.raises(ValueError):
        canonicalize_gateway_id(value)


@given(
    st.sampled_from(
        ["/", "../", "./", "//host/", "https://host/", "?x=1", "#x", "\\"]
    ),
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz", max_size=20),
)
@pytest.mark.unit
def test_forbidden_relative_path_components_are_always_rejected(prefix, suffix):
    with pytest.raises(ValueError):
        validate_relative_gateway_path(f"{prefix}{suffix}", field_name="path")


@given(st.lists(st.sampled_from(["a", "b", "c", "d"]), min_size=4, max_size=12))
@pytest.mark.unit
def test_fallback_target_count_is_bounded(targets):
    unique = list(dict.fromkeys(targets))
    if len(unique) < 4:
        return
    gateway = {
        "enabled": True,
        "base_url": "https://speech.example/v1/",
        "speech_path": "audio/speech",
        "api_key": "secret",
        "default_model": "Model",
        "default_voice": "voice",
        "capability_defaults": {"formats": ["mp3"]},
        "fallback": {
            "targets": [{"backend": f"gateway:{name}"} for name in unique]
        },
    }
    definitions = {"primary": gateway}
    definitions.update(
        {
            name: {**gateway, "fallback": {}}
            for name in unique
        }
    )
    with pytest.raises(ValueError, match="at most 3"):
        normalize_gateway_specs({}, definitions)


@given(
    st.lists(
        st.sampled_from(["", "a", "XYZ", "/", "~", "~0", "x/y", "a~b"]),
        min_size=0,
        max_size=6,
    )
)
@pytest.mark.unit
def test_json_pointer_unescaping_round_trips_tokens(tokens):
    pointer = ""
    if tokens:
        pointer = "/" + "/".join(
            token.replace("~", "~0").replace("/", "~1") for token in tokens
        )
    assert decode_json_pointer(pointer) == tuple(tokens)
