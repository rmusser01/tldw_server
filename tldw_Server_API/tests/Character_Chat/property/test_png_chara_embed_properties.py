"""Property test for the PNG tEXt 'chara' embed/extract round-trip (RA4).

This one IS bidirectional: ``_encode_png_with_chara_metadata`` writes a base64
'chara' tEXt chunk, and ``extract_json_from_image_file`` reads it back. Arbitrary
card JSON survives embed -> extract semantically (extract validates JSON, so the
round-trip is asserted via ``json.loads`` equality, immune to key ordering).
"""
from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings as hyp_settings, strategies as st

from tldw_Server_API.app.api.v1.endpoints.characters_endpoint import (
    _encode_png_with_chara_metadata,
)
from tldw_Server_API.app.core.Character_Chat.modules.character_io import (
    extract_json_from_image_file,
)

pytestmark = [pytest.mark.unit, pytest.mark.property]

_MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)

# JSON-serializable card payloads (arbitrary, incl. unicode / nesting)
_json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(max_size=40),
    lambda children: st.lists(children, max_size=5)
    | st.dictionaries(st.text(min_size=1, max_size=10), children, max_size=5),
    max_leaves=15,
)
_card_payloads = st.dictionaries(st.text(min_size=1, max_size=15), _json_values, min_size=1, max_size=8)


@hyp_settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(card=_card_payloads, use_carrier=st.booleans())
def test_chara_embed_extract_round_trip(card: dict[str, Any], use_carrier: bool) -> None:
    card_json = json.dumps(card, ensure_ascii=False)
    carrier = _MINIMAL_PNG if use_carrier else None

    png = _encode_png_with_chara_metadata(carrier, card_json)
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "encoder did not emit a valid PNG signature"

    extracted = extract_json_from_image_file(png)
    assert extracted is not None, "embedded chara metadata could not be extracted"
    assert json.loads(extracted) == card, "card did not survive the embed/extract round-trip"


@hyp_settings(max_examples=50, deadline=None)
@given(card=_card_payloads)
def test_encode_output_is_deterministic(card: dict[str, Any]) -> None:
    card_json = json.dumps(card, ensure_ascii=False)
    assert _encode_png_with_chara_metadata(None, card_json) == _encode_png_with_chara_metadata(
        None, card_json
    )
