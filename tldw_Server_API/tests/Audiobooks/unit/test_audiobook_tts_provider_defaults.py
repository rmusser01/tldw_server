import sys

import pytest

if sys.version_info < (3, 10):
    pytest.skip("Audiobook TTS provider default tests require Python 3.10+", allow_module_level=True)

from tldw_Server_API.app.services.audiobook_jobs_worker import (
    _resolve_audiobook_generation_defaults,
)


@pytest.mark.unit
@pytest.mark.parametrize("provider_name", ["omnivoice", "omni-voice", "omni_voice"])
def test_resolve_audiobook_generation_defaults_preserves_omnivoice(provider_name: str) -> None:
    provider, model, voice = _resolve_audiobook_generation_defaults(
        provider=provider_name,
        model=None,
        voice=None,
    )

    assert provider == "omnivoice"
    assert model == "omnivoice"
    assert voice == "auto"


@pytest.mark.unit
@pytest.mark.parametrize("model_name", ["omnivoice", "omni-voice", "omni_voice"])
def test_resolve_audiobook_generation_defaults_canonicalizes_explicit_omnivoice_model(model_name: str) -> None:
    provider, model, voice = _resolve_audiobook_generation_defaults(
        provider="omnivoice",
        model=model_name,
        voice=None,
    )

    assert provider == "omnivoice"
    assert model == "omnivoice"
    assert voice == "auto"


@pytest.mark.unit
@pytest.mark.parametrize("model_name", ["omnivoice", "omni-voice", "omni_voice"])
def test_resolve_audiobook_generation_defaults_infers_provider_from_omnivoice_alias(model_name: str) -> None:
    provider, model, voice = _resolve_audiobook_generation_defaults(
        provider=None,
        model=model_name,
        voice=None,
    )

    assert provider == "omnivoice"
    assert model == "omnivoice"
    assert voice == "auto"
