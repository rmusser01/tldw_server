"""Exercise runtime verification with real installed crypto and embedded storage."""

import pytest

pytestmark = pytest.mark.integration


def test_crypto_probe_signs_and_verifies_with_cryptography() -> None:
    from Helper_Scripts.Supply_Chain.runtime_probe import probe_crypto

    assert probe_crypto() == "jose.backends.cryptography_backend"


def test_storage_probe_uses_real_per_user_managers(monkeypatch: pytest.MonkeyPatch) -> None:
    from Helper_Scripts.Supply_Chain.runtime_probe import probe_chroma

    monkeypatch.delenv("CHROMADB_FORCE_STUB", raising=False)
    assert probe_chroma() == "chromadb.api.rust"


def test_storage_probe_rejects_test_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    from Helper_Scripts.Supply_Chain.runtime_probe import probe_chroma

    monkeypatch.setenv("CHROMADB_FORCE_STUB", "1")
    with pytest.raises(ValueError, match="real embedded"):
        probe_chroma()
