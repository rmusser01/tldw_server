from tldw_Server_API.app.core.VLLM_Management.capabilities import derive_effective_capabilities


def test_effective_capabilities_require_positive_declaration_and_probe():
    effective = derive_effective_capabilities(
        declared_capabilities={"chat": True, "embeddings": True, "vision": True},
        probed_capabilities={"chat": True, "embeddings": False},
    )

    assert effective["chat"] is True
    assert effective["embeddings"] is False
    assert effective["vision"] is False


def test_effective_capabilities_include_declared_only_when_probe_missing():
    effective = derive_effective_capabilities(
        declared_capabilities={"chat": True, "audio": False},
        probed_capabilities={},
    )

    assert effective["chat"] is True
    assert effective["audio"] is False
