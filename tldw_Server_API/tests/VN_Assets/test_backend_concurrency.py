from __future__ import annotations


def test_local_backend_gate_allows_one_generation_at_a_time() -> None:
    from tldw_Server_API.app.core.VN_Assets.concurrency import BackendGenerationGate

    gate = BackendGenerationGate(default_local_limit=1)

    first = gate.try_acquire("stable_diffusion_cpp", model="local-model")
    second = gate.try_acquire("stable_diffusion_cpp", model="local-model")

    assert first.acquired is True
    assert second.acquired is False

    first.release()
    third = gate.try_acquire("stable_diffusion_cpp", model="local-model")

    assert third.acquired is True
