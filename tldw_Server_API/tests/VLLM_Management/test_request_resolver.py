import pytest

from tldw_Server_API.app.core.VLLM_Management.models import VLLMInstanceRecord
from tldw_Server_API.app.core.VLLM_Management.resolver import resolve_vllm_instance_for_request


class InMemoryVLLMInstanceRepository:
    def __init__(self, instances: dict[str, VLLMInstanceRecord], default_instance_id: str | None = None):
        self._instances = instances
        self._default_instance_id = default_instance_id

    def get_instance(self, instance_id: str) -> VLLMInstanceRecord | None:
        return self._instances.get(instance_id)

    def get_default_instance_id(self) -> str | None:
        return self._default_instance_id


def fake_instance(
    instance_id: str,
    base_url: str,
    *,
    capabilities: dict[str, bool] | None = None,
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    observed_state: str = "healthy",
) -> VLLMInstanceRecord:
    return VLLMInstanceRecord(
        instance_id=instance_id,
        name=instance_id,
        execution_mode="local",
        transport_config={},
        launch_spec={"base_url": base_url, "model": model},
        routing_policy={},
        declared_capabilities=capabilities or {"chat": True},
        desired_state="running",
        observed_state=observed_state,
        created_at="2026-03-10T00:00:00+00:00",
        updated_at="2026-03-10T00:00:00+00:00",
        effective_capabilities=dict(capabilities or {"chat": True}),
    )


def test_resolver_prefers_request_instance_id_over_default():
    repo = InMemoryVLLMInstanceRepository(
        instances={
            "default-id": fake_instance("default-id", "http://127.0.0.1:8000/v1"),
            "vision-id": fake_instance("vision-id", "http://10.0.0.9:8000/v1"),
        },
        default_instance_id="default-id",
    )

    resolved = resolve_vllm_instance_for_request(
        provider="vllm",
        provider_instance_id="vision-id",
        required_capability="chat",
        repository=repo,
    )

    assert resolved is not None
    assert resolved.base_url == "http://10.0.0.9:8000/v1"
    assert resolved.instance_id == "vision-id"


def test_resolver_rejects_missing_required_capability():
    repo = InMemoryVLLMInstanceRepository(
        instances={
            "embed-id": fake_instance(
                "embed-id",
                "http://127.0.0.1:8010/v1",
                capabilities={"chat": False, "embeddings": True},
            )
        }
    )

    with pytest.raises(ValueError, match="required capability 'chat'"):
        resolve_vllm_instance_for_request(
            provider="vllm",
            provider_instance_id="embed-id",
            required_capability="chat",
            repository=repo,
        )


def test_resolver_accepts_multiple_required_capabilities():
    repo = InMemoryVLLMInstanceRepository(
        instances={
            "vision-id": fake_instance(
                "vision-id",
                "http://127.0.0.1:8011/v1",
                capabilities={"chat": True, "vision": True},
            )
        }
    )

    resolved = resolve_vllm_instance_for_request(
        provider="vllm",
        provider_instance_id="vision-id",
        required_capability=("chat", "vision"),
        repository=repo,
    )

    assert resolved is not None
    assert resolved.instance_id == "vision-id"


def test_resolver_rejects_when_any_required_capability_is_missing():
    repo = InMemoryVLLMInstanceRepository(
        instances={
            "chat-only-id": fake_instance(
                "chat-only-id",
                "http://127.0.0.1:8012/v1",
                capabilities={"chat": True, "vision": False},
            )
        }
    )

    with pytest.raises(ValueError, match="required capability 'vision'"):
        resolve_vllm_instance_for_request(
            provider="vllm",
            provider_instance_id="chat-only-id",
            required_capability=("chat", "vision"),
            repository=repo,
        )


@pytest.mark.parametrize("observed_state", ["starting", "stopped", "stopping", "failed", "unhealthy"])
def test_resolver_rejects_instances_that_are_not_runtime_healthy(observed_state: str):
    repo = InMemoryVLLMInstanceRepository(
        instances={
            "managed-id": fake_instance(
                "managed-id",
                "http://127.0.0.1:8013/v1",
                capabilities={"chat": True},
                observed_state=observed_state,
            )
        }
    )

    with pytest.raises(ValueError, match=f"Managed vLLM instance 'managed-id' is not healthy"):
        resolve_vllm_instance_for_request(
            provider="vllm",
            provider_instance_id="managed-id",
            required_capability="chat",
            repository=repo,
        )
