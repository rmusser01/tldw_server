from tldw_Server_API.app.core.VLLM_Management.models import VLLMInstanceCreate
from tldw_Server_API.app.core.VLLM_Management.sqlite_repo import SqliteVLLMInstanceRepository


def _instance_payload(name: str, *, execution_mode: str = "local") -> VLLMInstanceCreate:
    transport_config = {}
    if execution_mode == "ssh":
        transport_config = {
            "host": "gpu-a100-01.internal",
            "port": 22,
            "username": "ubuntu",
            "launcher_path": "/usr/local/bin/tldw-vllm-launcher",
            "auth": {"secret_ref": "ssh-key-vllm-prod"},
        }

    return VLLMInstanceCreate(
        name=name,
        execution_mode=execution_mode,
        transport_config=transport_config,
        launch_spec={"model": "Qwen/Qwen2.5-VL-7B-Instruct", "port": 8001},
        routing_policy={"is_default": False},
        declared_capabilities={"chat": True, "embeddings": False, "vision": True},
    )


def test_repository_round_trips_instance_and_default_route(tmp_path):
    repo = SqliteVLLMInstanceRepository(db_path=tmp_path / "vllm_instances.db")
    created = repo.create_instance(_instance_payload("vision-a100", execution_mode="ssh"))

    repo.set_default_instance(created.instance_id)
    fetched = repo.get_instance(created.instance_id)

    assert fetched is not None
    assert fetched.name == "vision-a100"
    assert fetched.transport_config["host"] == "gpu-a100-01.internal"
    assert repo.get_default_instance_id() == created.instance_id


def test_repository_lists_created_instances(tmp_path):
    repo = SqliteVLLMInstanceRepository(db_path=tmp_path / "vllm_instances.db")
    first = repo.create_instance(_instance_payload("embed-box"))
    second = repo.create_instance(_instance_payload("vision-box"))

    records = repo.list_instances()

    assert [record.instance_id for record in records] == [first.instance_id, second.instance_id]
    assert [record.name for record in records] == ["embed-box", "vision-box"]
