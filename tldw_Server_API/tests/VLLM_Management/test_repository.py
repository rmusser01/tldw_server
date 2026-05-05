import ast
from pathlib import Path

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


def test_vllm_management_core_does_not_own_sqlite_driver_imports():
    vllm_core_path = Path(__file__).resolve().parents[2] / "app/core/VLLM_Management"
    offenders: list[str] = []

    for module_path in sorted(vllm_core_path.rglob("*.py")):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(alias.name == "sqlite3" for alias in node.names):
                offenders.append(str(module_path.relative_to(vllm_core_path)))
            if isinstance(node, ast.ImportFrom) and node.module == "sqlite3":
                offenders.append(str(module_path.relative_to(vllm_core_path)))

    assert offenders == []


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


def test_repository_updates_instance_spec_and_runtime_metadata(tmp_path):
    repo = SqliteVLLMInstanceRepository(db_path=tmp_path / "vllm_instances.db")
    created = repo.create_instance(_instance_payload("embed-box"))

    updated = repo.update_instance(
        created.instance_id,
        {
            "name": "embed-box-v2",
            "routing_policy": {"is_default": True},
            "declared_capabilities": {"chat": False, "embeddings": True},
        },
    )
    runtime = repo.update_instance_runtime(
        created.instance_id,
        {
            "desired_state": "running",
            "observed_state": "healthy",
            "probed_capabilities": {"embeddings": True},
            "effective_capabilities": {"embeddings": True},
            "last_known_base_url": "http://127.0.0.1:8010/v1",
            "last_error": None,
            "executor_handle": {"pid": 4242},
        },
    )

    assert updated.name == "embed-box-v2"
    assert updated.routing_policy["is_default"] is True
    assert runtime.observed_state == "healthy"
    assert runtime.probed_capabilities == {"embeddings": True}
    assert runtime.effective_capabilities == {"embeddings": True}
    assert runtime.last_known_base_url == "http://127.0.0.1:8010/v1"
    assert runtime.executor_handle == {"pid": 4242}


def test_repository_delete_instance_clears_default_route(tmp_path):
    repo = SqliteVLLMInstanceRepository(db_path=tmp_path / "vllm_instances.db")
    created = repo.create_instance(_instance_payload("delete-me"))
    repo.set_default_instance(created.instance_id)

    deleted = repo.delete_instance(created.instance_id)

    assert deleted is True
    assert repo.get_instance(created.instance_id) is None
    assert repo.get_default_instance_id() is None
