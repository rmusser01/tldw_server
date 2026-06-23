from pathlib import Path

import yaml


EXPECTED_REDIS_URL = "redis://127.0.0.1:6379/0"


def _load_workflow() -> dict:
    return yaml.safe_load(Path(".github/workflows/e2e-required.yml").read_text(encoding="utf-8"))


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_e2e_required_declares_redis_service_and_environment_contract() -> None:
    workflow = _load_workflow()
    job = workflow["jobs"]["e2e-required"]

    services = job.get("services") or {}
    redis_service = services.get("redis")
    _expect(isinstance(redis_service, dict), "e2e-required.redis service missing")
    _expect(
        redis_service.get("image") == "mirror.gcr.io/library/redis:8-alpine",
        "e2e-required redis image must be mirror.gcr.io/library/redis:8-alpine (mirror avoids Docker Hub rate limits; matches ci.yml)",
    )
    _expect(
        "6379:6379" in (redis_service.get("ports") or []),
        "e2e-required redis service must map host/container port 6379:6379",
    )
    _expect("redis-cli ping" in str(redis_service.get("options", "")), "e2e-required redis service must define a ping health check")

    env = job.get("env") or {}
    _expect(env.get("REDIS_URL") == EXPECTED_REDIS_URL, "e2e-required REDIS_URL must bind to the redis service port")
    _expect(
        env.get("EMBEDDINGS_REDIS_URL") == EXPECTED_REDIS_URL,
        "e2e-required EMBEDDINGS_REDIS_URL must bind to the redis service port",
    )
