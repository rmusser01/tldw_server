from pathlib import Path

import yaml


EXPECTED_REDIS_URL = "redis://127.0.0.1:6379/0"
EXPECTED_REDIS_IMAGE = (
    "mirror.gcr.io/library/redis:8-alpine@"
    "sha256:9eb6a7ba3d344e1958c7e1589fa3dee90373a934e8159c634562a91d622759a0"
)


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
        redis_service.get("image") == EXPECTED_REDIS_IMAGE,
        "e2e-required redis image must use the pinned mirror.gcr.io Redis 8-alpine digest",
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
