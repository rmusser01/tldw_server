import configparser

from tldw_Server_API.app.core.Security import secret_manager as sm


class _CapturingLogger:
    def __init__(self):
        self.records = []

    def warning(self, message, *args, **kwargs):
        self.records.append(("warning", message, args, dict(kwargs)))


def _joined_records(logger: _CapturingLogger) -> str:
    return "\n".join(f"{level} {message} {args!r} {kwargs!r}" for level, message, args, kwargs in logger.records)


def _empty_config():


    return configparser.ConfigParser()


def test_get_secret_override_does_not_mutate_config(monkeypatch):


    monkeypatch.setattr(sm, "load_comprehensive_config", _empty_config)

    manager = sm.SecretManager(validate_on_startup=False)
    manager._secret_configs = {
        "alpha": sm.SecretConfig(
            name="alpha",
            secret_type=sm.SecretType.API_KEY,
            env_var="ALPHA",
            required=True,
            default_value="default",
            min_length=1,
        )
    }
    monkeypatch.setenv("ALPHA", "abc123")

    manager.get_secret("alpha", required=False, default="override")
    manager.list_secrets()

    assert manager._secret_configs["alpha"].required is True
    assert manager._secret_configs["alpha"].default_value == "default"


def test_load_env_file_failure_log_is_sanitized(monkeypatch):
    logger = _CapturingLogger()

    class BrokenPath:
        def __init__(self, *_args, **_kwargs):
            pass

        def resolve(self):
            raise RuntimeError("env probe failed at /private/config/.env")

    manager = object.__new__(sm.SecretManager)

    monkeypatch.setattr(sm, "logger", logger)
    monkeypatch.setattr(sm, "Path", BrokenPath)

    manager._load_env_file()

    joined = _joined_records(logger)
    assert "SecretManager: Error checking .env file" in joined
    assert "env probe failed" not in joined
    assert "/private/config/.env" not in joined


def test_secret_health_check_error_issue_is_sanitized(monkeypatch):
    monkeypatch.setattr(sm, "load_comprehensive_config", _empty_config)
    manager = sm.SecretManager(validate_on_startup=False)
    manager._secret_configs = {
        "alpha": sm.SecretConfig(
            name="alpha",
            secret_type=sm.SecretType.API_KEY,
            required=True,
            min_length=1,
        )
    }

    def _raise_get_secret(*_args, **_kwargs):
        raise RuntimeError("secret backend failed at /private/secrets.db")

    monkeypatch.setattr(manager, "get_secret", _raise_get_secret)

    health = manager.get_production_health_check()

    assert health["status"] == "unhealthy"
    assert "Error checking secret 'alpha'" in health["issues"]
    assert all("secret backend failed" not in issue for issue in health["issues"])
    assert all("/private/secrets.db" not in issue for issue in health["issues"])


def test_secret_health_check_outer_failure_is_sanitized(monkeypatch):
    monkeypatch.setattr(sm, "load_comprehensive_config", _empty_config)
    manager = sm.SecretManager(validate_on_startup=False)

    class BrokenSecretConfigs:
        def __len__(self):
            return 1

        def items(self):
            raise RuntimeError("secret config iteration failed at /private/secrets.db")

    manager._secret_configs = BrokenSecretConfigs()

    health = manager.get_production_health_check()

    assert health["status"] == "error"
    assert "Health check failed" in health["issues"]
    assert all("secret config iteration failed" not in issue for issue in health["issues"])
    assert all("/private/secrets.db" not in issue for issue in health["issues"])
