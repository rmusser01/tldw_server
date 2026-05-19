from tldw_Server_API.app.core.Evaluations.webhook_manager import WebhookManager


class _RecordingAdapter:
    backend_type = "postgresql"

    def __init__(self):
        self.schemas = []

    def init_schema(self, schema_sql: str):
        self.schemas.append(schema_sql)


def test_webhook_manager_uses_postgres_compatible_schema_for_postgres_adapter():
    adapter = _RecordingAdapter()
    WebhookManager(adapter=adapter)

    assert adapter.schemas
    schema_sql = adapter.schemas[-1]
    assert "BIGSERIAL" in schema_sql
    assert "AUTOINCREMENT" not in schema_sql
    assert "DEFAULT TRUE" in schema_sql
