import base64
import json
import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager


class FakePGCursor:
    def __init__(self, jobs):
             # jobs: dict id -> dict(row)
        self.jobs = jobs
        self._last = None
        self.rowcount = 0
        self._fetch_buffer = None
        self.job_event_inserts = []

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc, tb):

        return False

    def execute(self, sql, params=None):

        s = str(sql)
        self.rowcount = 0
        self._fetch_buffer = None
        # Idempotent insert returning * (first insert returns a row, later returns None)
        if "ON CONFLICT (domain, queue, job_type, idempotency_key) DO NOTHING RETURNING *" in s:
            # params: (uuid, domain, queue, job_type, owner, project, batch_group, idem_key, payload_json, priority, max_retries, available_at, request_id, trace_id)
            domain = params[1]
            queue = params[2]
            job_type = params[3]
            # batch_group was added ahead of idempotency_key; keep indices aligned with manager.py
            idem = params[7]
            # build a key
            key = (domain, queue, job_type, idem)
            # If not present, insert and return row
            for j in self.jobs.values():
                if (j.get("domain"), j.get("queue"), j.get("job_type"), j.get("idempotency_key")) == key:
                    # already exists: return None
                    self._fetch_buffer = None
                    return
            new_id = max(self.jobs.keys() or [0]) + 1
            row = {
                "id": new_id,
                "domain": domain,
                "queue": queue,
                "job_type": job_type,
                "idempotency_key": idem,
                "status": "queued",
                "priority": int(params[9]),
                "available_at": params[11],
            }
            self.jobs[new_id] = row
            self._fetch_buffer = row
            self.rowcount = 1
            return
        # Lookup by idempotency
        if "SELECT * FROM jobs WHERE domain = %s AND queue = %s AND job_type = %s AND idempotency_key = %s" in s:
            dom, que, jt, idem = params
            for j in self.jobs.values():
                if (j.get("domain"), j.get("queue"), j.get("job_type"), j.get("idempotency_key")) == (dom, que, jt, idem):
                    self._fetch_buffer = j
                    return
            self._fetch_buffer = None
            return
        if "INSERT INTO job_events(" in s:
            self.job_event_inserts.append(tuple(params or ()))
            return
        # Count(*) aliases c
        if s.startswith("SELECT COUNT(*) AS c FROM jobs"):
            self._fetch_buffer = {"c": 0}
            return
        # Read domain by id
        if "SELECT domain FROM jobs WHERE id=%s" in s:
            jid = int(params[0])
            row = self.jobs.get(jid)
            self._fetch_buffer = {"domain": row.get("domain")} if row else None
            return
        # Update completed with result jsonb
        if s.startswith("UPDATE jobs SET status='completed', result="):
            # job_id is always the third argument in the psycopg parameter sequence.
            job_id = int(params[2])
            res_json = params[0]
            try:
                obj = json.loads(res_json) if isinstance(res_json, str) else None
            except Exception:
                obj = None
            # store back into fake jobs for assertion convenience
            if job_id in self.jobs:
                self.jobs[job_id]["result_json"] = obj
            self.rowcount = 1
            self._fetch_buffer = None
            return
        # RETURNING * path for non-idempotent create (not used in these tests)
        if s.endswith("RETURNING *") and s.startswith("INSERT INTO jobs"):
            new_id = max(self.jobs.keys() or [0]) + 1
            row = {"id": new_id, "status": "queued", "domain": params[1], "queue": params[2], "job_type": params[3], "priority": int(params[9])}
            self.jobs[new_id] = row
            self._fetch_buffer = row
            self.rowcount = 1
            return

        # default: nothing to fetch
        self._fetch_buffer = None

    def fetchone(self):

        val = self._fetch_buffer
        self._fetch_buffer = None
        return val

    def fetchall(self):

        return []


class FailingJobEventsPGCursor(FakePGCursor):
    def execute(self, sql, params=None):
        s = str(sql)
        if "INSERT INTO job_events(" in s:
            raise RuntimeError("forced job_events insert failure")
        return super().execute(sql, params)


class FakePGConn:
    def __init__(self):
        pass

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc, tb):

        return False

    def close(self):

        pass


@pytest.fixture(autouse=True)
def _stub_pg_bootstrap(monkeypatch):
    import tldw_Server_API.app.core.Jobs.manager as mgr

    monkeypatch.setattr(mgr, "ensure_jobs_tables_pg", lambda dsn: dsn)
    monkeypatch.setattr(mgr, "ensure_job_counters_pg", lambda dsn: dsn, raising=False)


@pytest.mark.unit
def test_pg_create_job_idempotent_gates_created_metric(monkeypatch, tmp_path):
     # Capture increment_created calls
    calls = {"n": 0}

    def _inc(labels):

        calls["n"] += 1

    monkeypatch.setenv("JOBS_DB_URL", "postgresql://fake")
    jm = JobManager(db_path=tmp_path / "dummy.db")
    jm.backend = "postgres"

    # Patch manager-level symbol (it imports increment_created into module scope)
    import tldw_Server_API.app.core.Jobs.manager as mgr

    monkeypatch.setattr(mgr, "increment_created", _inc)

    jobs = {}
    monkeypatch.setattr(jm, "_connect", lambda: FakePGConn())
    monkeypatch.setattr(jm, "_pg_cursor", lambda conn: FakePGCursor(jobs))

    # First call should insert and increment created
    d1 = jm.create_job(domain="pg", queue="default", job_type="x", payload={}, owner_user_id=None, idempotency_key="K")
    assert d1 and d1.get("status") == "queued"
    assert calls["n"] == 1

    # Second call (same idem key) should not increment created
    d2 = jm.create_job(domain="pg", queue="default", job_type="x", payload={}, owner_user_id=None, idempotency_key="K")
    assert d2 and d2.get("status") == "queued"
    assert calls["n"] == 1


@pytest.mark.unit
def test_pg_create_job_logs_created_metric_failure_without_aborting(monkeypatch, tmp_path):
    warnings: list[tuple] = []

    class _LoggerStub:
        def warning(self, *args, **kwargs):
            warnings.append((args, kwargs))

        def info(self, *args, **kwargs):
            return None

        def debug(self, *args, **kwargs):
            return None

        def error(self, *args, **kwargs):
            return None

    def _inc(_labels):
        raise RuntimeError("metric backend offline")

    monkeypatch.setenv("JOBS_DB_URL", "postgresql://fake")
    jm = JobManager(db_path=tmp_path / "dummy.db")
    jm.backend = "postgres"

    import tldw_Server_API.app.core.Jobs.manager as mgr

    monkeypatch.setattr(mgr, "increment_created", _inc)
    monkeypatch.setattr(mgr, "logger", _LoggerStub())

    jobs = {}
    monkeypatch.setattr(jm, "_connect", lambda: FakePGConn())
    monkeypatch.setattr(jm, "_pg_cursor", lambda conn: FakePGCursor(jobs))

    row = jm.create_job(
        domain="pg",
        queue="default",
        job_type="x",
        payload={},
        owner_user_id=None,
        idempotency_key="K",
    )

    assert row and row.get("status") == "queued"
    assert len(warnings) == 1
    assert warnings[0][0][0] == "Non-critical jobs created metric update failed for {}:{}:{}: {}"
    assert warnings[0][0][1:4] == ("pg", "default", "x")
    assert str(warnings[0][0][4]) == "metric backend offline"


@pytest.mark.unit
def test_pg_idempotent_create_uses_current_request_context_for_job_created_event(monkeypatch, tmp_path):
    monkeypatch.setenv("JOBS_DB_URL", "postgresql://fake")
    jm = JobManager(db_path=tmp_path / "dummy.db")
    jm.backend = "postgres"

    jobs = {
        1: {
            "id": 1,
            "domain": "pg",
            "queue": "default",
            "job_type": "x",
            "idempotency_key": "K",
            "status": "queued",
            "priority": 5,
            "available_at": None,
            "owner_user_id": "owner-1",
            "request_id": "old-request",
            "trace_id": "old-trace",
            "retry_count": 0,
        }
    }
    cursor = FakePGCursor(jobs)
    monkeypatch.setattr(jm, "_connect", lambda: FakePGConn())
    monkeypatch.setattr(jm, "_pg_cursor", lambda conn: cursor)

    row = jm.create_job(
        domain="pg",
        queue="default",
        job_type="x",
        payload={},
        owner_user_id=None,
        idempotency_key="K",
        request_id="new-request",
        trace_id="new-trace",
    )

    assert row["id"] == 1
    assert cursor.job_event_inserts
    event_params = cursor.job_event_inserts[-1]
    assert event_params[7] == "new-request"
    assert event_params[8] == "new-trace"


@pytest.mark.unit
def test_pg_batch_complete_encrypts_results(monkeypatch, tmp_path):
     # Enable encryption for domain SECURE and provide AES key
    monkeypatch.setenv("JOBS_ENCRYPT_SECURE", "true")
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_ENC_KEY", base64.b64encode(b"1" * 32).decode("ascii"))

    jm = JobManager(db_path=tmp_path / "dummy.db")
    jm.backend = "postgres"
    # Fake jobs store: two processing jobs in domain 'secure'
    jobs = {
        1: {"id": 1, "domain": "secure", "status": "processing"},
        2: {"id": 2, "domain": "secure", "status": "processing"},
    }
    monkeypatch.setattr(jm, "_connect", lambda: FakePGConn())
    monkeypatch.setattr(jm, "_pg_cursor", lambda conn: FakePGCursor(jobs))

    items = [
        {"job_id": 1, "result": {"ok": True}, "domain": "secure"},
        {"job_id": 2, "result": {"ok": True}},  # no domain provided; code should SELECT domain
    ]

    n = jm.batch_complete_jobs(items)
    assert n == 2

    # Fake cursor captured stored result JSON per job
    r1 = jobs[1].get("result_json")
    r2 = jobs[2].get("result_json")
    assert isinstance(r1, dict) and "_encrypted" in r1
    assert isinstance(r2, dict) and "_encrypted" in r2


@pytest.mark.unit
@pytest.mark.parametrize("idempotency_key", [None, "K"])
def test_pg_create_failure_on_job_events_insert_does_not_increment_created(monkeypatch, tmp_path, idempotency_key):
    calls = {"n": 0}

    def _inc(_labels):
        calls["n"] += 1

    monkeypatch.setenv("JOBS_DB_URL", "postgresql://fake")
    jm = JobManager(db_path=tmp_path / "dummy.db")
    jm.backend = "postgres"

    import tldw_Server_API.app.core.Jobs.manager as mgr

    monkeypatch.setattr(mgr, "increment_created", _inc)

    jobs = {}
    monkeypatch.setattr(jm, "_connect", lambda: FakePGConn())
    monkeypatch.setattr(jm, "_pg_cursor", lambda conn: FailingJobEventsPGCursor(jobs))

    with pytest.raises(RuntimeError, match="job_events insert failure"):
        jm.create_job(
            domain="pg",
            queue="default",
            job_type="x",
            payload={},
            owner_user_id=None,
            idempotency_key=idempotency_key,
        )

    if calls["n"] != 0:
        raise AssertionError("increment_created must not be called when create fails at job_events insert")
