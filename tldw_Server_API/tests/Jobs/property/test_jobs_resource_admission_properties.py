"""Generated admission sequences keep one immutable creation per owner resource."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.tests.Jobs.test_jobs_resource_admission_sqlite import make_command, make_request, snapshot


@settings(max_examples=30, deadline=None)
@given(
    st.lists(st.tuples(st.integers(1, 1000), st.booleans(), st.sampled_from([None, "a", "b"])), min_size=1, max_size=20)
)
def test_generated_alias_queue_and_settings_sequences(operations):
    with TemporaryDirectory() as directory:
        manager = JobManager(Path(directory) / "jobs.db")
        active_min_size = operations[0][0]
        aliases = {}
        job_uuid = None
        for min_size, other_queue, key in operations:
            command = make_command(payload={**make_command().payload, "min_size": min_size})
            # Queue is immaterial even when current policy forbids the replay queue.
            if other_queue and job_uuid is not None:
                command = replace(command, queue="unconfigured")
            result = manager.admit_resource_job(command, request=make_request(command, key=key))
            if key is not None and key in aliases:
                expected = "request_replayed" if aliases[key] == min_size else "request_conflict"
            elif job_uuid is None:
                expected = "created"
            else:
                expected = "active_reused" if min_size == active_min_size else "active_conflict"
            assert result.kind == expected
            if result.job is not None:
                job_uuid = job_uuid or result.job["uuid"]
                assert result.job["uuid"] == job_uuid
                if key is not None:
                    aliases[key] = min_size
        facts = snapshot(manager)
        assert len(facts["jobs"]) == len(facts["job_events"]) == 1
        assert len(facts["job_resource_request_receipts"]) == len(aliases)
