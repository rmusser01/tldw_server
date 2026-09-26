"""Recorded synthetic archive validation probe; see Email_Archive_Ingestion_Throughput_2026-09-25.md."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import mailbox
import os
import platform
import socket
import tempfile
import time
from email.message import EmailMessage
from pathlib import Path
from secrets import token_urlsafe
from urllib.parse import quote

ROOT = Path(tempfile.mkdtemp(prefix="tldw-email-live-postgres-")).resolve()
PG = json.loads(Path(os.environ["EMAIL_PROBE_PG_MANIFEST"]).read_text(encoding="utf-8"))
PG_AUTH_DSN = f"postgresql://{PG['role']}:{quote(PG['password'])}@{PG['host']}:{PG['port']}/{PG['auth_db']}"
PG_CONTENT_DSN = f"postgresql://{PG['role']}:{quote(PG['password'])}@{PG['host']}:{PG['port']}/{PG['content_db']}"
for key in ("TEST_MODE", "TESTING", "TLDW_TEST_MODE", "PYTEST_CURRENT_TEST", "MINIMAL_TEST_APP", "ULTRA_MINIMAL_APP"):
    os.environ.pop(key, None)
os.environ.update(
    {
        "AUTH_MODE": "multi_user",
        "PROFILE": "multi-user-postgres",
        "DATABASE_URL": PG_AUTH_DSN,
        "USER_DB_BASE_DIR": str(ROOT / "users"),
        "CONTENT_DB_MODE": "postgresql",
        "TLDW_CONTENT_DB_BACKEND": "postgresql",
        "TLDW_CONTENT_PG_DSN": PG_CONTENT_DSN,
        "JWT_SECRET_KEY": token_urlsafe(48),
        "REDIS_URL": "",
        "EMAIL_NATIVE_PERSIST_ENABLED": "true",
        "EMAIL_OPERATOR_SEARCH_ENABLED": "true",
        "EMAIL_MEDIA_SEARCH_DELEGATION_MODE": "opt_in",
        "EMAIL_GMAIL_CONNECTOR_ENABLED": "false",
        "CONNECTORS_WORKER_ENABLED": "false",
        "TLDW_WORKERS_SIDECAR_MODE": "true",
        "DEFER_HEAVY_STARTUP": "true",
        "STORAGE_QUOTA_ENFORCEMENT": "1",
        "STORAGE_QUOTA_FAIL_OPEN": "0",
        "EVALS_HEAVY_ADMIN_ONLY": "true",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "LOG_LEVEL": "WARNING",
        "SCHEDULER_DATABASE_URL": f"sqlite:///{ROOT / 'scheduler.sqlite'}",
    }
)

outbound_attempts: list[str] = []
model_attempts: list[str] = []
original_connect = socket.socket.connect
original_connect_ex = socket.socket.connect_ex
original_getaddrinfo = socket.getaddrinfo


def _loopback_host(host: object) -> bool:
    if host in (None, "localhost"):
        return True
    try:
        return ipaddress.ip_address(str(host)).is_loopback
    except ValueError:
        return False


def guarded_connect(sock: socket.socket, address: object) -> object:
    if sock.family == socket.AF_UNIX:
        return original_connect(sock, address)
    host = address[0] if isinstance(address, tuple) else address
    if not _loopback_host(host):
        outbound_attempts.append(f"connect:{host}")
        raise OSError("external socket blocked by synthetic probe")
    return original_connect(sock, address)


def guarded_connect_ex(sock: socket.socket, address: object) -> int:
    if sock.family == socket.AF_UNIX:
        return original_connect_ex(sock, address)
    host = address[0] if isinstance(address, tuple) else address
    if not _loopback_host(host):
        outbound_attempts.append(f"connect_ex:{host}")
        raise OSError("external socket blocked by synthetic probe")
    return original_connect_ex(sock, address)


def guarded_getaddrinfo(host: object, *args: object, **kwargs: object) -> object:
    if not _loopback_host(host):
        outbound_attempts.append(f"getaddrinfo:{host}")
        raise OSError("external DNS blocked by synthetic probe")
    return original_getaddrinfo(host, *args, **kwargs)


socket.socket.connect = guarded_connect
socket.socket.connect_ex = guarded_connect_ex
socket.getaddrinfo = guarded_getaddrinfo

from fastapi import BackgroundTasks

from tldw_Server_API.app.core.Chunking.auto_boundary_assistant import ChatAutoChunkBoundaryAssistant
from tldw_Server_API.app.core.Claims_Extraction import claims_utils
from tldw_Server_API.app.core.Embeddings.jobs_adapter import EmbeddingsJobsAdapter
from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib


def forbidden_model(*args: object, **kwargs: object) -> None:
    model_attempts.append("model-or-background-task")
    raise AssertionError("model/background work blocked by synthetic probe")


for target, name in (
    (Summarization_General_Lib, "analyze"),
    (EmbeddingsJobsAdapter, "create_job"),
    (claims_utils, "extract_claims_for_chunks"),
    (ChatAutoChunkBoundaryAssistant, "refine"),
    (BackgroundTasks, "add_task"),
):
    setattr(target, name, forbidden_model)

import httpx
import uvicorn

from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
from tldw_Server_API.app.core.AuthNZ.repos.storage_quotas_repo import AuthnzStorageQuotasRepo
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo


async def _user(repo: AuthnzUsersRepo, manager: object, name: str) -> dict[str, object]:
    user_id = await repo.create_user(
        username=name,
        email=f"{name}@example.test",
        password_hash=token_urlsafe(32),
        is_verified=True,
    )
    await repo.assign_role_if_missing(user_id=user_id, role_name="user")
    read_key = await manager.create_api_key(user_id=user_id, name="synthetic-read", scope="read")
    write_key = await manager.create_api_key(user_id=user_id, name="synthetic-write", scope="write")
    return {"id": user_id, "read_key": read_key["key"], "write_key": write_key["key"]}


def _archive(batch: int) -> bytes:
    path = ROOT / f"synthetic-{batch}.mbox"
    box = mailbox.mbox(path)
    try:
        for i in range(100):
            message = EmailMessage()
            message["From"] = "sender@example.test"
            message["To"] = "recipient@example.test"
            message["Subject"] = f"ArchiveThroughput {batch}-{i:03d}"
            message["Message-ID"] = f"<archive-throughput-{batch}-{i}@example.test>"
            message["Date"] = "Tue, 10 Feb 2026 09:30:00 -0500"
            message.set_content(f"Unique synthetic archive body batch {batch} message {i}.")
            box.add(message)
        box.flush()
    finally:
        box.close()
    return path.read_bytes()


def _port() -> int:
    with socket.socket() as bound:
        bound.bind(("127.0.0.1", 0))
        return int(bound.getsockname()[1])


async def main() -> None:
    pool = await get_db_pool()
    repo = AuthnzUsersRepo(pool)
    manager = await get_api_key_manager()
    alice = await _user(repo, manager, "live_alice")
    bob = await _user(repo, manager, "live_bob")
    org_repo = AuthnzOrgsTeamsRepo(pool)
    quota_repo = AuthnzStorageQuotasRepo(pool)
    alice_org = await org_repo.create_organization(
        name="Synthetic SQLite Alice", owner_user_id=int(alice["id"]), slug="synthetic-sqlite-alice"
    )
    bob_org = await org_repo.create_organization(
        name="Synthetic SQLite Bob", owner_user_id=int(bob["id"]), slug="synthetic-sqlite-bob"
    )
    await org_repo.add_org_member(org_id=alice_org["id"], user_id=int(alice["id"]), role="owner")
    await org_repo.add_org_member(org_id=bob_org["id"], user_id=int(bob["id"]), role="owner")
    await quota_repo.upsert_org_quota(alice_org["id"], quota_mb=1024)

    port = _port()
    server = uvicorn.Server(
        uvicorn.Config(
            "tldw_Server_API.app.main:app",
            host="127.0.0.1",
            port=port,
            lifespan="on",
            log_level="warning",
            access_log=False,
            timeout_graceful_shutdown=15,
        )
    )
    server_task = asyncio.create_task(server.serve())
    try:
        for _ in range(1200):
            if server.started:
                break
            if server_task.done():
                await server_task
                raise AssertionError("server exited before startup")
            await asyncio.sleep(0.1)
        else:
            raise TimeoutError("server did not start within 120 seconds")

        # PostgreSQL role rows are seeded during main-app startup.
        await repo.assign_role_if_missing(user_id=int(alice["id"]), role_name="user")
        await repo.assign_role_if_missing(user_id=int(bob["id"]), role_name="user")

        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=180.0) as client:
            health = await client.get("/health")
            ready = await client.get("/internal/ready")
            assert health.status_code == 200, (health.status_code, health.text)
            assert ready.status_code == 200, (ready.status_code, ready.text)

            write_headers = {"X-API-KEY": str(alice["write_key"]), "X-TLDW-Org-Id": str(alice_org["id"])}
            read_headers = {"X-API-KEY": str(alice["read_key"]), "X-TLDW-Org-Id": str(alice_org["id"])}
            bob_headers = {"X-API-KEY": str(bob["read_key"]), "X-TLDW-Org-Id": str(bob_org["id"])}
            unauth = await client.get("/api/v1/email/search")
            assert unauth.status_code == 401, (unauth.status_code, unauth.text)
            runs = []
            all_ids = set()
            archives = [_archive(batch) for batch in range(3)]
            for batch, payload in enumerate(archives):
                start = time.perf_counter()
                upload = await client.post(
                    "/api/v1/media/add",
                    headers=write_headers,
                    files={"files": (f"synthetic-{batch}.mbox", payload, "application/mbox")},
                    data={
                        "media_type": "email",
                        "accept_mbox": "true",
                        "perform_analysis": "false",
                        "perform_claims_extraction": "false",
                        "perform_chunking": "false",
                        "auto_chunking_use_llm": "false",
                        "generate_embeddings": "false",
                        "keep_original_file": "false",
                    },
                )
                elapsed = time.perf_counter() - start
                assert upload.status_code == 200, (upload.status_code, upload.text[:1000])
                result = upload.json()["results"][0]
                assert result["status"] == "Success", result
                children = result.get("child_db_results") or []
                assert len(children) == 100, f"Expected 100 persisted children, found {len(children)}"
                ids = {row["db_id"] for row in children}
                assert len(ids) == 100 and not (ids & all_ids)
                all_ids.update(ids)
                runs.append(
                    {
                        "batch": batch,
                        "messages": 100,
                        "archive_bytes": len(payload),
                        "elapsed_seconds": elapsed,
                        "messages_per_second": 100 / elapsed,
                    }
                )
                print("ARCHIVE_RUN " + json.dumps(runs[-1]), flush=True)
            search = await client.get(
                "/api/v1/email/search", params={"q": "subject:ArchiveThroughput", "limit": 500}, headers=read_headers
            )
            assert search.status_code == 200, search.text
            assert len(search.json()["items"]) == 300, search.text[:1000]
            found_ids = {row["media_id"] for row in search.json()["items"]}
            assert found_ids == all_ids
            for media_id in (min(all_ids), max(all_ids)):
                detail = await client.get(f"/api/v1/email/messages/{media_id}", headers=read_headers)
                assert detail.status_code == 200, detail.text
                assert detail.json()["subject"].startswith("ArchiveThroughput")
            retry = await client.post(
                "/api/v1/media/add",
                headers=write_headers,
                files={"files": ("synthetic-0.mbox", archives[0], "application/mbox")},
                data={
                    "media_type": "email",
                    "accept_mbox": "true",
                    "perform_analysis": "false",
                    "perform_claims_extraction": "false",
                    "perform_chunking": "false",
                    "auto_chunking_use_llm": "false",
                    "generate_embeddings": "false",
                    "keep_original_file": "false",
                },
            )
            assert retry.status_code == 200, retry.text[:1000]
            retry_children = retry.json()["results"][0].get("child_db_results") or []
            assert {row["db_id"] for row in retry_children} <= all_ids and len(retry_children) == 100
            after = await client.get(
                "/api/v1/email/search", params={"q": "subject:ArchiveThroughput", "limit": 500}, headers=read_headers
            )
            assert {row["media_id"] for row in after.json()["items"]} == all_ids
            bob_search = await client.get("/api/v1/email/search", headers=bob_headers)
            assert bob_search.status_code == 200 and bob_search.json()["items"] == []
            bob_detail = await client.get(f"/api/v1/email/messages/{min(all_ids)}", headers=bob_headers)
            assert bob_detail.status_code == 404
            assert not outbound_attempts and not model_attempts
            summary = {
                "backend": "postgresql",
                "full_app": True,
                "auth_mode": "multi_user",
                "test_mode": False,
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "python": platform.python_version(),
                "runs": runs,
                "messages": 300,
                "attachment_ratio": 0,
                "rerun_messages": 100,
                "idempotent": True,
                "cross_user_search_count": 0,
                "cross_user_detail_status": 404,
                "outbound_attempts": len(outbound_attempts),
                "model_attempts": len(model_attempts),
                "aggregate_messages_per_second": 300 / sum(row["elapsed_seconds"] for row in runs),
                "target_messages_per_second": 50,
                "root": str(ROOT),
            }
            import asyncpg

            connection = await asyncpg.connect(PG_CONTENT_DSN)
            try:
                role = await connection.fetchrow(
                    "SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user"
                )
                rls = await connection.fetchrow(
                    "SELECT relrowsecurity,relforcerowsecurity FROM pg_class WHERE oid='media'::regclass"
                )
                assert not role["rolsuper"] and not role["rolbypassrls"]
                assert rls["relrowsecurity"] and rls["relforcerowsecurity"]
                await connection.execute("SELECT set_config('app.current_user_id',$1,false)", str(alice["id"]))
                owner_count = await connection.fetchval("SELECT COUNT(*) FROM media")
                await connection.execute("SELECT set_config('app.current_user_id',$1,false)", str(bob["id"]))
                other_count = await connection.fetchval("SELECT COUNT(*) FROM media")
                assert owner_count == 300 and other_count == 0
                summary.update(
                    {
                        "rls_enabled": True,
                        "rls_forced": True,
                        "superuser": False,
                        "bypass_rls": False,
                        "owner_media_rows": owner_count,
                        "other_media_rows": other_count,
                    }
                )
            finally:
                await connection.close()
            summary["all_batches_meet_target"] = all(row["messages_per_second"] >= 50 for row in runs)
            Path(os.environ.get("EMAIL_PROBE_OUT", str(ROOT / "throughput.json"))).write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            print("THROUGHPUT_RESULT " + json.dumps(summary), flush=True)

        assert not outbound_attempts, outbound_attempts
        assert not model_attempts, model_attempts

    finally:
        server.should_exit = True
        await asyncio.wait_for(server_task, timeout=30)


if __name__ == "__main__":
    asyncio.run(main())
