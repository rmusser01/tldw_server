"""Provision/clean isolated synthetic PostgreSQL databases on the local test service."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess  # nosec B404 - local Docker inspection, shell=False
import sys
from pathlib import Path
from secrets import token_hex

import asyncpg
from email_million_search_13376 import _configured_postgres_port

MANIFEST = Path(os.environ["EMAIL_PROBE_PG_MANIFEST"])
HOST = "127.0.0.1"
PORT = _configured_postgres_port()
CONTAINER = os.environ["EMAIL_PROBE_PG_CONTAINER"]


def validate_manifest(manifest: dict) -> None:
    """Restrict cleanup to one generated probe on the local fixture service."""
    match = re.fullmatch(r"email_probe_([0-9a-f]{10})", str(manifest.get("role", "")))
    if match is None:
        raise ValueError("Manifest is not a generated email probe role")
    suffix = match.group(1)
    if (
        manifest.get("host") != HOST
        or type(manifest.get("port")) is not int
        or manifest.get("port") != PORT
        or manifest.get("auth_db") != f"email_auth_{suffix}"
        or manifest.get("content_db") != f"email_content_{suffix}"
    ):
        raise ValueError("Manifest targets do not match the isolated local probe")


async def admin_connection() -> asyncpg.Connection:
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("Docker CLI is unavailable")
    environment = json.loads(
        subprocess.check_output(  # nosec B603 - fixed inspection arguments, no shell
            [docker, "inspect", "--format", "{{json .Config.Env}}", CONTAINER], text=True
        )
    )
    values = dict(item.split("=", 1) for item in environment if "=" in item)
    user = values["POSTGRES_USER"]
    password = values["POSTGRES_PASSWORD"]
    return await asyncpg.connect(host=HOST, port=PORT, user=user, password=password, database="postgres")


async def setup() -> None:
    if MANIFEST.exists():
        raise RuntimeError("probe manifest already exists")
    suffix = token_hex(5)
    role = f"email_probe_{suffix}"
    auth_db = f"email_auth_{suffix}"
    content_db = f"email_content_{suffix}"
    password = token_hex(24)
    connection = await admin_connection()
    try:
        await connection.execute(
            f"CREATE ROLE \"{role}\" LOGIN PASSWORD '{password}' "
            "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
        )
        await connection.execute(f'CREATE DATABASE "{auth_db}" OWNER "{role}"')
        await connection.execute(f'CREATE DATABASE "{content_db}" OWNER "{role}"')
    finally:
        await connection.close()
    manifest = {
        "host": HOST,
        "port": PORT,
        "role": role,
        "password": password,
        "auth_db": auth_db,
        "content_db": content_db,
    }
    descriptor = os.open(MANIFEST, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(manifest, output)
    print(json.dumps({key: value for key, value in manifest.items() if key != "password"}))


async def cleanup() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    connection = await admin_connection()
    try:
        await connection.execute(f'DROP DATABASE IF EXISTS "{manifest["auth_db"]}" WITH (FORCE)')
        await connection.execute(f'DROP DATABASE IF EXISTS "{manifest["content_db"]}" WITH (FORCE)')
        await connection.execute(f'DROP ROLE IF EXISTS "{manifest["role"]}"')
        remaining = await connection.fetchval(
            "SELECT (SELECT COUNT(*) FROM pg_database WHERE datname = ANY($1::text[])) "
            "+ (SELECT COUNT(*) FROM pg_roles WHERE rolname = $2)",
            [manifest["auth_db"], manifest["content_db"]],
            manifest["role"],
        )
        if remaining:
            raise RuntimeError("Disposable resources still exist after cleanup")
    finally:
        await connection.close()
    MANIFEST.unlink()
    print("Cleaned synthetic PostgreSQL probe databases and role")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"setup", "cleanup"}:
        raise SystemExit("Usage: probe_databases.py setup|cleanup")
    asyncio.run(setup() if sys.argv[1] == "setup" else cleanup())
