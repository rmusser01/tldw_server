"""Private role adapter; official PostgreSQL fixtures still own every database."""

import json
import os
import re
import secrets
import sys
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.rows import dict_row


def connect(config, database="postgres"):
    """Use explicit private credentials without printing a DSN."""
    return psycopg.connect(
        host=config["host"],
        port=int(config["port"]),
        user=config["user"],
        password=config["password"],
        dbname=database,
        autocommit=True,
        row_factory=dict_row,
        connect_timeout=5,
    )


def write_private(path, value):
    """Never overwrite a credential/ownership record from another attempt."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as output:
        json.dump(value, output, indent=2)


@contextmanager
def runtime_server(provisioner, *, record_path=None):
    """Give official fixtures a unique role; drop only that role after teardown."""
    name = f"tldw_matrix_{uuid4().hex[:16]}"
    runtime = {**provisioner, "user": name, "password": secrets.token_urlsafe(32)}
    if record_path is not None:
        write_private(record_path, runtime)
    created = False
    try:
        with connect(provisioner) as connection:
            connection.execute(
                sql.SQL(
                    "CREATE ROLE {} LOGIN NOSUPERUSER NOBYPASSRLS NOINHERIT "
                    "NOCREATEROLE NOREPLICATION CREATEDB PASSWORD {}"
                ).format(sql.Identifier(name), sql.Literal(runtime["password"]))
            )
        created = True
        yield {**runtime, "_matrix_provisioner": provisioner}
    finally:
        if created:
            # Dependent official fixtures must finish database cleanup first.
            # A failure here remains visible; never DROP OWNED across the cluster.
            with connect(provisioner) as connection:
                connection.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(name)))


def require_runtime_role(row, expected):
    """Accept only a direct, unprivileged login that owns its fixture database."""
    if (
        not row
        or not re.fullmatch(r"tldw_matrix_[a-f0-9]{16}", expected)
        or row.get("name") != expected
        or row.get("session_user") != expected
        or row.get("database_owner") != expected
        or row.get("login") is not True
        or row.get("memberships") != 0
        or row.get("row_security") != "on"
        or any(
            row.get(key) is not False
            for key in ("superuser", "bypassrls", "inherit", "createdb", "createrole", "replication")
        )
    ):
        raise RuntimeError("Restricted PostgreSQL runtime role verification failed")
    return row


def inspect_runtime(database):
    """Check the actual direct login and database ownership using read-only SQL."""
    with connect(database, database["database"]) as connection:
        row = connection.execute("""
            SELECT current_user::text AS name, session_user::text AS session_user,
                r.rolcanlogin AS login, r.rolsuper AS superuser, r.rolbypassrls AS bypassrls,
                r.rolinherit AS inherit, r.rolcreatedb AS createdb, r.rolcreaterole AS createrole,
                r.rolreplication AS replication,
                (SELECT count(*) FROM pg_auth_members m WHERE m.member=r.oid) AS memberships,
                current_setting('row_security') AS row_security,
                (SELECT pg_get_userbyid(d.datdba) FROM pg_database d WHERE d.datname=current_database()) AS database_owner
            FROM pg_roles r WHERE r.rolname=current_user
        """).fetchone()
    return require_runtime_role(row, database["user"])


def qualify_runtime(server, auth, content):
    """Remove fixture-only CREATEDB before recording held runtime credentials."""
    for database in (auth, content):
        if any(
            database.get(key) != server.get(key) for key in ("host", "port", "user", "password")
        ) or not re.fullmatch(r"tldw_test_[a-f0-9]{8}", database.get("database", "")):
            raise RuntimeError("PostgreSQL runtime role target mismatch")
    with connect(server["_matrix_provisioner"]) as connection:
        connection.execute(sql.SQL("ALTER ROLE {} NOCREATEDB").format(sql.Identifier(server["user"])))
    if auth["database"] == content["database"]:
        raise RuntimeError("Distinct official PostgreSQL databases required")
    rows = [inspect_runtime(database) for database in (auth, content)]
    return {
        key: rows[0][key]
        for key in (
            "name",
            "login",
            "superuser",
            "bypassrls",
            "inherit",
            "createdb",
            "createrole",
            "replication",
            "memberships",
        )
    }


def check_files(config_path, receipt_path):
    """Recheck current runtime privileges before a launcher starts any process."""
    for path in (config_path, receipt_path):
        if Path(path).stat().st_mode & 0o777 != 0o600:
            raise RuntimeError("Private role input must be mode0600")
    config = json.loads(Path(config_path).read_text())
    receipt = json.loads(Path(receipt_path).read_text())
    if (
        config.get("purpose") != "matrix-runtime"
        or config.get("cell") != receipt.get("profile")
        or config.get("run_id") != receipt.get("run_id")
        or receipt.get("status") != "held"
        or config.get("host") not in ("127.0.0.1", "localhost", "::1")
        or not re.fullmatch(r"tldw_matrix_[a-f0-9]{16}", config.get("user", ""))
        or not receipt.get("provisioning", {}).get("user")
        or receipt.get("provisioning", {}).get("user") == config.get("user")
        or any(receipt.get("provisioning", {}).get(key) != config.get(key) for key in ("host", "port"))
        or receipt.get("runtime_role", {}).get("name") != config.get("user")
    ):
        raise RuntimeError("PostgreSQL runtime role ownership mismatch")
    databases = [receipt[key] for key in ("auth", "content")]
    if databases[0]["database"] == databases[1]["database"]:
        raise RuntimeError("Distinct official PostgreSQL databases required")
    for database in databases:
        if any(
            database.get(key) != config.get(key) for key in ("host", "port", "user", "password")
        ) or not re.fullmatch(r"tldw_test_[a-f0-9]{8}", database.get("database", "")):
            raise RuntimeError("PostgreSQL runtime role target mismatch")
        inspect_runtime(database)


if __name__ == "__main__":
    try:
        if len(sys.argv) != 3:
            raise RuntimeError("Expected private configuration and receipt paths")
        check_files(*sys.argv[1:])
    except (RuntimeError, OSError, ValueError, KeyError, TypeError, psycopg.Error):
        print("PostgreSQL runtime role verification failed; inspect private inputs.", file=sys.stderr)
        raise SystemExit(1) from None
