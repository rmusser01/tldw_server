#!/usr/bin/env sh
set -eu

ENV_FILE="${TLDW_ENV_FILE:-/app/tldw_Server_API/Config_Files/.env}"
AUTH_MARKER_DIR="${TLDW_AUTH_MARKER_DIR:-/app/Databases}"
# NOTE: AUTH_MARKER_FILE is derived below *after* AUTH_MODE is resolved,
# so that the marker is mode-specific (e.g. .authnz_initialized_single_user
# vs .authnz_initialized_multi_user). Changing AUTH_MODE will re-trigger init.
RUN_AUTH_INIT_ON_START="${TLDW_RUN_AUTH_INIT_ON_START:-1}"

incoming_auth_mode="${AUTH_MODE:-}"
incoming_api_key="${SINGLE_USER_API_KEY:-}"
incoming_database_url="${DATABASE_URL:-}"
incoming_jobs_db_url="${JOBS_DB_URL:-}"
incoming_database_url_override="${TLDW_DATABASE_URL_OVERRIDE:-}"
incoming_jobs_db_url_override="${TLDW_JOBS_DB_URL_OVERRIDE:-}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "[entrypoint] python3 or python is required" >&2
    exit 127
  fi
fi

generate_key() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -base64 32 | tr -d '\n'
  else
    "$PYTHON_BIN" -c "import secrets, base64; print(base64.b64encode(secrets.token_bytes(32)).decode())"
  fi
}

is_invalid_key() {
  key="$1"
  case "$key" in
    ""|change-me|CHANGE_ME_TO_SECURE_API_KEY|CHANGE_ME*|changeme|default|test-key)
      return 0
      ;;
  esac
  [ "${#key}" -lt 16 ]
}

derive_postgres_database_url() {
  "$PYTHON_BIN" - <<'PY'
import os
import sys
from urllib.parse import quote

password = os.getenv("POSTGRES_PASSWORD") or ""
if not password:
    sys.exit(1)

host = os.getenv("POSTGRES_HOST") or "postgres"
port = os.getenv("POSTGRES_PORT") or "5432"
user = os.getenv("POSTGRES_USER") or "postgres"
db = os.getenv("POSTGRES_DB") or "postgres"

sys.stdout.write(
    f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}@{host}:{port}/{quote(db, safe='')}"
)
PY
}

load_env_file() {
  if [ ! -f "$ENV_FILE" ]; then
    return
  fi

  parsed_env_file="$(mktemp "${TMPDIR:-/tmp}/tldw-env.XXXXXX")"
  if "$PYTHON_BIN" - "$ENV_FILE" > "$parsed_env_file" <<'PY'
import re
import sys

path = sys.argv[1]
key_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

try:
    env_file = open(path, encoding="utf-8", newline="")
except Exception as exc:
    print(f"[entrypoint] Failed to open env file {path}: {exc}", file=sys.stderr)
    sys.exit(1)

try:
    with env_file:
        for line_number, raw_line in enumerate(env_file, start=1):
            line = raw_line.rstrip("\n")
            if "\r" in line:
                print(
                    f"[entrypoint] Refusing env line with carriage return at {path}:{line_number}",
                    file=sys.stderr,
                )
                sys.exit(1)
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if "=" not in line:
                print(f"[entrypoint] Invalid env line in {path}:{line_number}", file=sys.stderr)
                sys.exit(1)

            key, value = line.split("=", 1)
            if not key_re.match(key):
                print(f"[entrypoint] Invalid env key in {path}:{line_number}: {key!r}", file=sys.stderr)
                sys.exit(1)
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            if "\n" in value:
                print(f"[entrypoint] Refusing env value with newline for key {key}", file=sys.stderr)
                sys.exit(1)
            print(f"{key}={value}")
except Exception as exc:
    print(f"[entrypoint] Failed to parse env file {path}: {exc}", file=sys.stderr)
    sys.exit(1)
PY
  then
    while IFS= read -r assignment || [ -n "$assignment" ]; do
      [ -n "$assignment" ] || continue
      export "$assignment"
    done < "$parsed_env_file"
    rm -f "$parsed_env_file"
  else
    rm -f "$parsed_env_file"
    exit 1
  fi
}

upsert_env() {
  key="$1"
  value="$2"
  mkdir -p "$(dirname "$ENV_FILE")"
  tmp_file="$(mktemp "${ENV_FILE}.tmp.XXXXXX")"
  chmod 600 "$tmp_file"
  if [ -f "$ENV_FILE" ]; then
    awk -v k="$key" -v v="$value" '
      BEGIN { updated = 0 }
      $0 ~ ("^" k "=") { print k "=" v; updated = 1; next }
      { print }
      END { if (!updated) print k "=" v }
    ' "$ENV_FILE" > "$tmp_file"
  else
    echo "$key=$value" > "$tmp_file"
  fi
  mv "$tmp_file" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
}

has_existing_auth_state() {
  [ -f "${AUTH_MARKER_DIR}/.authnz_initialized_single_user" ] || \
    [ -f "${AUTH_MARKER_DIR}/.authnz_initialized_multi_user" ] || \
    [ -f "${AUTH_MARKER_FILE}" ]
}

count_existing_byok_payloads() {
  "$PYTHON_BIN" - <<'PY'
import os
import sqlite3
import sys
from urllib.parse import unquote, urlparse

TABLES = ("user_provider_secrets", "org_provider_secrets")


def count_sqlite(database_url: str) -> int:
    parsed = urlparse(database_url)
    raw_path = unquote(parsed.path or "")
    if raw_path.startswith("//"):
        raw_path = raw_path[1:]
    db_path = raw_path if os.path.isabs(raw_path) else os.path.abspath(raw_path or "./Databases/users.db")
    if not os.path.exists(db_path):
        return 0

    conn = sqlite3.connect(db_path)
    try:
        total = 0
        for table in TABLES:
            try:
                row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            except sqlite3.Error:
                continue
            total += int((row or (0,))[0] or 0)
        return total
    finally:
        conn.close()


def count_postgres(database_url: str) -> int:
    import psycopg

    total = 0
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            for table in TABLES:
                cur.execute("SELECT to_regclass(%s)", (table,))
                row = cur.fetchone()
                if not row or row[0] is None:
                    continue
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count_row = cur.fetchone()
                total += int((count_row or (0,))[0] or 0)
    return total


database_url = (os.getenv("DATABASE_URL") or "sqlite:///./Databases/users.db").strip()
try:
    if database_url.startswith("postgres"):
        result = count_postgres(database_url)
    else:
        result = count_sqlite(database_url)
except Exception as exc:  # pragma: no cover - shell guard path
    print(f"error:{exc}", file=sys.stderr)
    sys.exit(1)

sys.stdout.write(str(result))
PY
}

ensure_env_file() {
  if [ -f "$ENV_FILE" ]; then
    return
  fi
  mkdir -p "$(dirname "$ENV_FILE")"
  generated_key="$(generate_key)"
  tmp_file="$(mktemp "${ENV_FILE}.tmp.XXXXXX")"
  chmod 600 "$tmp_file"
  cat > "$tmp_file" <<EOF
AUTH_MODE=single_user
SINGLE_USER_API_KEY=$generated_key
DATABASE_URL=sqlite:///./Databases/users.db
EOF
  mv "$tmp_file" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  echo "[entrypoint] Created $ENV_FILE with generated SINGLE_USER_API_KEY."
}

process_env_multi_user=0
if [ "$incoming_auth_mode" = "multi_user" ]; then
  process_env_multi_user=1
fi

if [ "$process_env_multi_user" = "0" ]; then
  ensure_env_file
fi

if [ -f "$ENV_FILE" ]; then
  load_env_file
fi

if [ -n "$incoming_auth_mode" ]; then
  AUTH_MODE="$incoming_auth_mode"
fi
if [ -n "$incoming_api_key" ]; then
  SINGLE_USER_API_KEY="$incoming_api_key"
fi
if [ -n "$incoming_database_url" ]; then
  DATABASE_URL="$incoming_database_url"
fi
if [ -n "$incoming_jobs_db_url" ]; then
  JOBS_DB_URL="$incoming_jobs_db_url"
fi
if [ -n "$incoming_database_url_override" ]; then
  TLDW_DATABASE_URL_OVERRIDE="$incoming_database_url_override"
fi
if [ -n "$incoming_jobs_db_url_override" ]; then
  TLDW_JOBS_DB_URL_OVERRIDE="$incoming_jobs_db_url_override"
fi

AUTH_MODE="${AUTH_MODE:-single_user}"
if [ "$AUTH_MODE" = "multi_user" ]; then
  process_env_multi_user=1
fi
database_url_derived=0
jobs_db_url_derived=0
if [ "$AUTH_MODE" = "multi_user" ]; then
  if [ -n "${DATABASE_URL:-}" ] && [ -z "${TLDW_DATABASE_URL_OVERRIDE:-}" ]; then
    echo "" >&2
    echo "======================================================================" >&2
    echo "  ERROR: Multi-user mode refuses DATABASE_URL from the docker env file." >&2
    echo "" >&2
    echo "  Remove DATABASE_URL from the docker-multi-postgres env file, or set" >&2
    echo "  TLDW_DATABASE_URL_OVERRIDE to intentionally use an external database." >&2
    echo "======================================================================" >&2
    echo "" >&2
    exit 1
  fi
  if [ -n "${JOBS_DB_URL:-}" ] && [ -z "${TLDW_JOBS_DB_URL_OVERRIDE:-}" ]; then
    echo "" >&2
    echo "======================================================================" >&2
    echo "  ERROR: Multi-user mode refuses JOBS_DB_URL from the docker env file." >&2
    echo "" >&2
    echo "  Remove JOBS_DB_URL from the docker-multi-postgres env file, or set" >&2
    echo "  TLDW_JOBS_DB_URL_OVERRIDE to intentionally use an external jobs database." >&2
    echo "======================================================================" >&2
    echo "" >&2
    exit 1
  fi

  if [ -n "${TLDW_DATABASE_URL_OVERRIDE:-}" ]; then
    DATABASE_URL="$TLDW_DATABASE_URL_OVERRIDE"
    database_url_derived=1
  elif [ -n "${POSTGRES_PASSWORD:-}" ]; then
    DATABASE_URL="$(derive_postgres_database_url)"
    database_url_derived=1
  else
    echo "" >&2
    echo "======================================================================" >&2
    echo "  ERROR: Multi-user mode requires POSTGRES_PASSWORD or TLDW_DATABASE_URL_OVERRIDE." >&2
    echo "" >&2
    echo "  Set POSTGRES_PASSWORD for the bundled docker-multi-postgres profile," >&2
    echo "  or set TLDW_DATABASE_URL_OVERRIDE for an external database." >&2
    echo "======================================================================" >&2
    echo "" >&2
    exit 1
  fi

  if [ -n "${TLDW_JOBS_DB_URL_OVERRIDE:-}" ]; then
    JOBS_DB_URL="$TLDW_JOBS_DB_URL_OVERRIDE"
    jobs_db_url_derived=1
  elif [ "$database_url_derived" = "1" ]; then
    JOBS_DB_URL="$DATABASE_URL"
    jobs_db_url_derived=1
  else
    JOBS_DB_URL="$DATABASE_URL"
    jobs_db_url_derived=1
  fi
else
  DATABASE_URL="${DATABASE_URL:-sqlite:///./Databases/users.db}"
fi
export DATABASE_URL
if [ -n "${JOBS_DB_URL:-}" ]; then
  export JOBS_DB_URL
fi

# Derive mode-specific marker so switching AUTH_MODE re-triggers init.
AUTH_MARKER_FILE="${AUTH_MARKER_DIR}/.authnz_initialized_${AUTH_MODE}"

upsert_env "AUTH_MODE" "$AUTH_MODE"
if [ "$database_url_derived" = "0" ]; then
  upsert_env "DATABASE_URL" "$DATABASE_URL"
fi
if [ -n "${JOBS_DB_URL:-}" ] && [ "$jobs_db_url_derived" = "0" ]; then
  upsert_env "JOBS_DB_URL" "$JOBS_DB_URL"
fi

if [ "$AUTH_MODE" = "single_user" ]; then
  current_key="${SINGLE_USER_API_KEY:-}"
  if is_invalid_key "$current_key"; then
    current_key="$(generate_key)"
    echo "[entrypoint] Generated secure SINGLE_USER_API_KEY for single_user mode."
  fi
  export SINGLE_USER_API_KEY="$current_key"
  upsert_env "SINGLE_USER_API_KEY" "$current_key"
fi

# Auto-generate MCP secrets if missing or placeholder (min 32 chars each)
for mcp_var in MCP_JWT_SECRET MCP_API_KEY_SALT BYOK_ENCRYPTION_KEY; do
  eval current_val="\${$mcp_var:-}"
  case "$current_val" in
    ""|CHANGE_ME*)
      if [ "$mcp_var" = "BYOK_ENCRYPTION_KEY" ]; then
        if existing_byok_payloads="$(count_existing_byok_payloads 2>/dev/null)"; then
          if [ "${existing_byok_payloads:-0}" -gt 0 ]; then
            echo "[entrypoint] Refusing to auto-generate BYOK_ENCRYPTION_KEY because ${existing_byok_payloads} encrypted BYOK payload(s) already exist. Reuse the previous key or configure BYOK_SECONDARY_ENCRYPTION_KEY for rotation." >&2
            exit 1
          fi
        elif has_existing_auth_state; then
          echo "[entrypoint] Existing auth state was detected but BYOK_ENCRYPTION_KEY could not be validated. Refusing to auto-generate a replacement key; provide the prior key explicitly." >&2
          exit 1
        fi
      fi
      new_val="$(generate_key)"
      eval export "$mcp_var=$new_val"
      upsert_env "$mcp_var" "$new_val"
      echo "[entrypoint] Generated $mcp_var."
      ;;
  esac
done

should_run_auth_init=0
case "$*" in
  *uvicorn*|*tldw_Server_API.app.main:app*)
    should_run_auth_init=1
    ;;
esac

if [ "$RUN_AUTH_INIT_ON_START" != "0" ] && [ "$should_run_auth_init" = "1" ]; then
  mkdir -p "$AUTH_MARKER_DIR"

  # --- Standard AuthNZ initialization (single_user and multi_user) ---
  if [ ! -f "$AUTH_MARKER_FILE" ]; then
    echo "[entrypoint] Running first-use auth initialization..."
    if "$PYTHON_BIN" -m tldw_Server_API.app.core.AuthNZ.initialize --non-interactive 2>&1; then
      touch "$AUTH_MARKER_FILE"
      echo "[entrypoint] Auth initialization complete."
    else
      echo "" >&2
      echo "╔══════════════════════════════════════════════════════════════╗" >&2
      echo "║  AUTH INITIALIZATION FAILED                                  ║" >&2
      echo "║                                                              ║" >&2
      echo "║  Check the error above and verify:                           ║" >&2
      echo "║  - MCP_JWT_SECRET is set in .env (min 32 chars)             ║" >&2
      echo "║  - MCP_API_KEY_SALT is set in .env (min 32 chars)           ║" >&2
      echo "║  - BYOK_ENCRYPTION_KEY is set (base64 encoded)              ║" >&2
      echo "║  - DATABASE_URL is correct (if multi-user)                   ║" >&2
      echo "╚══════════════════════════════════════════════════════════════╝" >&2
      echo "" >&2
      echo "[first-run] Auth initialization failed in non-interactive startup; refusing to continue." >&2
      exit 1
    fi
  fi

  # --- Multi-user admin bootstrap via environment variables ---
  if [ "$AUTH_MODE" = "multi_user" ]; then
    if [ -n "${ADMIN_USERNAME:-}" ] && [ -n "${ADMIN_PASSWORD:-}" ]; then
      echo "[first-run] Creating initial admin user: $ADMIN_USERNAME"
      # Idempotent: exits 0 if user already exists
      "$PYTHON_BIN" -m tldw_Server_API.app.core.AuthNZ.create_admin \
        --username "$ADMIN_USERNAME" \
        --password "$ADMIN_PASSWORD" \
        ${ADMIN_EMAIL:+--email "$ADMIN_EMAIL"} \
        --non-interactive 2>&1 || {
          echo "[first-run] ERROR: Admin bootstrap failed; refusing to continue startup." >&2
          exit 1
        }
    else
      # Check whether users exist; fail separately if account state cannot be verified.
      probe_err="$(mktemp)"
      if has_users=$("$PYTHON_BIN" -c "
import asyncio, sys
async def check():
    try:
        from tldw_Server_API.app.core.DB_Management.Users_DB import get_users_db
        db = await get_users_db()
        users = await db.list_users(limit=1)
        return len(users) > 0
    except Exception as exc:
        print(f'[first-run] Failed to verify existing users: {exc}', file=sys.stderr)
        sys.exit(1)
sys.stdout.write('1' if asyncio.run(check()) else '0')
" 2>"$probe_err"); then
        rm -f "$probe_err"
      else
        echo "" >&2
        echo "======================================================================" >&2
        echo "  ERROR: Could not verify whether existing multi-user accounts exist." >&2
        echo "" >&2
        echo "  Check DATABASE_URL and database connectivity before retrying startup." >&2
        if [ -s "$probe_err" ]; then
          echo "" >&2
          sed 's/^/  /' "$probe_err" >&2
        fi
        echo "======================================================================" >&2
        echo "" >&2
        rm -f "$probe_err"
        exit 1
      fi

      if [ "$has_users" = "0" ]; then
        echo "" >&2
        echo "======================================================================" >&2
        echo "  ERROR: Multi-user mode has no admin user and no admin bootstrap env." >&2
        echo "" >&2
        echo "  Set ADMIN_USERNAME and ADMIN_PASSWORD in tldw_Server_API/Config_Files/.env" >&2
        echo "  before starting the public docker-multi-postgres profile." >&2
        echo "======================================================================" >&2
        echo "" >&2
        exit 1
      fi
    fi
  fi
fi

exec "$@"
