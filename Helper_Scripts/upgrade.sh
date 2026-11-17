#!/usr/bin/env bash
# ===========================================================================
# upgrade.sh — Automated upgrade script for tldw_server
#
# Usage:
#   ./Helper_Scripts/upgrade.sh [OPTIONS]
#
# Options:
#   --dry-run            Run pre-flight checks only; make no changes
#   --target-version TAG Pull a specific git tag/branch instead of latest
#   --confirm-offline    Confirm API and all workers are stopped and cannot reconnect
#
# Backups cover repository Databases/ only. Preserve external profile data,
# source files and configuration separately before upgrading.
#   --help               Show this help and exit
# ===========================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m' # No Colour

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "\n${BOLD}==> $*${NC}"; }

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DRY_RUN=false
CONFIRM_OFFLINE=false
TARGET_VERSION=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="${REPO_ROOT}/Backups/upgrade_${TIMESTAMP}"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --confirm-offline)
            CONFIRM_OFFLINE=true
            shift
            ;;
        --target-version)
            TARGET_VERSION="${2:-}"
            if [[ -z "$TARGET_VERSION" ]]; then
                log_error "--target-version requires a value"
                exit 1
            fi
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--confirm-offline] [--target-version TAG] [--help]"
            echo ""
            echo "Options:"
            echo "  --dry-run            Run pre-flight checks only; make no changes"
            echo "  --target-version TAG Pull a specific git tag/branch instead of latest"
            echo "  --confirm-offline    Confirm API and all workers are stopped and cannot reconnect"
            echo "  Backup scope: repository Databases/ only; preserve external data/files/config separately."
            echo "  --help               Show this help and exit"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Run $0 --help for usage."
            exit 1
            ;;
    esac
done

if ! $DRY_RUN && ! $CONFIRM_OFFLINE; then
    log_error "Stop the API and all workers, prevent reconnects, then pass --confirm-offline."
    exit 1
fi

cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Step 1: Pre-flight checks
# ---------------------------------------------------------------------------
log_step "Step 1/7: Running pre-upgrade checks"

PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" &>/dev/null; then
    log_error "Python not found. Set PYTHON= to your interpreter."
    exit 1
fi

# Run the upgrade helpers module; it exits 0 if can_proceed, 1 otherwise.
PREFLIGHT_FILE="$(mktemp "${TMPDIR:-/tmp}/tldw_upgrade_checks.XXXXXX")"
trap 'rm -f -- "$PREFLIGHT_FILE"' EXIT

if "$PYTHON" -m Helper_Scripts.upgrade_helpers --json > "$PREFLIGHT_FILE" 2>&1; then
    log_info "All pre-upgrade checks passed."
else
    log_error "Pre-upgrade checks failed:"
    cat "$PREFLIGHT_FILE"
    exit 1
fi

# Pretty-print the summary
"$PYTHON" -c "
import json, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(f\"  Summary: {data['summary']}\")
for c in data['checks']:
    icon = {'ok':'✓','warn':'!','fail':'✗'}.get(c['status'],'?')
    print(f\"    [{icon}] {c['name']}: {c.get('message','')}\")
" "$PREFLIGHT_FILE"

if $DRY_RUN; then
    log_info "Dry-run mode — stopping here. No changes were made."
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 2: Record current state for rollback
# ---------------------------------------------------------------------------
log_step "Step 2/7: Recording current state"

CURRENT_COMMIT="$(git rev-parse HEAD)"
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
log_info "Current commit: ${CURRENT_COMMIT}"
log_info "Current branch: ${CURRENT_BRANCH}"

# ---------------------------------------------------------------------------
# Step 3: Create backup
# ---------------------------------------------------------------------------
log_step "Step 3/7: Creating backup"

mkdir -p "${BACKUP_DIR}"

# Save rollback info
cat > "${BACKUP_DIR}/rollback_info.txt" <<ROLLBACK
# tldw_server upgrade rollback information
# Created: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
COMMIT=${CURRENT_COMMIT}
BRANCH=${CURRENT_BRANCH}

# Recovery:
#   Keep API and workers stopped and prevent reconnects.
#   Restore a verified backup into a separate empty target with the matching
#   prior-release environment; verify data/files/config before switching.
#   This directory contains repository Databases/ only, not external profile data.
#   Never run the old binary against migrated data or merge-copy over live stores.
#   See Docs/Database_Migrations.md for supported recovery limits.
ROLLBACK

# Backup databases if they exist
if [[ -d "${REPO_ROOT}/Databases" ]]; then
    log_info "Backing up Databases/ ..."
    cp -a "${REPO_ROOT}/Databases" "${BACKUP_DIR}/Databases"
    log_info "Database backup complete: ${BACKUP_DIR}/Databases/"
else
    log_info "No Databases/ directory found — skipping backup."
fi

log_info "Backup saved to ${BACKUP_DIR}/"

# ---------------------------------------------------------------------------
# Step 4: Pull latest code
# ---------------------------------------------------------------------------
log_step "Step 4/7: Pulling latest code"

if [[ -n "$TARGET_VERSION" ]]; then
    log_info "Fetching and checking out target: ${TARGET_VERSION}"
    git fetch --tags origin
    git checkout "${TARGET_VERSION}"
else
    log_info "Pulling latest from origin/${CURRENT_BRANCH}"
    git pull origin "${CURRENT_BRANCH}"
fi

NEW_COMMIT="$(git rev-parse HEAD)"
log_info "Now at commit: ${NEW_COMMIT}"

# ---------------------------------------------------------------------------
# Step 5: Install dependencies
# ---------------------------------------------------------------------------
log_step "Step 5/7: Installing dependencies"

if "$PYTHON" -m pip install -e . 2>&1; then
    log_info "Dependencies installed successfully."
else
    log_error "pip install failed. See output above."
    log_warn "Rollback instructions are in ${BACKUP_DIR}/rollback_info.txt"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 6: Database migrations
# ---------------------------------------------------------------------------
log_step "Step 6/7: Database migrations"
log_info "Migrations run automatically at server startup (ensure_authnz_schema_ready_once)."
log_info "Skipping explicit migration step — start the server to apply."

# ---------------------------------------------------------------------------
# Step 7: Post-upgrade validation
# ---------------------------------------------------------------------------
log_step "Step 7/7: Post-upgrade validation"

if "$PYTHON" -m Helper_Scripts.upgrade_helpers 2>&1; then
    log_info "Post-upgrade checks passed."
else
    log_error "Post-upgrade preflight failed. Keep API and workers stopped; inspect the failure before startup."
    log_warn "Recovery information: ${BACKUP_DIR}/rollback_info.txt"
    exit 1
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}${BOLD}Code and dependencies updated.${NC}"
echo "Database migrations are pending; start the server and verify every configured store before accepting the upgrade."
echo ""
echo "  Previous commit: ${CURRENT_COMMIT}"
echo "  Current commit:  ${NEW_COMMIT}"
echo "  Backup location: ${BACKUP_DIR}/"
echo ""
echo -e "${BOLD}Recovery:${NC}"
echo "  Keep API and workers stopped; restore a verified backup into a separate empty target."
echo "  Use the matching prior-release environment and verify data/files/config before switching."
echo "  This backup covers repository Databases/ only; preserve external data/files/config separately."
echo "  Never run an old binary against migrated data or merge-copy over live stores."
echo "  See Docs/Database_Migrations.md for supported recovery limits."
echo ""
echo "See ${BACKUP_DIR}/rollback_info.txt for details."
