#!/bin/sh
# Drive a pull request's six required gates to green, then merge it.
#
# Workaround for TASK-13355: required gates are cancelled before they report, so a PR
# stays BLOCKED indefinitely unless someone re-runs them in the right window. See
# Docs/Development/CI_REQUIRED_GATES.md, "Known Defect: Gates Are Cancelled Before They
# Report". Delete this script when that task is resolved.
#
# Usage, from a checkout with the PR branch already rebased onto the current dev and
# pushed:
#
#     sh Helper_Scripts/ci/land_required_gates.sh <branch> <pr-number> [--no-merge]
#
# Three rules are encoded here because each was learned by getting it wrong:
#
#  1. The audit wait is SHA-aware. A rebase-push spawns a NEW Frontend License Gate
#     Audit; waiting only for "the latest audit is completed" can match the PREVIOUS
#     one, so the gates get re-run while the new audit is still pending and its
#     completion then cancels them through the shared concurrency group.
#  2. A run already queued or in_progress is left alone. `gh run rerun` fails on a
#     non-completed run, which silently skips that gate.
#  3. Re-runs are paced. Firing them in a burst makes them cancel one another.
set -eu

BRANCH="${1:?usage: land_required_gates.sh <branch> <pr-number> [--no-merge]}"
PR="${2:?usage: land_required_gates.sh <branch> <pr-number> [--no-merge]}"
NO_MERGE="${3:-}"

REPO="${LAND_GATES_REPO:-rmusser01/tldw_server}"
GATES="backend-required security-required coverage-required frontend-required e2e-required container-build-check"
AUDIT="Frontend License Gate Audit"
PACE="${LAND_GATES_PACE:-40}"

# Resolved from the pull request, not from local HEAD: this has to work without being
# checked out on the branch, and local HEAD can differ from what was actually pushed.
HEAD_SHA="$(gh pr view --repo "$REPO" "$PR" --json headRefOid --template '{{.headRefOid}}')"
SHORT="$(printf '%s' "$HEAD_SHA" | cut -c1-8)"
if [ -z "$HEAD_SHA" ]; then
  echo "could not resolve the head sha for PR #$PR" >&2
  exit 2
fi

latest_run() {
  # $1 = workflow name; prints "<id> <status> <conclusion> <headSha>"
  gh run list --repo "$REPO" --branch "$BRANCH" --workflow "$1" --limit 1 \
    --json databaseId,status,conclusion,headSha \
    --template '{{range .}}{{printf "%.0f" .databaseId}} {{.status}} {{.conclusion}} {{.headSha}}{{end}}'
}

printf '=== PR #%s (%s @ %s): waiting for the audit on this sha ===\n' "$PR" "$BRANCH" "$SHORT"
i=0
while [ "$i" -lt 90 ]; do
  set -- $(latest_run "$AUDIT" || true)
  status="${2:-unknown}"
  sha="${4:-}"
  if [ "$status" = "completed" ] && [ "$sha" = "$HEAD_SHA" ]; then
    echo "audit for $SHORT completed"
    break
  fi
  printf '  waiting: audit %s on %s\n' "$status" "$(printf '%s' "$sha" | cut -c1-8)"
  i=$((i + 1))
  sleep 20
done

printf '=== PR #%s: ensuring the six gates run ===\n' "$PR"
for wf in $GATES; do
  set -- $(latest_run "$wf" || true)
  id="${1:-}"
  status="${2:-}"
  concl="${3:-}"
  case "$status:$concl" in
    completed:success)        printf '  %-24s already success\n' "$wf";;
    queued:*|in_progress:*)   printf '  %-24s already running (%s)\n' "$wf" "$status";;
    *)
      if [ -n "$id" ] && gh run rerun --repo "$REPO" "$id" >/dev/null 2>&1; then
        printf '  %-24s re-ran %s\n' "$wf" "$id"
        sleep "$PACE"
      else
        printf '  %-24s rerun failed (%s/%s)\n' "$wf" "$status" "$concl"
      fi
      ;;
  esac
done

printf '=== PR #%s: waiting for the six gates ===\n' "$PR"
i=0
while [ "$i" -lt 150 ]; do
  ok=0
  bad=''
  for wf in $GATES; do
    set -- $(latest_run "$wf" || true)
    case "${3:-}" in
      success)                  ok=$((ok + 1));;
      failure|timed_out|cancelled) bad="$bad $wf:${3}";;
    esac
  done
  if [ -n "$bad" ]; then
    printf 'PR #%s GATE PROBLEM:%s\n' "$PR" "$bad"
    exit 1
  fi
  if [ "$ok" -eq 6 ]; then
    printf 'PR #%s: all six gates green\n' "$PR"
    break
  fi
  i=$((i + 1))
  sleep 30
done

if [ "$NO_MERGE" = "--no-merge" ]; then
  printf 'PR #%s: --no-merge given, stopping before merge\n' "$PR"
  exit 0
fi

printf '=== PR #%s: merging ===\n' "$PR"
# --merge, not --squash: squash merges are rejected repository-wide.
gh pr merge --repo "$REPO" "$PR" --merge
