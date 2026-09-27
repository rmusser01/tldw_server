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
  gh run list --repo "$REPO" --branch "$BRANCH" --workflow "$1" --commit "$HEAD_SHA" --limit 1 \
    --json databaseId,status,conclusion,headSha \
    --template '{{range .}}{{printf "%.0f" .databaseId}} {{.status}} {{if .conclusion}}{{.conclusion}}{{else}}pending{{end}} {{.headSha}}{{end}}'
}

printf '=== PR #%s (%s @ %s): waiting for the audit on this sha ===\n' "$PR" "$BRANCH" "$SHORT"
audit_passed=false
i=0
while [ "$i" -lt 90 ]; do
  set -- $(latest_run "$AUDIT" || true)
  status="${2:-unknown}"
  sha="${4:-}"
  if [ "$status" = "completed" ] && [ "$sha" = "$HEAD_SHA" ]; then
    if [ "${3:-}" != "success" ]; then
      echo "audit for $SHORT did not pass" >&2
      exit 1
    fi
    audit_passed=true
    echo "audit for $SHORT passed"
    break
  fi
  printf '  waiting: audit %s on %s\n' "$status" "$(printf '%s' "$sha" | cut -c1-8)"
  i=$((i + 1))
  sleep 20
done

[ "$audit_passed" = true ] || { echo "Timed out waiting for the current-head audit" >&2; exit 1; }

printf '=== PR #%s: ensuring the six gates run ===\n' "$PR"
for wf in $GATES; do
  set -- $(latest_run "$wf" || true)
  id="${1:-}"
  status="${2:-}"
  concl="${3:-}"
  if [ "${4:-}" != "$HEAD_SHA" ]; then
    echo "No current-head run for $wf; refusing to use or rerun stale checks" >&2
    exit 1
  fi
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
gates_passed=false
i=0
while [ "$i" -lt 150 ]; do
  ok=0
  bad=''
  for wf in $GATES; do
    set -- $(latest_run "$wf" || true)
    if [ "${4:-}" != "$HEAD_SHA" ]; then
      bad="$bad $wf:stale-head"
      continue
    fi
    case "${2:-}:${3:-}" in
      completed:success)                  ok=$((ok + 1));;
      completed:failure|completed:timed_out|completed:cancelled) bad="$bad $wf:${3}";;
    esac
  done
  if [ -n "$bad" ]; then
    printf 'PR #%s GATE PROBLEM:%s\n' "$PR" "$bad"
    exit 1
  fi
  if [ "$ok" -eq 6 ]; then
    gates_passed=true
    printf 'PR #%s: all six gates green\n' "$PR"
    break
  fi
  i=$((i + 1))
  sleep 30
done

[ "$gates_passed" = true ] || { echo "Timed out waiting for six current-head gates" >&2; exit 1; }

CURRENT_HEAD="$(gh pr view --repo "$REPO" "$PR" --json headRefOid --template '{{.headRefOid}}')"
[ "$CURRENT_HEAD" = "$HEAD_SHA" ] || { echo "PR head changed during validation" >&2; exit 1; }

if [ "$NO_MERGE" = "--no-merge" ]; then
  printf 'PR #%s: --no-merge given, stopping before merge\n' "$PR"
  exit 0
fi

printf '=== PR #%s: merging ===\n' "$PR"
# --merge, not --squash: squash merges are rejected repository-wide.
gh pr merge --repo "$REPO" "$PR" --merge --match-head-commit "$HEAD_SHA"
