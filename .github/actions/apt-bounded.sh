# Sourced by composite actions that apt-get install on Linux runners.
#
# Acquire::*::Timeout only catches stalled connections. A mirror that keeps trickling
# data hung `apt-get update` until the job timeout (TASK-13415). Every apt call made
# after sourcing shares one deadline (default 10 minutes, below the shortest job that
# installs packages), and each attempt and the dpkg recovery between attempts is
# bounded by what is left of it, so the action reports its own error instead of the job
# timing out.

apt_opts=(-o Acquire::Retries=3 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20 -o Acquire::Languages=none)
apt_deadline=$((SECONDS + ${APT_TOTAL_SECONDS:-600}))

_apt_remaining_capped() {
  local remaining=$((apt_deadline - SECONDS))
  if [ "$remaining" -gt "$1" ]; then
    remaining=$1
  fi
  echo "$remaining"
}

apt_bounded() {
  local attempt limit
  for attempt in 1 2 3; do
    limit=$(_apt_remaining_capped "${APT_ATTEMPT_SECONDS:-240}")
    if [ "$limit" -le 0 ]; then
      break
    fi
    if sudo timeout --kill-after=15s "$limit" apt-get "$@"; then
      return 0
    fi
    echo "::warning::apt-get $1 attempt ${attempt} failed or timed out; retrying"
    limit=$(_apt_remaining_capped 60)
    if [ "$limit" -gt 0 ]; then
      sudo timeout --kill-after=10s "$limit" dpkg --configure -a || true
    fi
    sleep $((attempt * ${APT_RETRY_BASE_SECONDS:-5}))
  done
  echo "::error::apt-get $1 failed within the apt time budget (stalled package mirror?)"
  return 1
}
