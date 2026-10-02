# Sourced by composite actions that apt-get install on Linux runners.
#
# Acquire::*::Timeout only catches stalled connections. A mirror that keeps trickling
# data hung `apt-get update` until the job timeout (TASK-13415), so each attempt also
# gets a wall-clock bound, and a bounded number of retries.

apt_opts=(-o Acquire::Retries=3 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20 -o Acquire::Languages=none)

apt_bounded() {
  local attempt
  for attempt in 1 2 3; do
    if sudo timeout --kill-after=15s "${APT_ATTEMPT_SECONDS:-300}" apt-get "$@"; then
      return 0
    fi
    echo "::warning::apt-get $1 attempt ${attempt} failed or timed out; retrying"
    sudo dpkg --configure -a || true
    sleep $((attempt * ${APT_RETRY_BASE_SECONDS:-5}))
  done
  echo "::error::apt-get $1 failed after 3 bounded attempts (stalled package mirror?)"
  return 1
}
