/**
 * One periodic poller for Research Workspace source statuses (WP-42).
 *
 * The workspace used to run two independent 5 s intervals — the server status
 * projection and the per-source readiness checks — that kept issuing requests
 * while nothing was processing and while the page was hidden. The scheduler
 * below runs a single interval and a single `visibilitychange` listener:
 *
 *  - a tick only polls while the caller-supplied predicate says work is due
 *    (sources are processing);
 *  - ticks while the page is hidden are skipped, and one catch-up poll runs
 *    as soon as the page is shown again;
 *  - `stop()` clears both the interval and the listener.
 *
 * Callers keep their own one-shot reads (mount, workspace switch, processing
 * start); this module only owns the periodic cadence.
 */
export type WorkspaceStatusPollOptions = {
  intervalMs: number
  /** Re-evaluated on every tick and before every catch-up poll. */
  shouldPoll: () => boolean
  poll: () => void
}

export type WorkspaceStatusPoll = {
  stop: () => void
}

const isPageVisible = (): boolean =>
  typeof document === "undefined" || document.visibilityState !== "hidden"

export const startWorkspaceStatusPoll = (
  options: WorkspaceStatusPollOptions
): WorkspaceStatusPoll => {
  let missedWhileHidden = false

  const timer = window.setInterval(() => {
    if (!options.shouldPoll()) return
    if (!isPageVisible()) {
      missedWhileHidden = true
      return
    }
    options.poll()
  }, options.intervalMs)

  const onVisibilityChange = () => {
    if (!isPageVisible() || !missedWhileHidden) return
    missedWhileHidden = false
    if (options.shouldPoll()) options.poll()
  }
  document.addEventListener("visibilitychange", onVisibilityChange)

  return {
    stop() {
      window.clearInterval(timer)
      document.removeEventListener("visibilitychange", onVisibilityChange)
    }
  }
}
