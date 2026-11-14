/**
 * When the Buddy host re-reads Buddies and the attachment (XP-16).
 *
 * The host is mounted on every page, so a fixed 5 s poll kept two requests
 * going forever, even in background tabs. Instead:
 *  - a hidden page is not polled, and re-reads as soon as it is shown again;
 *  - each read that finds nothing new, or fails the same way again, doubles
 *    the wait, up to once a minute;
 *  - any change, or the page being shown again, returns to the 5 s cadence.
 *
 * Waits are counted in ticks of a fixed interval so the schedule is simple to
 * reason about and to drive from tests.
 */
export const BUDDY_POLL_TICK_MS = 5_000
export const BUDDY_POLL_MAX_IDLE_MS = 60_000

const MAX_IDLE_TICKS = Math.max(
  1,
  Math.floor(BUDDY_POLL_MAX_IDLE_MS / BUDDY_POLL_TICK_MS)
)

/** Ticks to wait before the next read after `unchangedReads` quiet reads. */
export const buddyPollTicksAfter = (unchangedReads: number): number =>
  Math.min(2 ** Math.max(0, unchangedReads), MAX_IDLE_TICKS)

const isPageVisible = (): boolean =>
  typeof document === "undefined" || document.visibilityState !== "hidden"

export type BuddyPoll = {
  /**
   * Report what a read saw: a successful read's `buddyReadSignature`, or a
   * failed read's `buddyFailureSignature`. A read that matches the previous
   * one slows the poll down; a different one speeds it back up.
   */
  settle: (signature: string) => void
  stop: () => void
}

/**
 * Calls `read` now (when the page is visible) and then on the schedule above.
 * The caller reports each read, successful or failed, through `settle`.
 */
export const startBuddyPoll = (read: () => void): BuddyPoll => {
  let unchangedReads = 0
  let ticksWaited = 0
  let lastSignature: string | null = null
  let missedWhileHidden = false

  const readNow = () => {
    ticksWaited = 0
    missedWhileHidden = false
    read()
  }

  if (isPageVisible()) readNow()
  else missedWhileHidden = true

  const timer = window.setInterval(() => {
    if (!isPageVisible()) {
      missedWhileHidden = true
      return
    }
    ticksWaited += 1
    if (ticksWaited >= buddyPollTicksAfter(unchangedReads)) readNow()
  }, BUDDY_POLL_TICK_MS)

  const onVisibilityChange = () => {
    if (!isPageVisible() || !missedWhileHidden) return
    unchangedReads = 0
    readNow()
  }
  document.addEventListener("visibilitychange", onVisibilityChange)

  return {
    settle(signature) {
      if (signature === lastSignature) {
        unchangedReads += 1
        return
      }
      unchangedReads = 0
      lastSignature = signature
    },
    stop() {
      window.clearInterval(timer)
      document.removeEventListener("visibilitychange", onVisibilityChange)
    }
  }
}

/** A compact fingerprint of what one Buddy read returned. */
export const buddyReadSignature = (read: {
  attachment: unknown
  profiles: readonly {
    id: string
    version?: number
    name?: string
    display_mode?: string
    optional_persona_available?: boolean
  }[]
  conversations: readonly unknown[]
}): string =>
  JSON.stringify([
    read.attachment,
    read.profiles.map((profile) => [
      profile.id,
      profile.version,
      profile.name,
      profile.display_mode,
      profile.optional_persona_available
    ]),
    read.conversations
  ])

/** The fingerprint of a failed read, so the same failure backs off too. */
export const buddyFailureSignature = (message: string): string =>
  `error:${message}`
