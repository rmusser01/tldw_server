/**
 * Baseline ratchets for problems we cannot fix yet but must not let grow.
 *
 * A committed baseline lists today's known problems, each tagged with the
 * review id and GitHub issue that tracks its fix. compareRatchet() turns any
 * difference between what a test observed and that baseline into a failure:
 *  - a problem that is not in the baseline is a regression;
 *  - a baselined problem that grew beyond its tolerance is a regression;
 *  - a baselined problem that disappeared, or shrank below its tolerance,
 *    fails too, asking the fixer to update the baseline. This keeps the
 *    baseline honest in the same way as e2e/utils/known-defect.ts.
 */

export type RatchetEntry = {
  /** Stable identifier within a scope, e.g. an axe rule id or "GET /api/v1/buddies". */
  key: string
  /** Expected count (nodes, requests, duplicates). */
  count: number
  /**
   * Allowed deviation in either direction, for measured run-to-run noise.
   * Document why in `note` when it is not 0. A tolerance >= count accepts an
   * absent problem, for race-dependent offenders; such an entry cannot
   * report its own fix, so its note must say so.
   */
  tolerance?: number
  /** Review ids from Docs/Design/2026-10-02-notes-chat-ux-review.md, e.g. ["AX-07"]. */
  reviewIds: string[]
  /** GitHub issue that tracks the fix. */
  issue: number
  note: string
}

export type RatchetObservation = {
  count: number
  /** Extra context printed with regressions (sample selectors, URLs). */
  details?: string[]
}

export type RatchetComparison = {
  /** Where the observation was made, e.g. "/notes (empty library)". */
  scope: string
  /** What is counted, e.g. "axe violation" or "duplicate GET". */
  subject: string
  /** Baseline file, relative to apps/tldw-frontend, for the fix-it hints. */
  baselineFile: string
  observed: ReadonlyMap<string, RatchetObservation>
  baseline: readonly RatchetEntry[]
}

const describeEntry = (entry: RatchetEntry): string =>
  `${entry.reviewIds.join(", ") || "no review id"} (#${entry.issue})`

const withDetails = (message: string, details: string[] | undefined): string =>
  details && details.length > 0 ? `${message}\n    ${details.join("\n    ")}` : message

/** Every difference between `observed` and `baseline`, as human-readable problems. */
export function compareRatchet({
  scope,
  subject,
  baselineFile,
  observed,
  baseline,
}: RatchetComparison): string[] {
  const problems: string[] = []
  const baselined = new Map<string, RatchetEntry>()
  for (const entry of baseline) {
    if (baselined.has(entry.key)) {
      problems.push(`${scope}: duplicate baseline entry for ${subject} "${entry.key}" in ${baselineFile}`)
    }
    baselined.set(entry.key, entry)
  }

  for (const [key, observation] of observed) {
    if (baselined.has(key)) continue
    problems.push(
      withDetails(
        `${scope}: new ${subject} "${key}" (${observation.count}). It is not in the baseline: ` +
          `fix it, or if it is a known review finding add it to ${baselineFile} with its review id and issue.`,
        observation.details
      )
    )
  }

  for (const entry of baselined.values()) {
    const observation = observed.get(entry.key)
    const count = observation?.count ?? 0
    const tolerance = entry.tolerance ?? 0
    if (count === 0 && entry.count - tolerance > 0) {
      problems.push(
        `${scope}: baselined ${subject} "${entry.key}" [${describeEntry(entry)}] is no longer present, ` +
          `remove it from the baseline (${baselineFile}) so the ratchet guards the fix.`
      )
      continue
    }
    if (!observation) continue
    if (observation.count > entry.count + tolerance) {
      problems.push(
        withDetails(
          `${scope}: ${subject} "${entry.key}" [${describeEntry(entry)}] regressed: ` +
            `${observation.count}, baseline ${entry.count} ±${tolerance}.`,
          observation.details
        )
      )
    } else if (observation.count < entry.count - tolerance) {
      problems.push(
        `${scope}: ${subject} "${entry.key}" [${describeEntry(entry)}] improved: ` +
          `${observation.count}, baseline ${entry.count} ±${tolerance}. ` +
          `Lower its count in ${baselineFile} to ${observation.count} so the ratchet holds.`
      )
    }
  }

  return problems
}

const REVIEW_ID = /^[A-Z]{2,3}-\d{2,3}$/

/**
 * Shape problems in one baseline entry. Every entry must say which review
 * finding and issue it belongs to, so whoever fixes the issue can find it.
 */
export function ratchetEntryProblems(entry: Partial<RatchetEntry>, where: string): string[] {
  const problems: string[] = []
  if (typeof entry.key !== "string" || entry.key.length === 0) problems.push(`${where}: missing key`)
  if (!Number.isInteger(entry.count) || (entry.count ?? 0) < 1) problems.push(`${where}: count must be a positive integer`)
  if (entry.tolerance !== undefined && (!Number.isInteger(entry.tolerance) || entry.tolerance < 0)) {
    problems.push(`${where}: tolerance must be a non-negative integer`)
  }
  if (!Array.isArray(entry.reviewIds) || entry.reviewIds.length === 0 || !entry.reviewIds.every((id) => REVIEW_ID.test(id))) {
    problems.push(`${where}: reviewIds must list review ids such as "AX-07"`)
  }
  if (!Number.isInteger(entry.issue) || (entry.issue ?? 0) <= 0) problems.push(`${where}: issue must be a GitHub issue number`)
  if (typeof entry.note !== "string" || entry.note.trim().length === 0) problems.push(`${where}: note is required`)
  return problems
}

/** Fails with every ratchet problem at once, so one run shows the whole picture. */
export function assertNoRatchetProblems(problems: readonly string[]): void {
  if (problems.length === 0) return
  throw new Error(`Ratchet check failed (${problems.length}):\n- ${problems.join("\n- ")}`)
}
