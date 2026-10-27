/**
 * Red-first reproductions for verified, still-open defects.
 *
 * Setup and preconditions run as ordinary test code, so a broken environment
 * still fails the test. Only the final "correct behaviour" assertion goes
 * through expectKnownDefect:
 *  - while the defect reproduces, the assertion fails and the test passes,
 *    recording the failure as evidence;
 *  - once the defect is fixed, the assertion passes and the test FAILS with a
 *    message asking the fixer to replace expectKnownDefect with a plain
 *    assertion, which turns the reproduction into a regression test.
 */
import type { TestInfo } from "@playwright/test"

export type KnownDefect = {
  /** Review issue id, e.g. "NL-01". */
  id: string
  /** GitHub issue that tracks the fix. */
  issue: number
  summary: string
}

export async function expectKnownDefect(
  testInfo: TestInfo,
  defect: KnownDefect,
  correctBehaviour: () => Promise<void> | void
): Promise<void> {
  const label = `${defect.id} (#${defect.issue})`
  testInfo.annotations.push({ type: "known-defect", description: `${label}: ${defect.summary}` })

  let reproduced = false
  try {
    await correctBehaviour()
  } catch (error) {
    reproduced = true
    const raw = error instanceof Error ? error.message : String(error)
    // eslint-disable-next-line no-control-regex -- strip terminal colour codes from expect output
    const message = raw.replace(/\u001b\[[0-9;]*m/g, "")
    testInfo.annotations.push({ type: "known-defect-evidence", description: message.slice(0, 500) })
  }

  if (!reproduced) {
    throw new Error(
      `${label} no longer reproduces: the correct behaviour now holds. ` +
        "Replace expectKnownDefect with a plain assertion so this test guards the fix."
    )
  }
}
