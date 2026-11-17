import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, test } from "bun:test"

const extensionRoot = path.resolve(__dirname, "..", "..")
const modelsSourcePath = path.resolve(
  extensionRoot,
  "../packages/ui/src/db/models.ts"
)

const countOccurrences = (haystack: string, needle: string): number => {
  let count = 0
  let cursor = 0
  while (true) {
    const at = haystack.indexOf(needle, cursor)
    if (at === -1) return count
    count += 1
    cursor = at + needle.length
  }
}

// The call form only: `.get(null` with a receiver dot, so prose mentions of
// get(null) in comments do not count as scans.
const NEEDLE = ".get(null"

/**
 * WP-17 source guard: chrome.storage.local.get(null) dumps the ENTIRE
 * storage area (every chat, prompt, and settings key) and is only ever
 * legitimate inside the one-time legacy migration scan. Any other code
 * path in db/models.ts must read through the `model:` index instead.
 */
describe("ModelDb get(null) source guard", () => {
  test("get(null) appears only inside the one-time legacy migration", () => {
    const source = readFileSync(modelsSourcePath, "utf8")

    const migrationStart = source.indexOf("const migrateLegacyRecords")
    const migrationEnd = source.indexOf("export class ModelDb")
    expect(migrationStart).toBeGreaterThan(-1)
    expect(migrationEnd).toBeGreaterThan(migrationStart)

    let cursor = 0
    const offenders: number[] = []
    while (true) {
      const at = source.indexOf(NEEDLE, cursor)
      if (at === -1) break
      if (at <= migrationStart || at >= migrationEnd) {
        offenders.push(at)
      }
      cursor = at + 1
    }
    expect(offenders).toEqual([])

    // The migration itself stays a single full-area scan.
    const migrationSource = source.slice(migrationStart, migrationEnd)
    expect(countOccurrences(migrationSource, NEEDLE)).toBe(1)
  })
})
