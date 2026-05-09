import { readFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

import { describe, expect, it } from "vitest"
import { formatModelsLastRefreshedTime } from "../modelsDisplayUtils"

describe("models display utilities", () => {
  it.each([
    [new Date(2026, 1, 18, 9, 5).getTime(), "09:05"],
    [new Date(2026, 1, 18, 0, 7).getTime(), "00:07"],
  ])("formats %s as a dayjs-compatible HH:mm time", (value, expected) => {
    expect(formatModelsLastRefreshedTime(value)).toBe(expected)
  })

  it("does not depend on dayjs for last-refreshed display formatting", () => {
    const testDir = dirname(fileURLToPath(import.meta.url))
    const source = readFileSync(resolve(testDir, "../index.tsx"), "utf8")
    const dayjsImportPattern = /^\s*import\s+(?:type\s+)?(?:.+?\s+from\s+)?["']dayjs(?:\/[^"']*)?["']/m

    expect(source).not.toMatch(dayjsImportPattern)
  })
})
