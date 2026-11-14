import { readFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

/**
 * WP-18 / TASK-13522 Stage 4 source guard: messageHandlers must not import
 * the legacy chrome.storage blob DB (`@/db`). Runtime and type-only imports
 * both count — the Dexie module graph (`@/db/dexie/*`) is the only permitted
 * DB surface here. The legacy class remains only as the Firefox-private-mode
 * fallback reached through `db/dexie/*` DatabaseClosedError handling.
 */

const testDir = dirname(fileURLToPath(import.meta.url))
const source = readFileSync(resolve(testDir, "../messageHandlers.ts"), "utf8")

const importLines = source.match(/^import[\s\S]*?from\s+["'][^"']+["']/gm) ?? []

describe("messageHandlers legacy db import guard", () => {
  it("imports nothing from the legacy chrome.storage blob DB", () => {
    const legacyImports = importLines.filter((line) =>
      /from\s+["'](@\/db|\.\.\/\.\.\/db|~\/db)["']\s*$/.test(line)
    )
    expect(legacyImports).toEqual([])
  })

  it("keeps pulling its DB helpers from the Dexie implementations", () => {
    const dexieImports = importLines.filter((line) =>
      /from\s+["']@\/db\/dexie\//.test(line)
    )
    expect(dexieImports.length).toBeGreaterThan(0)
    expect(
      dexieImports.some((line) => line.includes("@/db/dexie/helpers"))
    ).toBe(true)
  })
})
