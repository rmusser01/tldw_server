import { readFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const testDir = dirname(fileURLToPath(import.meta.url))

const readEntrypoint = (entrypoint: "options" | "sidepanel") =>
  readFileSync(resolve(testDir, "..", entrypoint, "main.tsx"), "utf8")

describe("split extension entrypoint i18n initialization", () => {
  it.each(["options", "sidepanel"] as const)(
    "%s initializes i18n before rendering React",
    (entrypoint) => {
      const source = readEntrypoint(entrypoint)
      const i18nImport = source.indexOf('import "@/i18n"')
      const renderCall = source.indexOf("ReactDOM.createRoot")

      expect(i18nImport).toBeGreaterThanOrEqual(0)
      expect(renderCall).toBeGreaterThan(i18nImport)
    }
  )
})
