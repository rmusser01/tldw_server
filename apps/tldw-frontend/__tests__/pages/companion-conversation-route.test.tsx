import { existsSync, readFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const page = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../pages/companion/conversation.tsx"
)

describe("Companion conversation WebUI route", () => {
  it("mounts the shared client route at the URL advertised by Home", () => {
    expect(existsSync(page)).toBe(true)
    const source = readFileSync(page, "utf8")
    expect(source).toContain('import("@/routes/option-companion-conversation")')
    expect(source).toContain("ssr: false")
  })
})
