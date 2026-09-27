import { existsSync, readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

const loadSource = (...candidates: string[]) => {
  const path = candidates.find((candidate) => existsSync(candidate))
  if (!path) {
    throw new Error(`Missing settings chat-macros page shim: ${candidates.join(" | ")}`)
  }
  return readFileSync(path, "utf8")
}

describe("settings chat macros Next.js page shim", () => {
  it("loads the settings-shell chat macro manager", () => {
    const source = loadSource(
      "pages/settings/chat-macros.tsx",
      "tldw-frontend/pages/settings/chat-macros.tsx",
      "apps/tldw-frontend/pages/settings/chat-macros.tsx"
    )

    expect(source).toContain('import("@/components/Option/Settings/ChatMacrosSettings")')
    expect(source).toContain("SettingsRoute")
    expect(source).toContain("ChatMacrosSettings")
  })
})
