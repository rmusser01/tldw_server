import { existsSync, readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

const loadSource = (...candidates: string[]) => {
  const path = candidates.find((candidate) => existsSync(candidate))
  if (!path) {
    throw new Error(`Missing settings sessions page shim: ${candidates.join(" | ")}`)
  }
  return readFileSync(path, "utf8")
}

describe("settings sessions Next.js page shim", () => {
  it("loads the settings-shell Active sessions route with its title", () => {
    const source = loadSource(
      "pages/settings/sessions.tsx",
      "tldw-frontend/pages/settings/sessions.tsx",
      "apps/tldw-frontend/pages/settings/sessions.tsx"
    )

    expect(source).toContain('import("@/components/Option/Settings/ActiveSessionsSettings")')
    expect(source).toContain("SettingsRoute")
    expect(source).toContain("ActiveSessionsSettings")
    expect(source).toContain('withPageTitle("Active sessions"')
  })
})
