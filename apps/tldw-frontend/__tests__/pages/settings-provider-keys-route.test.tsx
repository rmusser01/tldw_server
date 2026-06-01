import { existsSync, readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

const loadSource = (...candidates: string[]) => {
  const path = candidates.find((candidate) => existsSync(candidate))
  if (!path) {
    throw new Error(`Missing settings provider-keys page shim: ${candidates.join(" | ")}`)
  }
  return readFileSync(path, "utf8")
}

describe("settings provider keys Next.js page shim", () => {
  it("loads the settings-shell provider key management route", () => {
    const source = loadSource(
      "pages/settings/provider-keys.tsx",
      "tldw-frontend/pages/settings/provider-keys.tsx",
      "apps/tldw-frontend/pages/settings/provider-keys.tsx"
    )

    expect(source).toContain('import("@/components/Option/Settings/ProviderKeysSettings")')
    expect(source).toContain("SettingsRoute")
    expect(source).toContain("ProviderKeysSettings")
    expect(source).not.toContain("TldwSettings")
  })
})
