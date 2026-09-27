import { existsSync, readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

const loadSource = (...candidates: string[]) => {
  const path = candidates.find((candidate) => existsSync(candidate))
  if (!path) {
    throw new Error(`Missing Workflow prompts page shim: ${candidates.join(" | ")}`)
  }
  return readFileSync(path, "utf8")
}

describe("Workflow prompts Next.js page shim", () => {
  it("loads the shared Service Prompts editor inside SettingsRoute", () => {
    const source = loadSource(
      "pages/settings/prompt.tsx",
      "tldw-frontend/pages/settings/prompt.tsx",
      "apps/tldw-frontend/pages/settings/prompt.tsx"
    )

    expect(source).toContain(
      'import("@/components/Option/Settings/ServicePromptsSettings")'
    )
    expect(source).toContain("ServicePromptsSettings")
    expect(source).toContain("SettingsRoute")
    expect(source).not.toContain("PromptWorkspaceSettings")
  })
})
