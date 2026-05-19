import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const readOnboardingSource = () =>
  fs.readFileSync(path.resolve(__dirname, "..", "OnboardingConnectForm.tsx"), "utf8")

describe("OnboardingConnectForm review fixes", () => {
  it("keeps researcher shortcut persistence best-effort for ingest flow", () => {
    const source = readOnboardingSource()

    expect(source).toContain('const handleOpenIngestFlow = useCallback(async () => {')
    expect(source).toContain('console.debug("[OnboardingConnectForm]')
    expect(source).toContain('await finishAndNavigate("/media", { openQuickIngestIntro: true })')
  })

  it("includes actions and handleGoToChat as chat flow dependencies", () => {
    const source = readOnboardingSource()
    const chatFlowBlock = source.match(/const handleOpenChatFlow\s*=\s*useCallback\([\s\S]*?\},\s*\[([^\]]*)\]\)/)

    expect(chatFlowBlock).not.toBeNull()
    const deps = chatFlowBlock![1].split(",").map((d: string) => d.trim())
    expect(deps).toContain("actions")
    expect(deps).toContain("handleGoToChat")
  })
})
