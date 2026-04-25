import { existsSync, readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

const chatFormPathCandidates = [
  "src/components/Sidepanel/Chat/form.tsx",
  "../packages/ui/src/components/Sidepanel/Chat/form.tsx",
  "apps/packages/ui/src/components/Sidepanel/Chat/form.tsx"
]

const chatFormPath = chatFormPathCandidates.find((candidate) =>
  existsSync(candidate)
)

if (!chatFormPath) {
  throw new Error("Unable to locate Sidepanel chat form source for compact toolbar contract test")
}

const chatFormSource = readFileSync(chatFormPath, "utf8")

describe("sidepanel chat compact toolbar contract", () => {
  it("keeps compact icon controls at a minimum 44px touch target", () => {
    expect(chatFormSource).toMatch(/h-11 w-11 min-h-\[44px\] min-w-\[44px\]/)
  })

  it("includes visible compact labels for key icon actions", () => {
    expect(chatFormSource).toContain("playground:actions.uploadShort")
    expect(chatFormSource).toContain("playground:voiceChat.toggleShort")
    expect(chatFormSource).toContain("playground:actions.speechShort")
    expect(chatFormSource).toContain("playground:composer.stopShort")
  })

  it("keeps a visible pro-mode image attachment control outside overflow menus", () => {
    expect(chatFormSource).toContain('data-testid="chat-upload-image-inline"')
    expect(chatFormSource).toContain('aria-label={t(\n                                          "playground:actions.upload",\n                                          "Attach image"\n                                        )}')
    expect(chatFormSource).toContain("min-h-[44px] min-w-[44px]")
  })
})
