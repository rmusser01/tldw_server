import extensionSidepanel from "@/public/_locales/en/sidepanel.json"
import sidepanel from "@/assets/locale/en/sidepanel.json"
import { describe, expect, it } from "vitest"

describe("sidepanel handoff locale copy", () => {
  it("mirrors the destination and bounded-copy descriptions used by the controls", () => {
    const copyKeys = ["openCharacterChatInFullExtension", "openCharacterChatExtensionDescription", "copyDraftContext", "copyDraftContextExtensionDescription", "copyDraftContextWebuiDescription", "copyDraftNothingToSend", "copyDraftPageReferencesUnavailable", "copyDraftFailed", "copyDraftOpened", "copyDraftOpenedPartial"]
    const headerKeys = ["openFullChatWebuiHandoffDescription", "openFullChatExtension", "openFullChatExtensionDescription"]
    for (const [section, keys] of [["header", headerKeys], ["controlRow", copyKeys]] as const) {
      for (const key of keys) {
        expect((extensionSidepanel as Record<string, { message: string }>)[`${section}_${key}`]?.message).toBe((sidepanel[section] as Record<string, string>)[key])
      }
    }
  })
})

it("mirrors opening choices and recovery copy in the source and flattened extension catalog", () => {
  const source = sidepanel as unknown as { greetingPicker?: Record<string, string> }
  expect(source.greetingPicker?.start).toBe("Start with this greeting")
  for (const [key, text] of Object.entries(source.greetingPicker ?? {}))
    expect((extensionSidepanel as Record<string, { message: string }>)[`greetingPicker_${key}`]?.message).toBe(text)
})
