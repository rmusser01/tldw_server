import playground from "@/assets/locale/en/playground.json"
import extensionPlayground from "@/public/_locales/en/playground.json"
import extensionSidepanel from "@/public/_locales/en/sidepanel.json"
import sidepanel from "@/assets/locale/en/sidepanel.json"
import { describe, expect, it } from "vitest"

const extensionMessage = (catalog: Record<string, unknown>, key: string) => {
  const entry = catalog[key]
  return typeof entry === "object" && entry !== null && "message" in entry
    ? entry.message
    : undefined
}

describe("sidepanel handoff locale copy", () => {
  it("mirrors the destination and bounded-copy descriptions used by the controls", () => {
    const copyKeys = ["openCharacterChatInFullExtension", "openCharacterChatExtensionDescription", "copyDraftContext", "copyDraftContextExtensionDescription", "copyDraftContextWebuiDescription", "copyDraftNothingToSend", "copyDraftPageReferencesUnavailable", "copyDraftFailed", "copyDraftOpened", "copyDraftOpenedPartial"]
    const headerKeys = ["openFullChatWebuiHandoffDescription", "openFullChatExtension", "openFullChatExtensionDescription"]
    for (const [section, keys] of [["header", headerKeys], ["controlRow", copyKeys]] as const) {
      for (const key of keys) {
        expect(extensionMessage(extensionSidepanel, `${section}_${key}`)).toBe((sidepanel[section] as Record<string, string>)[key])
      }
    }
  })
})

it("mirrors opening choices and recovery copy in the source and flattened extension catalog", () => {
  expect(sidepanel.greetingPicker.start).toBe("Start with this greeting")
  for (const [key, text] of Object.entries(sidepanel.greetingPicker))
    expect(extensionMessage(extensionSidepanel, `greetingPicker_${key}`)).toBe(text)
})

it("mirrors fresh-copy checking and recovery feedback", () => {
  for (const [key, value] of Object.entries(playground.sidepanelHandoff))
    expect(extensionMessage(extensionPlayground, `sidepanelHandoff_${key}`)).toBe(value)
})
