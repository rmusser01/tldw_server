import extensionSidepanel from "@/public/_locales/en/sidepanel.json"
import sidepanel from "@/assets/locale/en/sidepanel.json"
import { describe, expect, it } from "vitest"

describe("sidepanel handoff locale copy", () => {
  it("keeps English full-chat handoff copy explicit about transfer behavior", () => {
    expect(sidepanel.header.openFullChatWebuiDescription).toBe(
      "Opens /chat in a new tab. Use Continue in WebUI from the composer tools to carry a draft or page context."
    )
    expect(sidepanel.header.openFullChatWebuiRouteOnlyDescription).toBe(
      "Opens /chat in a new tab without transferring the current draft or page context."
    )
    expect(sidepanel.controlRow.openFullAppDescription).toBe(
      "Opens /chat in a new tab. Use Continue in WebUI to carry a draft or page context."
    )
    expect(sidepanel.controlRow.openRolePlayFullAppDescription).toBe(
      "Opens /chat in a new tab with the active role-play route. Use Continue in WebUI to carry a draft or page context."
    )
  })
})

it("mirrors opening choices and recovery copy in the source and flattened extension catalog", () => {
  const source = sidepanel as unknown as { greetingPicker?: Record<string, string> }
  expect(source.greetingPicker?.start).toBe("Start with this greeting")
  for (const [key, text] of Object.entries(source.greetingPicker ?? {}))
    expect((extensionSidepanel as Record<string, { message: string }>)[`greetingPicker_${key}`]?.message).toBe(text)
})
