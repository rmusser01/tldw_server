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
