import { afterEach, expect, it } from "vitest"
import { useStoreMessageOption } from "@/store/option"
import type { HistorySelectionController } from "../useHistorySelection"
import { isSavedRoleplayConversation } from "../roleplay-resume-source"

const initial = useStoreMessageOption.getState()
afterEach(() => useStoreMessageOption.setState(initial, true))
const selection = {
  getCurrent: () => ({
    owner: {
      kind: "native",
      conversation_id: "saved",
      scope: { type: "global" }
    }
  })
} as unknown as HistorySelectionController
it("keeps a persona with a legacy character ID on its persona route", () => {
  useStoreMessageOption.setState({
    serverChatId: "saved",
    serverChatAssistantKind: "persona",
    serverChatCharacterId: 12,
    temporaryChat: false
  })
  expect(isSavedRoleplayConversation(selection)).toBe(false)
})
it("recognizes a saved snapshot character after the live card has been deleted", () => {
  useStoreMessageOption.setState({
    serverChatId: "saved",
    serverChatAssistantKind: "character",
    serverChatCharacterId: null,
    temporaryChat: false
  })
  expect(isSavedRoleplayConversation(selection)).toBe(true)
})
