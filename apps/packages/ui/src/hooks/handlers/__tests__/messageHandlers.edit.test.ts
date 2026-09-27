import { describe, expect, it, vi } from "vitest"
import type { ChatHistory, Message } from "@/store/option"
vi.mock("@/db/dexie/helpers", () => ({ updateMessageByIndex: vi.fn(), deleteChatForEdit: vi.fn() }))
import { createEditMessage } from "../messageHandlers"

const fixture = () => {
  const generationInfo = { mode: "rag", grounded: false, reason: "selected_source_retrieval_failed" }
  const messages: Message[] = [
    { id: "diagnostic-user", isBot: false, name: "You", message: "Source question", sources: [], images: [], generationInfo },
    { id: "diagnostic-answer", parentMessageId: "diagnostic-user", isBot: true, name: "Assistant", message: "Diagnostic", sources: [], generationInfo },
    { id: "real-user", isBot: false, name: "You", message: "Real question", sources: [], images: [] },
    { id: "real-answer", parentMessageId: "real-user", isBot: true, name: "Assistant", message: "Real answer", sources: [] }
  ]
  const history: ChatHistory = [{ role: "user", content: "Real question" }, { role: "assistant", content: "Real answer" }]
  const setHistory = vi.fn()
  const onSubmit = vi.fn()
  return { history, setHistory, onSubmit, edit: createEditMessage({ messages, history, setMessages: vi.fn(), setHistory, historyId: "local", validateBeforeSubmitFn: () => true, onSubmit }) }
}

describe("editing visible rows after a local-only diagnostic", () => {
  it("resends a genuine edited question without retaining its old answer in prompt memory", async () => {
    const { edit, onSubmit } = fixture()
    await edit(2, "Edited question", true, true)
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({ message: "Edited question", memory: [] }))
  })
  it("updates the genuine prompt row at its eligible index", async () => {
    const { edit, setHistory } = fixture()
    await edit(2, "Edited question", true, false)
    expect(setHistory).toHaveBeenCalledWith([{ role: "user", content: "Edited question" }, { role: "assistant", content: "Real answer" }])
  })
  it("keeps a local diagnostic edit out of the genuine prompt row", async () => {
    const { edit, setHistory, history } = fixture()
    await edit(0, "Edited diagnostic question", true, false)
    expect(setHistory).toHaveBeenCalledWith(history)
  })
})
