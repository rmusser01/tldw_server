import { afterEach, beforeEach, expect, it, vi } from "vitest"
import { exportPageAssistData, importPageAssistData } from "../export-import"
const state = vi.hoisted(() => ({ owner: "alice", exporting: vi.fn(), later: vi.fn(), imports: vi.fn(), inTransaction: false, ownerInTransaction: false, reader: null as any }))
vi.mock("@/db", () => ({ importChatHistory: vi.fn(), importPrompts: vi.fn() }))
vi.mock("@/services/recipe-persistence-uncertainty", () => ({ resolveRecipePersistenceOwnerView: async () => { if (state.inTransaction) state.ownerInTransaction = true; return { ownerId: state.owner } } }))
vi.mock("@/db/dexie/helpers", () => ({
  exportChatHistory: async () => [], exportPrompts: (...a: any[]) => state.exporting(...a), exportModels: (...a: any[]) => state.later(...a), exportNicknames: async () => [], exportOAIConfigs: async () => [],
  importChatHistoryV2: async () => {}, importModelsV2: async () => {}, importNicknamesV2: async () => {}, importPromptsV2: (...a: any[]) => state.imports(...a)
}))
vi.mock("@/db/dexie/schema", () => ({ db: { transaction: async (_mode: any, _tables: any, run: any) => { state.inTransaction = true; try { await run() } finally { state.inTransaction = false } } } }))
class Reader {
  result = ""; onload: any; onerror: any
  readAsText() { state.reader = this }
}
beforeEach(() => {
  state.owner = "alice"; state.inTransaction = false; state.ownerInTransaction = false; state.reader = null
  state.exporting.mockReset().mockResolvedValue([{ content: "ALICE PRIVATE" }]); state.later.mockReset().mockResolvedValue([]); state.imports.mockReset().mockResolvedValue({ imported: 1 })
  vi.stubGlobal("FileReader", Reader); URL.createObjectURL = vi.fn(() => "blob:private"); URL.revokeObjectURL = vi.fn()
})
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals() })
it("retires the settings backup download after the prompt read and a later account change", async () => {
  let release!: (v: any) => void
  state.later.mockReturnValue(new Promise(r => { release = r }))
  const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {})
  const pending = exportPageAssistData()
  await vi.waitFor(() => expect(state.later).toHaveBeenCalled())
  state.owner = "bob"; window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
  const outcome = expect(pending).rejects.toThrow(/account changed/)
  release([]); await outcome
  expect(click).not.toHaveBeenCalled(); expect(state.exporting).toHaveBeenCalledWith("alice")
})
it("retires a file restore before a new account can claim its old content", async () => {
  const pending = importPageAssistData({} as File)
  await vi.waitFor(() => expect(state.reader).not.toBeNull())
  state.owner = "bob"; window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
  state.reader.result = JSON.stringify({ prompts: [{ content: "ALICE" }] })
  const outcome = expect(pending).rejects.toThrow(/account changed/)
  await state.reader.onload(); await outcome
  expect(state.imports).not.toHaveBeenCalled()
})
it("passes captured authority into the atomic restore without auth work in the transaction", async () => {
  const pending = importPageAssistData({} as File)
  await vi.waitFor(() => expect(state.reader).not.toBeNull())
  state.reader.result = JSON.stringify({ chat: [], prompts: [{ id: "copy", content: "ALICE" }] })
  await state.reader.onload(); await pending
  const scope = state.imports.mock.calls[0][1].ownerScope
  expect(scope.ownerId).toBe("alice"); expect(state.ownerInTransaction).toBe(false)
})
