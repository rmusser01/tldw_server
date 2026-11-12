import { act, renderHook } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
import { usePromptImportExport } from "../hooks/usePromptImportExport"
const state = vi.hoisted(() => ({ exports: vi.fn(), server: vi.fn(), imports: vi.fn(), notice: vi.fn() }))
vi.mock("@/db/dexie/helpers", () => ({ exportPrompts: (...a: any[]) => state.exports(...a), importPromptsV2: (...a: any[]) => state.imports(...a) }))
vi.mock("@/services/prompts-api", () => ({ exportPromptsServer: (...a: any[]) => state.server(...a) }))
vi.mock("antd", () => ({ notification: { error: state.notice, success: state.notice, warning: state.notice, info: state.notice } }))
const deferred = () => { let resolve!: (v: any) => void; const promise = new Promise<any>(r => { resolve = r }); return { promise, resolve } }
const setup = (confirmDanger = async () => true) => renderHook(({ owner }) => usePromptImportExport({ expectedOwnerId: owner, queryClient: { invalidateQueries: vi.fn() } as any, data: [{ id: "a" }], isOnline: true, t: key => key, guardPrivateMode: () => false, confirmDanger }), { initialProps: { owner: "alice" } })
beforeEach(() => {
  vi.restoreAllMocks(); state.exports.mockReset().mockResolvedValue([]); state.server.mockReset(); state.imports.mockReset().mockResolvedValue({ imported: 1, skipped: 0, failed: 0 }); state.notice.mockReset()
  URL.createObjectURL = vi.fn(() => "blob:private"); URL.revokeObjectURL = vi.fn()
})
it.each(["json", "csv"] as const)("does not download late %s export after unmount", async format => {
  const gate = deferred()
  if (format === "json") state.exports.mockReturnValue(gate.promise)
  else state.server.mockReturnValue(gate.promise)
  const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {})
  const view = setup(); act(() => view.result.current.setExportFormat(format))
  const pending = view.result.current.triggerExport(); view.unmount()
  await act(async () => { gate.resolve(format === "json" ? [{ content: "ALICE PRIVATE" }] : { file_content_b64: btoa("ALICE PRIVATE") }); await pending })
  expect(click).not.toHaveBeenCalled(); expect(state.notice).not.toHaveBeenCalled()
})
it.each([true, false])("retires exports on config authority changes while preserving rotation (%s)", async authorityChanged => {
  const gate = deferred(); state.exports.mockReturnValue(gate.promise)
  const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {})
  const view = setup(); const pending = view.result.current.triggerExport()
  window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged } }))
  await act(async () => { gate.resolve([{ content: "ALICE PRIVATE" }]); await pending })
  expect(click).toHaveBeenCalledTimes(authorityChanged ? 0 : 1)
  view.unmount()
})
it("does not import a file after the owning account changes during its read", async () => {
  const gate = deferred(); const view = setup()
  const pending = view.result.current.handleImportFile({ text: () => gate.promise } as File)
  view.rerender({ owner: "bob" })
  await act(async () => { gate.resolve(JSON.stringify([{ id: "new", title: "Private", content: "ALICE" }])); await pending })
  expect(state.imports).not.toHaveBeenCalled(); expect(state.notice).not.toHaveBeenCalled()
})
it("does not perform confirmed replacement after its view retires", async () => {
  const gate = deferred(); const confirmation = vi.fn(() => gate.promise); const view = setup(confirmation)
  act(() => view.result.current.setImportMode("replace"))
  const pending = view.result.current.handleImportFile({ text: async () => JSON.stringify([{ title: "New", content: "ALICE" }]) } as File)
  await act(async () => { await Promise.resolve() }); expect(confirmation).toHaveBeenCalled(); view.unmount()
  await act(async () => { gate.resolve(true); await pending })
  expect(state.exports).not.toHaveBeenCalled(); expect(state.imports).not.toHaveBeenCalled()
})
