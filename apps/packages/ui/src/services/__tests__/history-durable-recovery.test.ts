import { beforeEach, describe, expect, it, vi } from "vitest"
import * as wire from "../history-durable-turn"
import type { HistoryTurnRecovery } from "@/db/dexie/types"
import type { NativeHistoryOwnerV1 } from "../chat-history-selection"
import { selectionDigest } from "@/utils/history-selection"

const calls = vi.hoisted(() => ({ request: vi.fn() }))
vi.mock("@/services/background-proxy", () => ({ bgRequest: (...args: unknown[]) => calls.request(...args) }))
const input = "12345678-1234-4321-8123-123456789abc"
const result = "22345678-1234-4321-8123-123456789abc"
const owner: NativeHistoryOwnerV1 = { kind: "native", owner_key: "native-key", conversation_id: "chat",
  scope: { type: "workspace", workspaceId: "space" }, validate_lease: () => true,
  request_scope: { config: { serverUrl: "https://pinned.example", authMode: "single-user" }, userId: 7 } }
const selection = { version: 1 as const, owner_key: "native-key", conversation_id: "chat",
  interpretation: { kind: "parent_graph_v1" as const }, cursor: { kind: "empty" as const }, selection_revision: 1,
  purpose: "send" as const, messages: [], fences: { conversation: "1", history: "1", settings: "1" },
  storage_context_digest: "storage", request_context_digest: "a".repeat(64), selection_digest: "" }
const finalized = { ...selection, selection_digest: selectionDigest(selection) }
const admission = { version: 1 as const, owner_key: "native-key", conversation_id: "chat", input_message_id: input,
  input_message_revision: "1", selection_digest: finalized.selection_digest }
const fullAdmission = { ...admission, messages: [], originating_selection_revision: 1 }
const observed = { version: 1 as const, result_message_id: result, result_message_revision: "1" as const,
  admission, request_context_digest: "b".repeat(64), sources: [] }
const turn = (): Extract<HistoryTurnRecovery, { persistence: "server" }> => ({ persistence: "server", operation_id: "operation", owner_key: "native-key",
  conversation_id: "chat", origin_view: { ...selection, view_session_id: "view" },
  selection_digest: finalized.selection_digest, request_context_digest: "b".repeat(64), created_at: 1,
  input_text: " Original\n", input_images: [], result_text: "Answer", state: "unknown",
  logical_user_message_id: input, finalized_selection: finalized })
const scope = { scope_type: "workspace", workspace_id: "space" }
const inputRead = () => ({ id: input, conversation_id: "chat", sender: "user", content: " Original\n", version: 1,
  tldw_history_recovery_v1: { version: 1, status: "input_verified", scope, admission: fullAdmission } })
const resultRead = () => ({ id: result, conversation_id: "chat", sender: "assistant", content: "Answer", version: 1,
  parent_message_id: input, tldw_history_recovery_v1: { version: 1, status: "result_verified", scope, result: observed } })
beforeEach(() => {
  calls.request.mockReset()
  expect(wire.inspectHistoryDurableRecovery, "protected exact recovery helper is missing").toBeTypeOf("function")
})
describe("exact protected durable recovery", () => {
  it("reads only the logical UUID, pinned to account and workspace, and installs the protected input reference", async () => {
    calls.request.mockResolvedValue(inputRead())
    const value = await wire.inspectHistoryDurableRecovery(owner, turn())
    expect(value.admission).toEqual(admission)
    expect(value.input_id).toBe(input)
    expect(value.assistant_id).toBeUndefined()
    expect(calls.request).toHaveBeenCalledTimes(1)
    expect(calls.request.mock.calls[0][0]).toMatchObject({ method: "GET",
      path: `/api/v1/messages/${input}?scope_type=workspace&workspace_id=space&include_history_recovery_v1=true`,
      headers: { "X-TLDW-Expected-User-ID": "7" }, servicePromptConfig: { expectedUserId: 7 } })
  })
  it("reads only an already observed result and never lists siblings", async () => {
    calls.request.mockResolvedValue(resultRead())
    const value = await wire.inspectHistoryDurableRecovery(owner, { ...turn(), admission, input_id: input, observed_result: observed })
    expect(value).toMatchObject({ observed_result: observed })
    expect(value.assistant_id).toBe(result)
    expect(calls.request).toHaveBeenCalledTimes(1)
    expect(calls.request.mock.calls[0][0].path).toContain(`/messages/${result}?`)
  })
  it("first verifies input if an observed result exists without an admission", async () => {
    calls.request.mockResolvedValueOnce(inputRead()).mockResolvedValueOnce(resultRead())
    const value = await wire.inspectHistoryDurableRecovery(owner, { ...turn(), observed_result: observed })
    expect(value.assistant_id).toBe(result)
    expect(calls.request.mock.calls.map(call => call[0].path.split("?")[0])).toEqual([
      `/api/v1/messages/${input}`, `/api/v1/messages/${result}`
    ])
  })
  it.each([
    { tldw_history_recovery_v1: undefined },
    { tldw_history_recovery_v1: { version: 1, status: "unverified", code: "no_protected_binding" } },
    { id: result }, { content: "Original" }, { version: 2 }, { sender: "assistant" }, { conversation_id: "other" },
    { tldw_history_recovery_v1: { ...inputRead().tldw_history_recovery_v1, scope: { scope_type: "global", workspace_id: null } } },
    { tldw_history_recovery_v1: { ...inputRead().tldw_history_recovery_v1, admission: { ...fullAdmission, originating_selection_revision: 2 } } },
    { tldw_history_recovery_v1: { ...inputRead().tldw_history_recovery_v1, admission: { ...fullAdmission, messages: [{ id: "other", revision: "1" }] } } },
    { tldw_history_recovery_v1: { ...inputRead().tldw_history_recovery_v1, headers: {} } }
  ])("rejects missing, mismatched or unverified protected input %#", async changed => {
    calls.request.mockResolvedValue({ ...inputRead(), ...changed })
    await expect(wire.inspectHistoryDurableRecovery(owner, turn())).rejects.toThrow()
  })
  it.each([
    { id: input }, { parent_message_id: result }, { sender: "user" }, { version: 2 },
    { tldw_history_recovery_v1: { ...resultRead().tldw_history_recovery_v1, result: { ...observed, request_context_digest: "c".repeat(64) } } },
    { tldw_history_recovery_v1: { ...resultRead().tldw_history_recovery_v1, result: { ...observed, result_message_id: "32345678-1234-4321-8123-123456789abc" } } }
  ])("rejects wrong known-result bindings %#", async changed => {
    calls.request.mockResolvedValue({ ...resultRead(), ...changed })
    await expect(wire.inspectHistoryDurableRecovery(owner, { ...turn(), admission, input_id: input, observed_result: observed })).rejects.toThrow()
  })
  it("rejects stale leases before any request and after a response", async () => {
    const lease = vi.fn().mockReturnValue(false)
    await expect(wire.inspectHistoryDurableRecovery({ ...owner, validate_lease: lease }, turn())).rejects.toThrow()
    expect(calls.request).not.toHaveBeenCalled()
    lease.mockReturnValueOnce(true).mockReturnValue(false)
    calls.request.mockResolvedValue(inputRead())
    await expect(wire.inspectHistoryDurableRecovery({ ...owner, validate_lease: lease }, turn())).rejects.toThrow()
  })
  it("rejects an unbound local record and never adopts public admission metadata", async () => {
    const value = { ...turn(), persistence: "client", input_id: input, assistant_id: result } as HistoryTurnRecovery
    await expect(wire.inspectHistoryDurableRecovery(owner, value)).rejects.toThrow()
    expect(calls.request).not.toHaveBeenCalled()
  })
  it("rejects a conflicting stored input ID even before an admission is observed", async () => {
    calls.request.mockResolvedValue(inputRead())
    await expect(wire.inspectHistoryDurableRecovery(owner, { ...turn(), input_id: result })).rejects.toThrow()
    expect(calls.request).not.toHaveBeenCalled()
  })
})
