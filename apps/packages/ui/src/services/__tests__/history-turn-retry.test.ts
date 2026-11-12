import { beforeEach, expect, it, vi } from "vitest"
import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import type { HistoryTurnRecoveryEntry } from "@/services/history-turn-keep"
import { markHistoryTurnLive, resetHistoryTurnRegistry } from "@/services/history-turn-keep"
import { loadHistoryRecoveryRetryContext, retryHistoryTurnRecovery, reviewServerSavedQuestion } from "../history-turn-retry"
import { OUTPUT_FORMATTING_GUIDE_SYSTEM_PROMPT_SUFFIX } from "@/utils/output-formatting-guide"

const boundary = vi.hoisted(() => ({ load: vi.fn(), capture: vi.fn(), prompt: vi.fn(), defaultPrompt: vi.fn() }))
vi.mock("@/db/dexie/history-selection", async original => ({ ...await original<typeof import("@/db/dexie/history-selection")>(), loadHistoryTurnRecoveries: boundary.load }))
vi.mock("@/services/chat-history-selection", async original => ({ ...await original<typeof import("@/services/chat-history-selection")>(), captureHistorySnapshot: boundary.capture }))
vi.mock("@/db/dexie/helpers", async original => ({ ...await original<typeof import("@/db/dexie/helpers")>(), getPromptById: boundary.prompt }))
vi.mock("@/services/tldw-server", async original => ({ ...await original<typeof import("@/services/tldw-server")>(), systemPromptForNonRagOption: boundary.defaultPrompt }))

const fixture = () => {
  let epoch = 1
  let valid = true
  const owner = { kind: "native", owner_key: "alice", conversation_id: "saved-chat", validate_lease: () => valid }
  const view = { owner_key: "alice", conversation_id: "saved-chat", view_session_id: "reopened", selection_revision: 1,
    interpretation: { kind: "parent_graph_v1" }, cursor: { kind: "after_message", message_id: "original-user" } }
  const entry = { scope: { profile_id: "profile", client_session_id: "original-session" }, turn: {
    operation_id: "failed-send", persistence: "client", origin_view: { ...view, view_session_id: "original" },
    owner_key: "alice", conversation_id: "saved-chat", input_id: "original-user", assistant_id: "never-saved",
    created_at: 1, input_text: "Tell me about the weather today.", input_images: [], result_text: "", state: "accepted_unsent",
    selection_digest: "original-selection", request_context_digest: "old-pirate-request",
    admission: { version: 1, owner_key: "alice", conversation_id: "saved-chat", input_message_id: "original-user",
      input_message_revision: "1", selection_digest: "original-selection" }
  } } as HistoryTurnRecoveryEntry
  const question = { id: "original-user", revision: "sha256:accepted-user-state", role: "user", parent_id: null, settled: true }
  const capture = { status: "captured", validated_admission: { admission: structuredClone(entry.turn.admission), input_revision: question.revision }, rows: [question], selected_content: [{ id: question.id, revision: question.revision, message: entry.turn.input_text, images: [] }], snapshot: { nodes: [question] } }
  const current = { status: "ready", owner, view, bookmarkScope: { profile_id: "profile", client_session_id: "new-session" }, settingsQualified: true, capture }
  const choose = vi.fn(async cursor => { epoch += 1; current.view = { ...current.view, cursor, selection_revision: epoch }; return true })
  const controller = { getCurrent: () => current, fence: () => { const token = epoch; return () => token === epoch }, choose } as unknown as HistorySelectionController
  const review = { contextReviewed: true, context: { model: "qwen", systemPrompts: ["Answer like a pirate."], isCurrent: () => valid } }
  boundary.load.mockResolvedValue([structuredClone(entry)])
  boundary.capture.mockResolvedValue(capture)
  const submit = vi.fn(async () => ({ status: "submitted" as const }))
  return { entry, current, capture, controller, review, submit, choose, invalidate: () => { epoch += 1 }, revoke: () => { valid = false } }
}

beforeEach(() => { vi.clearAllMocks(); resetHistoryTurnRegistry(); boundary.prompt.mockResolvedValue(null); boundary.defaultPrompt.mockResolvedValue(undefined) })

it("keeps local semantic revision validation without a native proof", async () => {
  const f = fixture()
  Object.assign(f.current.owner, { kind: "local" })
  f.capture.rows[0].revision = f.entry.turn.admission!.input_message_revision
  f.capture.selected_content[0].revision = f.capture.rows[0].revision
  Object.assign(f.capture, { validated_admission: undefined })
  expect(await retryHistoryTurnRecovery(f.controller, f.entry, f.review, f.submit)).toEqual({ status: "submitted" })
  expect(boundary.capture.mock.calls[0][5]).toBeUndefined()
})

it("retries the exact accepted user with its admission and original recovery, without a new admission", async () => {
  const f = fixture()
  expect(await retryHistoryTurnRecovery(f.controller, f.entry, f.review, f.submit)).toEqual({ status: "submitted" })
  expect(f.choose).toHaveBeenCalledWith({ kind: "before_message", message_id: "original-user" })
  expect(f.submit).toHaveBeenCalledWith(expect.objectContaining({ message: f.entry.turn.input_text, image: "",
    historyRetryAdmission: f.entry.turn.admission, historyRetryRecoveries: [f.entry], historyRecoveryRetry: true,
    historyRecoveryRetrySystemPrompts: f.review.context.systemPrompts }))
})

it.each(["unreviewed", "server", "unknown", "no-admission", "partial-output", "outcome", "settled", "live", "profile", "owner", "conversation", "settings", "input-id", "input-revision", "input-text", "image", "answered", "record-session", "record-changed", "record-dismissed", "proof-missing", "proof-admission", "proof-revision"])(
  "refuses %s recovery without selecting another branch or dispatching", async refusal => {
    const f = fixture()
    if (refusal === "unreviewed") f.review.contextReviewed = false
    else if (refusal === "server") Object.assign(f.entry.turn, { persistence: "server" })
    else if (refusal === "unknown") f.entry.turn.state = "unknown"
    else if (refusal === "no-admission") f.entry.turn.admission = undefined
    else if (refusal === "partial-output") f.entry.turn.result_text = "partial"
    else if (refusal === "outcome") f.entry.turn.outcome = "complete"
    else if (refusal === "settled") f.entry.turn.settled_message_id = "saved-answer"
    else if (refusal === "live") markHistoryTurnLive(f.entry.turn.operation_id)
    else if (refusal === "profile") f.entry.scope.profile_id = "other-profile"
    else if (refusal === "owner") f.entry.turn.owner_key = "other-owner"
    else if (refusal === "conversation") f.entry.turn.conversation_id = "other-chat"
    else if (refusal === "settings") f.current.settingsQualified = false
    else if (refusal === "input-id") f.capture.rows[0].id = "other-user"
    else if (refusal === "input-revision") f.capture.rows[0].revision = "edited"
    else if (refusal === "input-text") f.capture.selected_content[0].message = "edited question"
    else if (refusal === "image") Object.assign(f.capture.selected_content[0], { images: ["changed-image"] })
    else if (refusal === "answered") f.capture.snapshot.nodes.push({ ...f.capture.rows[0], id: "saved-answer", role: "assistant", parent_id: "original-user" } as never)
    else if (refusal === "record-session") boundary.load.mockResolvedValue([{ ...f.entry, scope: { ...f.entry.scope, client_session_id: "other-session" } }])
    else if (refusal === "record-changed") boundary.load.mockResolvedValue([{ ...f.entry, turn: { ...f.entry.turn, input_text: "changed original" } }])
    else if (refusal === "proof-missing") Object.assign(f.capture, { validated_admission: undefined })
    else if (refusal === "proof-admission") f.capture.validated_admission.admission.selection_digest = "different receipt"
    else if (refusal === "proof-revision") f.capture.validated_admission.input_revision = "different capture"
    else boundary.load.mockResolvedValue([])
    expect((await retryHistoryTurnRecovery(f.controller, f.entry, f.review, f.submit)).status).toBe("failed")
    expect(f.choose).not.toHaveBeenCalled()
    expect(f.submit).not.toHaveBeenCalled()
  }
)

it.each(["record navigation", "capture navigation", "capture auth", "choose navigation", "context change"])(
  "refuses %s during recovery preparation and retains the original entry", async boundaryName => {
    const f = fixture()
    if (boundaryName === "record navigation") boundary.load.mockImplementation(async () => { f.invalidate(); return [f.entry] })
    else if (boundaryName === "capture navigation") boundary.capture.mockImplementation(async () => { f.invalidate(); return f.capture })
    else if (boundaryName === "capture auth") boundary.capture.mockImplementation(async () => { f.revoke(); return f.capture })
    else if (boundaryName === "choose navigation") f.choose.mockImplementation(async () => { f.invalidate(); return false })
    else f.review.context.isCurrent = () => false
    const original = structuredClone(f.entry)
    expect((await retryHistoryTurnRecovery(f.controller, f.entry, f.review, f.submit)).status).toBe("failed")
    expect(f.submit).not.toHaveBeenCalled()
    expect(f.entry).toEqual(original)
  }
)

it("refuses a prompt edited after review and a second simultaneous attempt", async () => {
  const f = fixture()
  const changed = async () => ({ ...f.review.context, systemPrompts: ["New instructions"] })
  expect(await retryHistoryTurnRecovery(f.controller, f.entry, f.review, f.submit, changed)).toMatchObject({ status: "failed", errorMessage: "retry_context_changed" })
  let release!: () => void
  boundary.load.mockImplementation(async () => { await new Promise<void>(resolve => { release = resolve }); return [f.entry] })
  const first = retryHistoryTurnRecovery(f.controller, f.entry, f.review, f.submit)
  expect(await retryHistoryTurnRecovery(f.controller, f.entry, f.review, f.submit)).toMatchObject({ status: "failed", errorMessage: "history_retry_in_progress" })
  release()
  await first
  expect(f.submit).toHaveBeenCalledOnce()
})

it("loads the reapplied saved Pirate instruction and refuses an unavailable selected prompt", async () => {
  boundary.prompt.mockResolvedValue({ id: "pirate", system_prompt: "Answer like a pirate." })
  expect(await loadHistoryRecoveryRetryContext({ model: "qwen", selectedSystemPrompt: "pirate" }, () => true)).toMatchObject({ model: "qwen", systemPrompts: ["Answer like a pirate."] })
  boundary.prompt.mockResolvedValue(null)
  await expect(loadHistoryRecoveryRetryContext({ model: "qwen", selectedSystemPrompt: "pirate" }, () => true)).rejects.toThrow("retry_prompt_unavailable")
})

it.each(["default", "selected", "inline", "blank default"])("review resolves %s instructions with normal whitespace and appendix rules", async branch => {
  boundary.defaultPrompt.mockResolvedValue(branch === "blank default" ? "" : "  Default instruction.\n ")
  if (branch === "selected" || branch === "inline")
    boundary.prompt.mockResolvedValue({ id: "pirate", system_prompt: "  Answer like a pirate.\n " })
  const source = { model: "qwen", selectedSystemPrompt: branch === "selected" || branch === "inline" ? "pirate" : "",
    systemPrompt: branch === "inline" ? "  Inline instruction.\n " : undefined,
    systemPromptAppendix: OUTPUT_FORMATTING_GUIDE_SYSTEM_PROMPT_SUFFIX }
  const base = branch === "default" ? "Default instruction." : branch === "selected" ? "Answer like a pirate." : branch === "inline" ? "Inline instruction." : ""
  expect((await loadHistoryRecoveryRetryContext(source, () => true)).systemPrompts)
    .toEqual([base ? `${base}\n\n${OUTPUT_FORMATTING_GUIDE_SYSTEM_PROMPT_SUFFIX}` : OUTPUT_FORMATTING_GUIDE_SYSTEM_PROMPT_SUFFIX])
})

it.each([false, true])("failed retry retains the saved question selection without moving a new view (navigation: %s)", async navigation => {
  const f = fixture()
  const original = structuredClone(f.entry)
  const submit = vi.fn(async () => {
    if (navigation) f.invalidate()
    return { status: "failed" as const, errorMessage: "provider unavailable" }
  })
  expect(await retryHistoryTurnRecovery(f.controller, f.entry, f.review, submit)).toMatchObject({ status: "failed", errorMessage: "provider unavailable" })
  expect(f.entry).toEqual(original)
  expect(f.choose).toHaveBeenCalledTimes(navigation ? 1 : 2)
  if (!navigation) expect(f.current.view.cursor).toEqual({ kind: "after_message", message_id: "original-user" })
})

it("a held consumer send cannot report successful recovery or hide its saved question", async () => {
  const f = fixture()
  const submit = vi.fn(async () => undefined)
  expect(await retryHistoryTurnRecovery(f.controller, f.entry, f.review, submit)).toMatchObject({ status: "failed", errorMessage: "history_recovery_retry_not_submitted" })
  expect(f.current.view.cursor).toEqual({ kind: "after_message", message_id: "original-user" })
})


it.each(["ready", "navigation", "account", "answered", "asset"])("B02 saved-question review binds canonical input (%s)", async mode => {
  const f = fixture()
  if (mode === "answered") f.capture.snapshot.nodes.push({id: "answer", role: "assistant", parent_id: "original-user"} as any)
  if (mode === "asset") f.capture.selected_content[0].images = ["image"] as never[]
  boundary.capture.mockImplementation(async () => {
    if (mode === "navigation") f.invalidate()
    if (mode === "account") f.revoke()
    return f.capture
  })
  if (mode === "ready") {
    const review = await reviewServerSavedQuestion(f.controller, f.review.context)
    expect(review).toMatchObject({id: "original-user", revision: f.capture.rows[0].revision, text: f.entry.turn.input_text, model: "qwen"})
    expect(review.isCurrent()).toBe(true)
    f.invalidate()
    expect(review.isCurrent()).toBe(false)
  } else await expect(reviewServerSavedQuestion(f.controller, f.review.context)).rejects.toThrow()
  expect(boundary.capture.mock.calls[0][4]).toBe("tldw_history_server_existing_input_v1")
  expect(f.choose).not.toHaveBeenCalled()
  expect(f.submit).not.toHaveBeenCalled()
  expect(boundary.load).not.toHaveBeenCalled()
})
