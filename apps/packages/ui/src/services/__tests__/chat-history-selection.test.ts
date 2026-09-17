import { beforeEach, describe, expect, it, vi } from "vitest"
const calls = vi.hoisted(() => ({
  capture: vi.fn(),
  confirm: vi.fn(),
  append: vi.fn(),
  pending: vi.fn(),
  ack: vi.fn(),
  reject: vi.fn(),
  dispatched: vi.fn()
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    captureHistorySelection: calls.capture,
    confirmHistoryProjection: calls.confirm,
    addChatMessage: calls.append
  }
}))
vi.mock("@/db/dexie/history-selection", async (importOriginal) => ({
  ...(await importOriginal<any>()),
  savePendingHistoryConfirmation: calls.pending,
  acknowledgeHistoryConfirmation: calls.ack,
  rejectPendingHistoryConfirmation: calls.reject,
  markPendingHistoryConfirmationDispatched: calls.dispatched
}))
vi.mock("@/db/dexie/helpers", () => ({ generateID: () => "new-id" }))
import * as service from "../chat-history-selection"
import { selectionDigest } from "@/utils/history-selection"
const owner = () => ({
  kind: "native" as const,
  conversation_id: "chat",
  request_scope: {
    config: {
      serverUrl: "https://server.test",
      authMode: "multi-user" as const
    },
    userId: "alice"
  },
  validate_lease: () => true
})
const view = {
  owner_key: "native-key",
  conversation_id: "chat",
  view_session_id: "view",
  cursor: { kind: "empty" as const },
  interpretation: { kind: "parent_graph_v1" as const },
  selection_revision: 1
}
const capture = () => ({
  status: "captured",
  snapshot: {
    version: 1,
    owner_key: "native-key",
    conversation_id: "chat",
    fences: { conversation: "1", history: "1", settings: "1" },
    nodes: [],
    source_digest: "source",
    interpretation_status: { kind: "parent_graph_v1" },
    storage_context_digest: "storage"
  },
  rows: [],
  selected_content: [],
  view,
  purpose: "send",
  storage_context_digest: "storage"
})
beforeEach(() => {
  vi.clearAllMocks()
  calls.pending.mockResolvedValue("view")
  calls.capture.mockResolvedValue(capture())
})
describe("history owner service", () => {
  it("returns capture before composition and freezes the exact explicit payload", async () => {
    const native = owner()
    const captured = await service.captureHistorySnapshot(native, view, "send")
    if (captured.status !== "captured") throw Error(captured.status)
    const payload = { messages: [{ content: "new input" }], model: "chosen" }
    const prepared = service.prepareHistoryContext(payload, () => true)
    payload.messages[0].content = "mutated"
    const finalized = service.finalizeHistorySelection(
      native,
      captured,
      prepared,
      view
    )
    expect((prepared.payload as any).messages[0].content).toBe("new input")
    expect(Object.isFrozen(prepared.payload)).toBe(true)
    expect(finalized.selection_digest).toBe(selectionDigest(finalized))
    expect(calls.append).not.toHaveBeenCalled()
  })
  it.each(["old", "content", "owner", "view", "rows"])(
    "rejects malformed/mismatched %s responses",
    async (kind) => {
      const response: any = capture()
      if (kind === "old") response.snapshot.version = 0
      if (kind === "content")
        response.selected_content = [
          { id: "invented", revision: "x", message: "x", images: [] }
        ]
      if (kind === "owner") response.snapshot.owner_key = "different"
      if (kind === "view") response.view = { ...view, selection_revision: 3 }
      if (kind === "rows") response.rows = [{ id: "invented", revision: "x" }]
      calls.capture.mockResolvedValue(response)
      await expect(
        service.captureHistorySnapshot(owner(), view, "send")
      ).rejects.toThrow()
      expect(calls.append).not.toHaveBeenCalled()
    }
  )
  it.each(["captured", "legacy_review_required"])(
    "checks lease after %s response",
    async (status) => {
      const native = owner()
      let current = true
      native.validate_lease = () => current
      const response: any = capture()
      if (status !== "captured") {
        delete response.rows
        delete response.selected_content
        delete response.purpose
        delete response.storage_context_digest
        response.status = status
        response.code = status
      }
      calls.capture.mockImplementation(async () => {
        current = false
        return response
      })
      await expect(
        service.captureHistorySnapshot(native, view, "send")
      ).rejects.toMatchObject({ code: "request_config_scope_changed" })
    }
  )
  it("rejects navigation and changed prepared payload before finalization", async () => {
    const native = owner()
    const captured = await service.captureHistorySnapshot(native, view, "send")
    if (captured.status !== "captured") throw Error(captured.status)
    const prepared = service.prepareHistoryContext(
      { model: "chosen" },
      () => true
    )
    expect(() =>
      service.finalizeHistorySelection(native, captured, prepared, {
        ...view,
        selection_revision: 2
      })
    ).toThrow("stale_selection")
    expect(() =>
      service.finalizeHistorySelection(
        native,
        captured,
        { ...prepared, payload: { model: "other" } },
        view
      )
    ).toThrow("request_context_mismatch")
  })
  it("requires capability handshake and rejects loss of multiple images before append", async () => {
    const native = owner()
    const message = {
      id: "u",
      history_id: "chat",
      role: "user",
      name: "You",
      content: "new",
      images: ["data:image/png;base64,YQ==", "data:image/png;base64,Yg=="],
      createdAt: 1
    }
    await expect(
      service.appendSelectedUser(native, {} as any, message)
    ).rejects.toMatchObject({ code: "unsupported_history_capability" })
    const captured = await service.captureHistorySnapshot(native, view, "send")
    if (captured.status !== "captured") throw Error(captured.status)
    const selected = service.finalizeHistorySelection(
      native,
      captured,
      service.prepareHistoryContext({ model: "m" }, () => true),
      view
    )
    await expect(
      service.appendSelectedUser(native, selected, message)
    ).rejects.toMatchObject({ code: "unsupported_message_payload" })
    expect(calls.append).not.toHaveBeenCalled()
  })
  it("preserves opaque native input revision and sends only an admission reference for settlement", async () => {
    const native = owner()
    const captured = await service.captureHistorySnapshot(native, view, "send")
    if (captured.status !== "captured") throw Error(captured.status)
    const selected = service.finalizeHistorySelection(
      native,
      captured,
      service.prepareHistoryContext({ model: "m" }, () => true),
      view
    )
    const admission = {
      version: 1,
      owner_key: view.owner_key,
      conversation_id: "chat",
      input_message_id: "u",
      input_message_revision: "native-row-version-1",
      selection_digest: selected.selection_digest,
      messages: [],
      originating_selection_revision: 1
    }
    calls.append.mockResolvedValueOnce({
      id: "u",
      tldw_history_admission_v1: admission
    })
    const accepted = await service.appendSelectedUser(native, selected, {
      id: "u",
      history_id: "chat",
      role: "user",
      name: "You",
      content: "hello",
      createdAt: 1
    })
    calls.append.mockResolvedValueOnce({ id: "reply" })
    await service.settleAcceptedAssistant(native, accepted, {
      id: "reply",
      history_id: "chat",
      role: "assistant",
      name: "Assistant",
      content: "answer",
      createdAt: 1
    })
    expect(calls.append.mock.calls[1][1].tldw_history_admission_v1).toEqual({
      version: 1,
      owner_key: view.owner_key,
      conversation_id: "chat",
      input_message_id: "u",
      input_message_revision: "native-row-version-1",
      selection_digest: selected.selection_digest
    })
  })
  it("persists pending projection before dispatch and bookmark only after acknowledgement", async () => {
    const native = owner()
    await service.captureHistorySnapshot(native, view, "send")
    const confirmation = {
      version: 1 as const,
      owner_key: "native-key",
      conversation_id: "chat",
      projection_id: "p",
      source_digest: "source",
      fences: capture().snapshot.fences,
      source_members: [],
      ordered_path_ids: [],
      cursor: view.cursor,
      selection_revision: 1
    }
    calls.confirm.mockImplementation(async () => {
      expect(calls.pending).toHaveBeenCalled()
      expect(calls.ack).not.toHaveBeenCalled()
      return {
        ...confirmation,
        projection_digest: "digest",
        created_at: "date"
      }
    })
    await service.confirmLegacyHistoryProjection(
      native,
      { profile_id: "profile", client_session_id: "session" },
      confirmation,
      view
    )
    expect(calls.ack).toHaveBeenCalledOnce()
  })
})

describe("lossless native message capability", () => {
  it("does not invent an ignored generic metadata field", () => {
    expect(() =>
      service.nativeHistoryMessagePayload({
        id: "u",
        history_id: "chat",
        role: "user",
        name: "You",
        content: "hello",
        createdAt: 1,
        metadataExtra: { tool_call_id: "tool" }
      })
    ).toThrow("unsupported_message_payload")
  })
  it("represents image-only input without an invalid empty content string", () => {
    expect(
      service.nativeHistoryMessagePayload({
        id: "u",
        history_id: "chat",
        role: "user",
        name: "You",
        content: "",
        images: [
          "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aJp0AAAAASUVORK5CYII="
        ],
        createdAt: 1
      })
    ).not.toHaveProperty("content")
  })
  it("rejects empty input before owner append", () => {
    expect(() =>
      service.nativeHistoryMessagePayload({
        id: "u",
        history_id: "chat",
        role: "user",
        name: "You",
        content: "",
        createdAt: 1
      })
    ).toThrow("unsupported_message_payload")
  })
})

it("temporary/unavailable owners cannot bootstrap storage or dispatch", async () => {
  await expect(
    service.captureHistorySnapshot(
      { kind: "unavailable", code: "temporary_owner_unavailable" },
      view,
      "send"
    )
  ).rejects.toMatchObject({ code: "temporary_owner_unavailable" })
  expect(calls.capture).not.toHaveBeenCalled()
  expect(calls.pending).not.toHaveBeenCalled()
  expect(calls.append).not.toHaveBeenCalled()
})

it("rejects unsupported image bytes and a declared MIME that would change at persistence", () => {
  const base = {
    id: "u",
    history_id: "chat",
    role: "user",
    name: "You",
    content: "image",
    createdAt: 1
  }
  expect(() =>
    service.nativeHistoryMessagePayload({
      ...base,
      images: ["data:image/png;base64,YQ=="]
    })
  ).toThrow("unsupported_message_payload")
  const png =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aJp0AAAAASUVORK5CYII="
  expect(() =>
    service.nativeHistoryMessagePayload({
      ...base,
      images: [`data:image/jpeg;base64,${png}`]
    })
  ).toThrow("unsupported_message_payload")
  expect(
    service.nativeHistoryMessagePayload({
      ...base,
      images: [`data:image/png;base64,${png}`]
    })
  ).toHaveProperty("image_base64", `data:image/png;base64,${png}`)
})

it("retains the accepted native intent if the mutable caller row changes during append", async () => {
  const native = owner()
  const captured = await service.captureHistorySnapshot(native, view, "send")
  if (captured.status !== "captured") throw Error(captured.status)
  const chosen = service.finalizeHistorySelection(
    native,
    captured,
    service.prepareHistoryContext({ model: "m" }, () => true),
    view
  )
  const input = {
    id: "u",
    history_id: "chat",
    role: "user",
    name: "You",
    content: "hello",
    createdAt: 1
  }
  const admission = {
    version: 1 as const,
    owner_key: view.owner_key,
    conversation_id: "chat",
    input_message_id: "u",
    input_message_revision: "1",
    selection_digest: chosen.selection_digest,
    messages: [],
    originating_selection_revision: 1
  }
  calls.append.mockImplementation(async () => {
    input.id = "changed"
    return { id: "u", tldw_history_admission_v1: admission }
  })
  await expect(
    service.appendSelectedUser(native, chosen, input)
  ).resolves.toEqual(admission)
})

it("does not settle through a bound owner that has not proved the versioned endpoint", async () => {
  const native = { ...owner(), owner_key: "native-key" }
  const reference = {
    version: 1 as const,
    owner_key: "native-key",
    conversation_id: "chat",
    input_message_id: "u",
    input_message_revision: "1",
    selection_digest: "digest"
  }
  calls.append.mockResolvedValue({ id: "reply" })
  await expect(
    service.settleAcceptedAssistant(native, reference, {
      id: "reply",
      history_id: "chat",
      role: "assistant",
      name: "Assistant",
      content: "reply",
      createdAt: 1
    })
  ).rejects.toMatchObject({ code: "unsupported_history_capability" })
  expect(calls.append).not.toHaveBeenCalled()
})

it("normalizes singular images and empty placeholders before generic native preflight", () => {
  const image =
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aJp0AAAAASUVORK5CYII="
  expect(
    service.nativeHistoryMessagePayload({
      id: "u",
      history_id: "chat",
      role: "user",
      name: "You",
      content: "hello",
      createdAt: 1,
      image,
      images: [""]
    })
  ).toHaveProperty("image_base64", image)
})

it("retains a proved capability across later non-ready source drift but does not bootstrap writes from failure alone", async () => {
  const native = owner()
  await service.captureHistorySnapshot(native, view, "send")
  calls.capture.mockResolvedValue({
    status: "stale_selection",
    code: "stale_projection",
    snapshot: capture().snapshot,
    view
  })
  expect(
    (await service.captureHistorySnapshot(native, view, "send")).status
  ).toBe("stale_selection")
  const reference = {
    version: 1 as const,
    owner_key: "native-key",
    conversation_id: "chat",
    input_message_id: "u",
    input_message_revision: "1",
    selection_digest: "digest"
  }
  const result = {
    id: "reply",
    history_id: "chat",
    role: "assistant",
    name: "Assistant",
    content: "reply",
    createdAt: 1
  }
  calls.append.mockResolvedValue({ id: "reply" })
  await expect(
    service.settleAcceptedAssistant(native, reference, result)
  ).resolves.toHaveProperty("id", "reply")
  const reopened = owner()
  await service.captureHistorySnapshot(reopened, view, "send")
  await expect(
    service.settleAcceptedAssistant(reopened, reference, result)
  ).rejects.toMatchObject({ code: "unsupported_history_capability" })
  expect(calls.append).toHaveBeenCalledTimes(1)
})

it.each([
  { messageType: "tool" },
  { reasoning_time_taken: 0 },
  { reasoning_time_taken: 12 },
  { modelName: "model" },
  { modelImage: "https://image.test/model.png" }
])(
  "rejects metadata the generic native route cannot retain: %j",
  (metadata) => {
    expect(() =>
      service.nativeHistoryMessagePayload({
        id: "reply",
        history_id: "chat",
        name: "Assistant",
        role: "assistant",
        content: "Reply",
        createdAt: 1,
        ...metadata
      })
    ).toThrow("unsupported_message_payload")
  }
)
it("clears pending when the lease expires after persistence but before dispatch", async () => {
  const native = owner()
  await service.captureHistorySnapshot(native, view, "send")
  calls.pending.mockImplementationOnce(async () => {
    native.validate_lease = () => false
    return "view"
  })
  const intent = {
    version: 1 as const,
    owner_key: view.owner_key,
    conversation_id: "chat",
    projection_id: "p",
    source_digest: "source",
    fences: capture().snapshot.fences,
    source_members: [],
    ordered_path_ids: [],
    cursor: view.cursor,
    selection_revision: 1
  }
  await expect(
    service.confirmLegacyHistoryProjection(
      native,
      { profile_id: "profile", client_session_id: "session" },
      intent,
      view
    )
  ).rejects.toMatchObject({ code: "request_config_scope_changed" })
  expect(calls.confirm).not.toHaveBeenCalled()
  expect(calls.reject).toHaveBeenCalledWith(
    { profile_id: "profile", client_session_id: "session" },
    view,
    intent,
    { onlyIfUndispatched: true }
  )
})

it.each(["lease", "parse", "ack"])(
  "retains pending when post-dispatch %s validation fails",
  async (failure) => {
    const native = owner()
    await service.captureHistorySnapshot(native, view, "send")
    const intent = {
      version: 1 as const,
      owner_key: view.owner_key,
      conversation_id: "chat",
      projection_id: "p",
      source_digest: "source",
      fences: capture().snapshot.fences,
      source_members: [],
      ordered_path_ids: [],
      cursor: view.cursor,
      selection_revision: 1
    }
    calls.confirm.mockImplementationOnce(async () => {
      if (failure === "lease") native.validate_lease = () => false
      return failure === "parse"
        ? {}
        : { ...intent, projection_digest: "digest", created_at: "now" }
    })
    if (failure === "ack")
      calls.ack.mockRejectedValueOnce(new Error("storage failed"))
    await expect(
      service.confirmLegacyHistoryProjection(
        native,
        { profile_id: "profile", client_session_id: "session" },
        intent,
        view
      )
    ).rejects.toThrow()
    expect(calls.confirm).toHaveBeenCalledOnce()
    expect(calls.reject).not.toHaveBeenCalled()
  }
)
it("normalizes absent empty display defaults on the generic native route", () => {
  expect(
    service.nativeHistoryMessagePayload({
      id: "reply",
      history_id: "chat",
      name: "Assistant",
      role: "assistant",
      content: "Reply",
      createdAt: 1,
      messageType: "",
      modelName: "",
      modelImage: ""
    })
  ).toEqual({ id: "reply", role: "assistant", content: "Reply" })
})
