import { beforeEach, describe, expect, it, vi } from "vitest"
import type { HistorySelectionCaptureV1 } from "@/types/history-selection"
const io = vi.hoisted(() => ({
  capture: vi.fn(),
  read: vi.fn(),
  write: vi.fn()
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    captureHistorySelection: io.capture,
    getMessage: io.read,
    editMessage: io.write
  }
}))
import * as service from "../chat-history-selection"

const original = "Original {{char}}  \n"
const edited = "Edited {{user}}  \n"
const owner = () => ({
  kind: "native" as const,
  conversation_id: "chat",
  owner_key: "owner",
  scope: { type: "workspace" as const, workspaceId: "original" },
  request_scope: {
    config: {
      serverUrl: "https://original.test",
      authMode: "multi-user" as const
    },
    userId: "alice"
  },
  validate_lease: () => true
})
const capture = (version = 1, text = original): HistorySelectionCaptureV1 => {
  const nodes = [
    {
      id: "u",
      conversation_id: "chat",
      revision: `opaque-target-${version}`,
      parent_id: null,
      role: "user",
      settled: true,
      metadata: [{ id: "u", kind: "message_metadata", revision: "metadata" }],
      assets: []
    },
    {
      id: "a",
      conversation_id: "chat",
      revision: "1",
      parent_id: "u",
      role: "assistant",
      settled: true
    },
    {
      id: "sibling",
      conversation_id: "chat",
      revision: "1",
      parent_id: "u",
      role: "assistant",
      settled: true
    }
  ]
  const view = {
    owner_key: "owner",
    conversation_id: "chat",
    view_session_id: "view",
    selection_revision: 1,
    interpretation: { kind: "parent_graph_v1" as const },
    cursor: { kind: "after_message" as const, message_id: "a" }
  }
  return {
    status: "captured",
    view,
    purpose: "send",
    storage_context_digest: "storage",
    snapshot: {
      version: 1,
      owner_key: "owner",
      conversation_id: "chat",
      source_digest: "source",
      storage_context_digest: "storage",
      fences: { conversation: "1", history: String(version), settings: "1" },
      interpretation_status: view.interpretation,
      nodes
    },
    rows: nodes.slice(0, 2),
    selected_content: [
      {
        id: "u",
        revision: `opaque-target-${version}`,
        message: text,
        images: [],
        extra_metadata: {
          sender_role: "user",
          pinned: true,
          nested: { value: 1 }
        }
      },
      { id: "a", revision: "1", message: "Answer", images: [] }
    ]
  }
}
const response = (version = 1, text = original) => ({
  id: "u",
  conversation_id: "chat",
  sender: "user",
  content: text,
  parent_message_id: null,
  has_image: false,
  version
})
const save = (
  native: ReturnType<typeof owner>,
  source: HistorySelectionCaptureV1,
  options = {}
) =>
  service.saveNativeHistoryMessage(
    native,
    source,
    { id: "u", message: original },
    edited,
    options
  )
beforeEach(() => {
  vi.resetAllMocks()
  io.capture.mockResolvedValue(capture())
  io.read.mockResolvedValue(response())
  // PUT expands placeholders for display; canonical capture is the raw acknowledgement.
  io.write.mockResolvedValue(response(2, "Edited Alice  \n"))
})
describe("native plaintext Save authority", () => {
  it("uses the original numeric version for one scoped content-only CAS", async () => {
    const native = owner(),
      signal = new AbortController().signal
    const source = (await service.captureHistorySnapshot(
      native,
      capture().view,
      "send"
    )) as HistorySelectionCaptureV1
    io.capture.mockClear()
    const receipt = await save(native, source, {
      signal,
      validate_lease: () => true
    })
    expect(receipt).toEqual({ message_id: "u", message: edited, version: 2 })
    expect(io.read).toHaveBeenCalledExactlyOnceWith("u", {
      requestScope: native.request_scope,
      scope: native.scope,
      signal
    })
    expect(io.capture).toHaveBeenCalledExactlyOnceWith(
      "chat",
      { view: source.view, purpose: "send" },
      { requestScope: native.request_scope, scope: native.scope, signal }
    )
    expect(io.write).toHaveBeenCalledExactlyOnceWith("u", edited, 1, "chat", {
      requestScope: native.request_scope,
      scope: native.scope,
      signal
    })
  })
  it.each([
    { version: 0 },
    { version: "1" },
    { version: Number.MAX_SAFE_INTEGER },
    { id: "foreign" },
    { conversation_id: "foreign" },
    { sender: "assistant" },
    { content: "changed" },
    { parent_message_id: "foreign" },
    { has_image: true }
  ])("rejects a cold target mismatch before writing %j", async (patch) => {
    const native = owner()
    const source = (await service.captureHistorySnapshot(
      native,
      capture().view,
      "send"
    )) as HistorySelectionCaptureV1
    io.read.mockResolvedValue({ ...response(), ...patch })
    await expect(save(native, source)).rejects.toThrow()
    expect(io.write).not.toHaveBeenCalled()
  })
  it.each(["revision", "metadata", "content"])(
    "a fresh %s change cannot adopt a newer target version",
    async (change) => {
      const native = owner()
      const source = (await service.captureHistorySnapshot(
        native,
        capture().view,
        "send"
      )) as HistorySelectionCaptureV1
      const fresh = capture()
      if (change === "revision") {
        fresh.snapshot.nodes[0].revision = "other-revision"
        fresh.rows[0].revision = "other-revision"
        fresh.selected_content[0].revision = "other-revision"
      } else if (change === "metadata")
        fresh.snapshot.nodes[0].metadata![0].revision = "changed"
      else fresh.selected_content[0].message = "changed"
      io.capture.mockResolvedValue(fresh)
      await expect(save(native, source)).rejects.toThrow()
      expect(io.write).not.toHaveBeenCalled()
    }
  )
  it.each([
    null,
    {},
    { ...response(2), version: 1 },
    { ...response(2), version: "2" },
    { ...response(2), id: "foreign" },
    { ...response(2), conversation_id: "foreign" },
    { ...response(2), sender: "assistant" },
    { ...response(2), parent_message_id: "foreign" }
  ])(
    "a malformed or foreign PUT response cannot acknowledge Save %j",
    async (ack) => {
      const native = owner()
      const source = (await service.captureHistorySnapshot(
        native,
        capture().view,
        "send"
      )) as HistorySelectionCaptureV1
      io.write.mockResolvedValue(ack)
      await expect(save(native, source)).rejects.toThrow()
      expect(io.write).toHaveBeenCalledOnce()
    }
  )
  it.each(["read", "capture", "write"])(
    "rechecks source retirement after held %s",
    async (stage) => {
      const native = owner(),
        source = (await service.captureHistorySnapshot(
          native,
          capture().view,
          "send"
        )) as HistorySelectionCaptureV1
      let current = true
      const boundary =
        stage === "read" ? io.read : stage === "write" ? io.write : io.capture
      boundary.mockImplementationOnce(async () => {
        current = false
        return stage === "capture"
          ? capture()
          : response(stage === "write" ? 2 : 1)
      })
      await expect(
        save(native, source, { validate_lease: () => current })
      ).rejects.toThrow()
      expect(io.write).toHaveBeenCalledTimes(stage === "write" ? 1 : 0)
    }
  )
  it("preserves a conflict without replacing the expected version or retrying", async () => {
    const native = owner(),
      source = (await service.captureHistorySnapshot(
        native,
        capture().view,
        "send"
      )) as HistorySelectionCaptureV1
    const conflict = Object.assign(new Error("version conflict"), {
      status: 409
    })
    io.write.mockRejectedValue(conflict)
    await expect(save(native, source)).rejects.toBe(conflict)
    expect(io.write).toHaveBeenCalledOnce()
  })
  it("qualifies saved raw content, target revision and unchanged parent/metadata before installation", () => {
    const source = capture(),
      fresh = capture(2, edited)
    expect(() =>
      service.qualifyNativeMessageEditCapture(source, fresh, {
        message_id: "u",
        message: edited,
        version: 2
      })
    ).not.toThrow()
  })
  it.each(["content", "revision", "parent", "metadata", "extra"])(
    "a post-write %s mismatch stays unacknowledged",
    (change) => {
      const source = capture(),
        fresh = capture(2, edited)
      if (change === "content")
        fresh.selected_content[0].message = "display-only text"
      if (change === "revision") {
        fresh.rows[0].revision = source.rows[0].revision
        fresh.snapshot.nodes[0].revision = source.rows[0].revision
        fresh.selected_content[0].revision = source.rows[0].revision
      }
      if (change === "parent") fresh.rows[0].parent_id = "foreign"
      if (change === "metadata") fresh.rows[0].metadata![0].revision = "changed"
      if (change === "extra")
        fresh.selected_content[0].extra_metadata = { pinned: false }
      expect(() =>
        service.qualifyNativeMessageEditCapture(source, fresh, {
          message_id: "u",
          message: edited,
          version: 2
        })
      ).toThrow()
    }
  )
})

it("native Save refuses a reviewed legacy projection before reading or writing", async () => {
  const native = owner(),
    legacy = capture()
  legacy.view.interpretation = {
    kind: "legacy_linear_v1",
    projection_id: "reviewed"
  }
  legacy.snapshot.interpretation_status = {
    kind: "legacy_linear_v1",
    projection_id: "reviewed",
    ordered_path_ids: ["u", "a"]
  }
  io.capture.mockResolvedValue(legacy)
  const source = (await service.captureHistorySnapshot(
    native,
    legacy.view,
    "send"
  )) as HistorySelectionCaptureV1
  await expect(save(native, source)).rejects.toMatchObject({
    code: "unsupported_history_capability"
  })
  expect(io.read).not.toHaveBeenCalled()
  expect(io.write).not.toHaveBeenCalled()
})

it.each(["version", "content", "parent", "retired"])(
  "post-capture raw read-back refuses a later %s before installation",
  async (change) => {
    const native = owner(),
      source = (await service.captureHistorySnapshot(
        native,
        capture().view,
        "send"
      )) as HistorySelectionCaptureV1
    const row = response(2, edited)
    if (change === "version") row.version = 3 // Same text at v3 cannot certify the acknowledged v2.
    if (change === "content") row.content = "Changed raw text"
    if (change === "parent") row.parent_message_id = "foreign" as any
    let current = true
    io.read.mockImplementationOnce(async () => {
      if (change === "retired") current = false
      return row
    })
    await expect(
      service.verifyNativeMessageEditReceipt(
        native,
        source,
        { message_id: "u", message: edited, version: 2 },
        { validate_lease: () => current }
      )
    ).rejects.toThrow()
    expect(io.write).not.toHaveBeenCalled()
  }
)

it("post-capture raw read-back verifies the exact acknowledged numeric version under the original scope", async () => {
  const native = owner(),
    source = (await service.captureHistorySnapshot(
      native,
      capture().view,
      "send"
    )) as HistorySelectionCaptureV1
  const signal = new AbortController().signal
  io.read.mockResolvedValue(response(2, edited))
  await expect(
    service.verifyNativeMessageEditReceipt(
      native,
      source,
      { message_id: "u", message: edited, version: 2 },
      { signal }
    )
  ).resolves.toBeUndefined()
  expect(io.read).toHaveBeenCalledExactlyOnceWith("u", {
    requestScope: native.request_scope,
    scope: native.scope,
    signal
  })
  expect(io.write).not.toHaveBeenCalled()
})

it.each(["blank", "image", "unsettled", "system"])(
  "unsupported %s targets are refused before native message IO",
  async (kind) => {
    const native = owner(),
      candidate = capture()
    if (kind === "image") candidate.selected_content[0].images = ["image-asset"]
    if (kind === "unsettled") candidate.rows[0].settled = false
    if (kind === "system") candidate.rows[0].role = "system"
    io.capture.mockResolvedValue(candidate)
    if (kind === "unsettled")
      await expect(
        service.captureHistorySnapshot(native, candidate.view, "send")
      ).rejects.toThrow()
    else {
      const source = (await service.captureHistorySnapshot(
        native,
        candidate.view,
        "send"
      )) as HistorySelectionCaptureV1
      await expect(
        service.saveNativeHistoryMessage(
          native,
          source,
          { id: "u", message: original },
          kind === "blank" ? " \n" : edited
        )
      ).rejects.toThrow()
    }
    expect(io.read).not.toHaveBeenCalled()
    expect(io.write).not.toHaveBeenCalled()
  }
)
