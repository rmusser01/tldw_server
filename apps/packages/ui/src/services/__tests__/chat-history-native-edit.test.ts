import { beforeEach, describe, expect, it, vi } from "vitest"
import type { HistorySelectionCaptureV1 } from "@/types/history-selection"
const io = vi.hoisted(() => ({
  capture: vi.fn(),
  read: vi.fn(),
  write: vi.fn(),
  remove: vi.fn()
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    captureHistorySelection: io.capture,
    getMessage: io.read,
    editMessage: io.write,
    deleteMessage: io.remove
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


describe("native leaf Delete authority", () => {
  const target = {
    id: "a",
    message: "Answer",
    isBot: true,
    images: [] as string[]
  }
  const leafRow = (patch = {}) => ({
    id: "a",
    conversation_id: "chat",
    sender: "assistant",
    content: "Answer",
    parent_message_id: "u",
    has_image: false,
    version: 7,
    ...patch
  })
  const setup = async () => {
    const native = owner()
    const source = (await service.captureHistorySnapshot(
      native,
      capture().view,
      "send"
    )) as HistorySelectionCaptureV1
    io.capture.mockClear()
    io.read.mockResolvedValue(leafRow())
    io.remove.mockResolvedValue(undefined)
    return { native, source }
  }
  it("uses distinct numeric and opaque CAS with the original scope and signal", async () => {
    const { native, source } = await setup()
    const signal = new AbortController().signal
    expect(
      await service.deleteNativeHistoryMessage(native, source, target, {
        signal
      })
    ).toEqual({ message_id: "a", parent_message_id: "u" })
    expect(io.remove).toHaveBeenCalledExactlyOnceWith("a", 7, "chat", {
      requestScope: native.request_scope,
      scope: native.scope,
      signal,
      requireLeaf: true,
      expectedRevision: "1"
    })
    expect(io.write).not.toHaveBeenCalled()
  })
  it.each([
    { version: "7" },
    { version: 0 },
    { version: Number.MAX_SAFE_INTEGER },
    { content: "Other" },
    { parent_message_id: null },
    { sender: "user" },
    { has_image: true },
    { conversation_id: "other" }
  ])("rejects raw target mismatch before deletion %j", async (patch) => {
    const { native, source } = await setup()
    io.read.mockResolvedValue(leafRow(patch))
    await expect(
      service.deleteNativeHistoryMessage(native, source, target)
    ).rejects.toThrow("message_changed")
    expect(io.remove).not.toHaveBeenCalled()
  })
  it("rejects a hidden descendant before a destructive request", async () => {
    const { native, source } = await setup()
    const changed = structuredClone(source)
    changed.snapshot.nodes.push({
      id: "hidden",
      revision: "hidden",
      parent_id: "a",
      role: "user",
      settled: true
    })
    await expect(
      service.deleteNativeHistoryMessage(native, changed, target)
    ).rejects.toThrow("message_has_descendants")
    expect(io.read).not.toHaveBeenCalled()
    expect(io.remove).not.toHaveBeenCalled()
  })
  it("does not adopt same-text replacement authority from a fresh capture", async () => {
    const { native, source } = await setup()
    const changed = capture()
    changed.rows[1].revision =
      changed.snapshot.nodes[1].revision =
      changed.selected_content[1].revision =
        "changed"
    io.capture.mockResolvedValue(changed)
    await expect(
      service.deleteNativeHistoryMessage(native, source, target)
    ).rejects.toThrow("message_changed")
    expect(io.remove).not.toHaveBeenCalled()
  })
  it("supports exact ordered image-only leaf payloads", async () => {
    const native = owner(),
      initial = capture()
    initial.selected_content[1].message = ""
    initial.selected_content[1].images = [
      "data:image/png;base64,YQ==",
      "data:image/png;base64,Yg=="
    ]
    io.capture.mockResolvedValue(initial)
    const source = (await service.captureHistorySnapshot(
      native,
      initial.view,
      "send"
    )) as HistorySelectionCaptureV1
    io.read.mockResolvedValue(leafRow({ content: "", has_image: true }))
    io.remove.mockResolvedValue(undefined)
    await expect(
      service.deleteNativeHistoryMessage(native, source, {
        ...target,
        message: "",
        images: initial.selected_content[1].images
      })
    ).resolves.toMatchObject({ message_id: "a" })
    expect(io.remove).toHaveBeenCalledOnce()
  })
  it.each(["read", "capture", "remove"])(
    "retirement after held %s prevents success and replay",
    async (stage) => {
      const { native, source } = await setup()
      let current = true
      const boundary =
        stage === "read"
          ? io.read
          : stage === "capture"
            ? io.capture
            : io.remove
      boundary.mockImplementationOnce(async () => {
        current = false
        return stage === "read"
          ? leafRow()
          : stage === "capture"
            ? capture()
            : undefined
      })
      await expect(
        service.deleteNativeHistoryMessage(native, source, target, {
          validate_lease: () => current
        })
      ).rejects.toThrow()
      expect(io.remove).toHaveBeenCalledTimes(stage === "remove" ? 1 : 0)
    }
  )
  it("preserves a refusal without dispatching another Delete", async () => {
    const { native, source } = await setup(),
      conflict = Object.assign(new Error("message_has_descendants"), {
        status: 409
      })
    io.remove.mockRejectedValue(conflict)
    await expect(
      service.deleteNativeHistoryMessage(native, source, target)
    ).rejects.toBe(conflict)
    expect(io.remove).toHaveBeenCalledOnce()
  })
  const refreshed = (source: HistorySelectionCaptureV1) => {
    const result = structuredClone(source)
    result.view = {
      ...source.view,
      cursor: { kind: "after_message", message_id: "u" },
      selection_revision: 2
    }
    result.snapshot.nodes = result.snapshot.nodes.filter(
      (row) => row.id !== "a"
    )
    result.rows = result.rows.filter((row) => row.id !== "a")
    result.selected_content = result.selected_content.filter(
      (row) => row.id !== "a"
    )
    return result
  }
  it("qualifies the selected parent and preserves the original view and other branch", () => {
    const source = capture(),
      result = refreshed(source),
      receipt = { message_id: "a", parent_message_id: "u" }
    expect(() =>
      service.qualifyNativeMessageDeleteCapture(source, result, receipt)
    ).not.toThrow()
    expect(source.view.cursor).toEqual({
      kind: "after_message",
      message_id: "a"
    })
    expect(result.snapshot.nodes.map((row) => row.id)).toEqual(["u", "sibling"])
  })
  it.each([
    "not deleted",
    "parent changed",
    "wrong cursor",
    "settings changed",
    "other row missing"
  ])("rejects a falsely qualified post-Delete %s", (change) => {
    const source = capture(),
      result = refreshed(source)
    if (change === "not deleted")
      result.snapshot.nodes.push(source.snapshot.nodes[1])
    if (change === "parent changed") result.rows[0].revision = "changed"
    if (change === "wrong cursor") result.view = source.view
    if (change === "settings changed")
      result.snapshot.storage_context_digest = "changed"
    if (change === "other row missing")
      result.snapshot.nodes = result.snapshot.nodes.filter(
        (row) => row.id !== "sibling"
      )
    expect(() =>
      service.qualifyNativeMessageDeleteCapture(source, result, {
        message_id: "a",
        parent_message_id: "u"
      })
    ).toThrow()
  })
})
