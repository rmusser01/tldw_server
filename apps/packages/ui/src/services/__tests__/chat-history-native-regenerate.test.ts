import type {
  HistoryNodeV1,
  HistorySelectedContentV1,
  HistorySelectionCaptureV1
} from "@/types/history-selection"
import { beforeEach, describe, expect, it, vi } from "vitest"

import * as service from "../chat-history-selection"

const io = vi.hoisted(() => ({ capture: vi.fn() }))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { captureHistorySelection: io.capture }
}))
const RED_IMAGE =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
const BLUE_IMAGE =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYPgPAAEDAQAIicLsAAAAAElFTkSuQmCC"
const owner = () => ({
  kind: "native" as const,
  conversation_id: "chat",
  owner_key: "owner",
  request_scope: {
    config: {
      serverUrl: "https://original.test",
      authMode: "multi-user" as const
    },
    userId: "alice"
  },
  validate_lease: () => true
})
const intent = () => ({
  owner_key: "owner",
  conversation_id: "chat",
  message_id: "a",
  message_revision: "opaque-answer",
  input_message_id: "u",
  input_message_revision: "opaque-user",
  isCurrent: () => true,
  isOriginCurrent: () => true
})
const capture = (
  input: Partial<HistorySelectedContentV1> = {}
): HistorySelectionCaptureV1 => {
  const rows = [
    {
      id: "u",
      conversation_id: "chat",
      revision: "opaque-user",
      parent_id: null,
      role: "user",
      settled: true
    },
    {
      id: "a",
      conversation_id: "chat",
      revision: "opaque-answer",
      parent_id: "u",
      role: "assistant",
      settled: true
    }
  ]
  return {
    status: "captured",
    purpose: "send",
    view: {
      owner_key: "owner",
      conversation_id: "chat",
      view_session_id: "view",
      selection_revision: 7,
      interpretation: { kind: "parent_graph_v1" },
      cursor: { kind: "after_message", message_id: "a" }
    },
    storage_context_digest: "storage",
    snapshot: {
      version: 1,
      owner_key: "owner",
      conversation_id: "chat",
      source_digest: "source",
      storage_context_digest: "storage",
      fences: { conversation: "1", history: "1", settings: "1" },
      interpretation_status: { kind: "parent_graph_v1" },
      nodes: [
        ...rows,
        { ...rows[1], id: "sibling", revision: "opaque-sibling" }
      ]
    },
    rows,
    selected_content: [
      {
        id: "u",
        revision: "opaque-user",
        message: "  Literal {{char}} input  \n",
        images: [],
        extra_metadata: { sender_role: "user" },
        ...input
      },
      {
        id: "a",
        revision: "opaque-answer",
        message: "Original answer",
        images: []
      }
    ]
  }
}
const admission = () => ({
  version: 1 as const,
  owner_key: "owner",
  conversation_id: "chat",
  input_message_id: "u",
  input_message_revision: "1",
  selection_digest: "original-admission-digest",
  originating_selection_revision: 1,
  messages: []
})
beforeEach(() => {
  vi.resetAllMocks()
  io.capture.mockResolvedValue(capture())
})
describe("explicit native regeneration service contract", () => {
  it("captures after the exact saved assistant without changing the mounted origin", async () => {
    const native = owner(),
      source = await service.captureHistorySnapshot(
        native,
        capture().view,
        "send"
      )
    if (source.status !== "captured") throw new Error("missing fixture capture")
    const before = structuredClone(source)
    expect(
      service.nativeRegenerateView(native, source.view, source, intent())
    ).toEqual({
      ...source.view,
      selection_revision: 8,
      cursor: { kind: "after_message", message_id: "a" }
    })
    expect(source).toEqual(before)
  })
  const boundaries: Array<{
    label: string
    target?: Partial<HistoryNodeV1>
    user?: Partial<HistoryNodeV1>
    owner?: string
    conversation?: string
    unselected?: boolean
  }> = [
    { label: "target revision", target: { revision: "new-answer" } },
    { label: "user revision", user: { revision: "new-user" } },
    { label: "target role", target: { role: "tool" } },
    { label: "user role", user: { role: "assistant" } },
    { label: "direct parent", target: { parent_id: "sibling" } },
    { label: "unsettled target", target: { settled: false } },
    { label: "unsettled user", user: { settled: false } },
    { label: "foreign owner", owner: "foreign" },
    { label: "foreign conversation", conversation: "foreign" },
    { label: "target not selected", unselected: true }
  ]
  it.each(boundaries)(
    "refuses changed $label before returning a dispatch view",
    async (patch) => {
      const native = owner(),
        source = await service.captureHistorySnapshot(
          native,
          capture().view,
          "send"
        )
      if (source.status !== "captured")
        throw new Error("missing fixture capture")
      const change = (node: HistoryNodeV1): HistoryNodeV1 => ({
        ...node,
        ...(node.id === "a" ? patch.target : node.id === "u" ? patch.user : {})
      })
      const changed: HistorySelectionCaptureV1 = {
        ...source,
        snapshot: {
          ...source.snapshot,
          owner_key: patch.owner ?? source.snapshot.owner_key,
          conversation_id:
            patch.conversation ?? source.snapshot.conversation_id,
          nodes: source.snapshot.nodes.map(change)
        },
        rows: (patch.unselected ? source.rows.slice(0, -1) : source.rows).map(
          change
        ),
        selected_content: (patch.unselected
          ? source.selected_content.slice(0, -1)
          : source.selected_content
        ).map((row) => ({
          ...row,
          revision:
            (row.id === "a"
              ? patch.target?.revision
              : row.id === "u"
                ? patch.user?.revision
                : undefined) ?? row.revision
        }))
      }
      expect(() =>
        service.nativeRegenerateView(native, source.view, changed, intent())
      ).toThrow(
        /message_changed|owner_conversation_mismatch|unsupported_history_regeneration/
      )
    }
  )
  it.each(["source", "origin"])(
    "refuses retired %s leases",
    async (boundary) => {
      const native = owner(),
        source = await service.captureHistorySnapshot(
          native,
          capture().view,
          "send"
        )
      if (source.status !== "captured")
        throw new Error("missing fixture capture")
      const action = intent()
      if (boundary === "source") action.isCurrent = () => false
      else action.isOriginCurrent = () => false
      expect(() =>
        service.nativeRegenerateView(native, source.view, source, action)
      ).toThrow("stale_selection")
    }
  )
  it("preserves raw placeholder text and whitespace", () => {
    expect(service.nativeRegenerationMessage(capture(), intent(), 1)).toEqual({
      role: "user",
      content: "  Literal {{char}} input  \n"
    })
  })
  it.each(["", "  Literal {{char}} input  \n"])(
    "preserves ordered inline image details with text %j",
    (text) => {
      const source = capture({
        message: text || "<Image attachment x2>",
        images: [RED_IMAGE, BLUE_IMAGE],
        extra_metadata: {
          sender_role: "user",
          image_details: ["high", "low"],
          ...(text ? {} : { content_placeholder_reason: "image_attachment" })
        }
      })
      expect(service.nativeRegenerationMessage(source, intent(), 1)).toEqual({
        role: "user",
        content: [
          ...(text ? [{ type: "text", text }] : []),
          { type: "image_url", image_url: { url: RED_IMAGE, detail: "high" } },
          { type: "image_url", image_url: { url: BLUE_IMAGE, detail: "low" } }
        ]
      })
    }
  )
  it("does not reinterpret an edited literal image placeholder as empty input", () => {
    const source = capture({
      message: "<Image attachment x1>",
      images: [RED_IMAGE],
      extra_metadata: {
        sender_role: "user",
        content_placeholder_reason: "image_attachment"
      }
    })
    expect(
      service.nativeRegenerationMessage(source, intent(), 2).content
    ).toEqual([
      { type: "text", text: "<Image attachment x1>" },
      { type: "image_url", image_url: { url: RED_IMAGE, detail: "auto" } }
    ])
  })
  it.each([
    { image_details: ["high"] },
    { image_details: ["high", "unsupported"] },
    { sender_role: "assistant" },
    { tool_call_id: "tool" },
    { function_call: { name: "tool" } }
  ])(
    "refuses incomplete or unsupported saved input metadata %j",
    (metadata) => {
      const source = capture({
        images: [RED_IMAGE, BLUE_IMAGE],
        extra_metadata: { sender_role: "user", ...metadata }
      })
      expect(() =>
        service.nativeRegenerationMessage(source, intent(), 1)
      ).toThrow("unsupported_history_regeneration")
    }
  )
  it("qualifies an original admission without replacing its original selection provenance", async () => {
    const native = owner(),
      source = await service.captureHistorySnapshot(
        native,
        capture().view,
        "send"
      )
    if (source.status !== "captured") throw new Error("missing fixture capture")
    const original = admission(),
      accepted = service.parseNativeRegenerationAdmission(
        native,
        source,
        1,
        original
      )
    expect(accepted).toEqual(original)
    expect(Object.isFrozen(accepted)).toBe(true)
  })
  it.each([
    { owner_key: "foreign" },
    { conversation_id: "foreign" },
    { input_message_id: "sibling" },
    { input_message_revision: "2" },
    { input_message_revision: "opaque-user" },
    { version: 2 },
    { input_message_id: "" },
    { selection_digest: "" },
    { messages: [{ id: "ancestor", revision: "" }] },
    { unsafe_extra: "secret" }
  ])("rejects unrelated or malformed original admission %j", async (patch) => {
    const native = owner(),
      source = await service.captureHistorySnapshot(
        native,
        capture().view,
        "send"
      )
    if (source.status !== "captured") throw new Error("missing fixture capture")
    expect(() =>
      service.parseNativeRegenerationAdmission(native, source, 1, {
        ...admission(),
        ...patch
      })
    ).toThrow(
      /owner_conversation_mismatch|invalid_history_admission|unsupported_history_response/
    )
  })
})

it.each([
  { inputVersion: Number.MAX_SAFE_INTEGER },
  { messages: [{ id: "u", revision: "1" }] },
  { messages: [{ id: "a", revision: "1" }] },
  {
    messages: [
      { id: "ancestor", revision: "1" },
      { id: "ancestor", revision: "1" }
    ]
  }
])(
  "rejects impossible original input admission provenance %j",
  async (patch) => {
    const native = owner(),
      source = await service.captureHistorySnapshot(
        native,
        capture().view,
        "send"
      )
    if (source.status !== "captured") throw new Error("missing fixture capture")
    expect(() =>
      service.parseNativeRegenerationAdmission(
        native,
        source,
        patch.inputVersion ?? 1,
        {
          ...admission(),
          ...(patch.messages ? { messages: patch.messages } : {}),
          ...(patch.inputVersion
            ? { input_message_revision: String(patch.inputVersion) }
            : {})
        }
      )
    ).toThrow("invalid_history_admission")
  }
)
