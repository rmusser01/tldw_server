import { webcrypto } from "node:crypto"
import { beforeEach, describe, expect, it, vi } from "vitest"
import {
  prepareWebCaptureAcceptance,
  confirmWebCaptureAcceptance,
  assertWebCaptureHeadCurrent
} from "../research-web-capture"
import type { WorkspaceSource } from "@/types/workspace"
import type { WebClipperStatusResponse } from "@/services/web-clipper/types"
const mocks = vi.hoisted(() => ({
  status: vi.fn(),
  sources: vi.fn(),
  versions: vi.fn(),
  version: vi.fn(),
  details: vi.fn()
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getWebClipStatus: mocks.status,
    getWorkspaceSources: mocks.sources,
    listMediaDocumentVersions: mocks.versions,
    getMediaDocumentVersion: mocks.version,
    getMediaDetails: mocks.details
  }
}))
const input = {
  url: "https://EXAMPLE.com:443/%61?x=1",
  title: "Article",
  text: "\u001c  Café 🐎 \u0085",
  capturedAt: "2026-10-07T12:34:56Z",
  workspaceId: "ws-1"
}
const options = {
  requestScope: {
    config: {
      serverUrl: "https://owner.example",
      authMode: "single-user" as const,
      apiKey: "private"
    },
    userId: null
  },
  signal: new AbortController().signal
}
const versionUuid = "b4f6a910-1edb-4df4-a5fd-cd4c46532e58"
const canonicalNoteId = "166b2a30-6e51-45e0-9a04-dcc695dff248"
beforeEach(() => {
  vi.clearAllMocks()
  vi.stubGlobal("crypto", webcrypto)
})
async function fixture() {
  const body = await prepareWebCaptureAcceptance(input)
  const source = {
    id: `web-clipper:${body.clip_id}`,
    workspace_id: input.workspaceId,
    media_id: 71,
    url: input.url,
    title: input.title,
    source_type: "web_clip",
    position: 0,
    selected: true,
    added_at: input.capturedAt,
    review_state: "needs_review",
    version: 1
  }
  const version = {
    uuid: versionUuid,
    media_id: 71,
    version_number: 9,
    created_at: input.capturedAt,
    content: "Café 🐎",
    safe_metadata: {
      source: "web_clipper",
      clip_type: "article",
      clip_id: body.clip_id,
      workspace_id: input.workspaceId,
      source_url: input.url,
      capture_metadata: body.capture_metadata
    }
  }
  const status: WebClipperStatusResponse = {
    clip_id: body.clip_id,
    status: "saved",
    note: { id: canonicalNoteId, title: input.title, version: 200 },
    workspace_placements: [
      {
        workspace_id: input.workspaceId,
        workspace_note_id: 2,
        source_note_id: canonicalNoteId,
        source_note_version: 1
      }
    ],
    attachments: [],
    analysis: {},
    content_budget: {}
  }
  mocks.status.mockResolvedValue(status)
  mocks.sources.mockResolvedValue([source])
  mocks.versions.mockResolvedValue([version])
  mocks.version.mockResolvedValue(version)
  mocks.details.mockResolvedValue({ media_id: 71, content: { text: version.content } })
  return { body, source, version, status }
}
describe("public article acceptance", () => {
  it("freezes exact URL, Python-trimmed full body and fresh identity without fetching", async () => {
    const body = await prepareWebCaptureAcceptance(input)
    expect(body.content.full_extract).toBe("Café 🐎")
    expect(body.source_url).toBe(input.url)
    expect(body.capture_metadata.web_capture_v1.content_sha256).toBe(
      "9c6a202b476089b3950203cab347bc02f3c09c9e4fc491faa467228bef9895d3"
    )
    expect(Object.isFrozen(body.capture_metadata.web_capture_v1)).toBe(true)
    expect(Object.isFrozen(body.content)).toBe(true)
    expect(Object.isFrozen(body)).toBe(true)
    expect((await prepareWebCaptureAcceptance(input)).clip_id).not.toBe(
      body.clip_id
    )
    expect(JSON.stringify(body)).not.toContain("private")
    expect(mocks.status).not.toHaveBeenCalled()
    expect(body.workspace).toEqual({
      workspace_id: "ws-1",
      default_review_state: "needs_review"
    })
    expect(body.enhancements).toEqual({ run_ocr: false, run_vlm: false })
  })
  it.each([
    "file:///article",
    "https://@example.com/a",
    "https://user:pw@example.com",
    "https://localhost/a",
    "http://127.0.0.1/a",
    "https://example.local",
    "https://example.com/ a",
    "https://example.com/\\a"
  ])("rejects invalid URL %s", async (url) => {
    await expect(
      prepareWebCaptureAcceptance({ ...input, url })
    ).rejects.toThrow()
  })
  it.each(["2026-02-30T12:34:56Z", "2026-10-07T12:34:56+01:00", "2026-10-07"])(
    "rejects invalid UTC time %s",
    async (capturedAt) => {
      await expect(
        prepareWebCaptureAcceptance({ ...input, capturedAt })
      ).rejects.toThrow()
    }
  )
  it("preserves BOM and rejects empty, oversized and ill-formed text", async () => {
    expect(
      (await prepareWebCaptureAcceptance({ ...input, text: " \ufeffa " }))
        .content.full_extract
    ).toBe("\ufeffa")
    for (const text of [" \u0085", "a".repeat(1_000_001), "\ud800"])
      await expect(
        prepareWebCaptureAcceptance({ ...input, text })
      ).rejects.toThrow()
    expect(
      (
        await prepareWebCaptureAcceptance({
          ...input,
          text: "a".repeat(1_000_000)
        })
      ).content.full_extract.length
    ).toBe(1_000_000)
  })
  it("confirms a distinct canonical Note identity and exact Media version, recovering the same body", async () => {
    const { body, source, status } = await fixture()
    expect(status.note.id).not.toBe(body.clip_id)
    const current = vi.fn()
    const confirmed = await confirmWebCaptureAcceptance(body, options, current)
    expect(confirmed.source).toEqual(source)
    expect(confirmed.pin).toMatchObject({
      clipId: body.clip_id,
      mediaId: 71,
      versionNumber: 9,
      versionUuid,
      requestedUrl: input.url
    })
    expect(mocks.version).toHaveBeenCalledWith(71, 9, options)
    expect(mocks.status).toHaveBeenCalledWith(body.clip_id, options)
    expect(current).toHaveBeenCalled()
    // An independent Note edit leaves the immutable Media capture unchanged.
    mocks.status.mockResolvedValue({
      ...status,
      note: { ...status.note, title: "Edited Note", version: 201 }
    })
    expect(await confirmWebCaptureAcceptance(body, options, current)).toEqual(
      confirmed
    )
    expect(mocks.status.mock.calls.map((call) => call[0])).toEqual([
      body.clip_id,
      body.clip_id
    ])
  })
  it("fails on partial placement and owner invalidation", async () => {
    const { body, status } = await fixture()
    mocks.status.mockResolvedValue({
      ...status,
      status: "partially_saved",
      workspace_placements: []
    })
    await expect(
      confirmWebCaptureAcceptance(body, options, () => {})
    ).rejects.toThrow()
    await fixture()
    await expect(
      confirmWebCaptureAcceptance(body, options, () => {
        throw new Error("owner changed")
      })
    ).rejects.toThrow("owner changed")
  })
  it.each([
    ["missing canonical Note", { note: undefined }],
    ["null canonical Note", { note: null }],
    ["missing canonical Note ID", { note: { title: "Article", version: 200 } }],
    [
      "empty canonical Note ID",
      { note: { id: "", title: "Article", version: 200 } }
    ],
    [
      "blank canonical Note ID",
      { note: { id: " ", title: "Article", version: 200 } }
    ],
    [
      "wrong canonical Note identity",
      { note: { id: versionUuid, title: "Article", version: 200 } }
    ],
    ["wrong capture identity", { clip_id: versionUuid }],
    ["missing placement", { workspace_placements: [] }]
  ])("rejects %s before reading sources or versions", async (_name, change) => {
    const { body, status } = await fixture()
    mocks.status.mockResolvedValue({ ...status, ...change })
    await expect(
      confirmWebCaptureAcceptance(body, options, () => {})
    ).rejects.toThrow()
    expect(mocks.sources).not.toHaveBeenCalled()
    expect(mocks.versions).not.toHaveBeenCalled()
  })
  it("rejects a capture UUID masquerading as the placement's canonical Note ID", async () => {
    const { body, status } = await fixture()
    mocks.status.mockResolvedValue({
      ...status,
      workspace_placements: [
        { ...status.workspace_placements[0], source_note_id: body.clip_id }
      ]
    })
    await expect(
      confirmWebCaptureAcceptance(body, options, () => {})
    ).rejects.toThrow()
    expect(mocks.sources).not.toHaveBeenCalled()
  })
  it.each(["", " "])(
    "rejects matching but empty canonical identities %j",
    async (id) => {
      const { body, status } = await fixture()
      mocks.status.mockResolvedValue({
        ...status,
        note: { ...status.note, id },
        workspace_placements: [
          { ...status.workspace_placements[0], source_note_id: id }
        ]
      })
      await expect(
        confirmWebCaptureAcceptance(body, options, () => {})
      ).rejects.toThrow()
      expect(mocks.sources).not.toHaveBeenCalled()
    }
  )
  it("uses the canonical string identity contract without requiring a new UUID admission rule", async () => {
    const { body, status } = await fixture()
    mocks.status.mockResolvedValue({
      ...status,
      note: { ...status.note, id: "canonical-note-legacy" },
      workspace_placements: [
        {
          ...status.workspace_placements[0],
          source_note_id: "canonical-note-legacy"
        }
      ]
    })
    const { pin } = await confirmWebCaptureAcceptance(body, options, () => {})
    expect(pin.versionNumber).toBe(9)
    expect(pin.versionUuid).toBe(versionUuid)
  })
  it.each([
    ["missing placement Note ID", { source_note_id: undefined }],
    ["empty placement Note ID", { source_note_id: "" }],
    ["wrong placement Note identity", { source_note_id: versionUuid }],
    ["wrong placement workspace", { workspace_id: "other-workspace" }]
  ])("rejects %s before reading sources", async (_name, change) => {
    const { body, status } = await fixture()
    mocks.status.mockResolvedValue({
      ...status,
      workspace_placements: [{ ...status.workspace_placements[0], ...change }]
    })
    await expect(
      confirmWebCaptureAcceptance(body, options, () => {})
    ).rejects.toThrow()
    expect(mocks.sources).not.toHaveBeenCalled()
  })
  it.each(["workspace_id", "url", "id", "media_id"])(
    "rejects source mismatch %s",
    async (field) => {
      const { body, source } = await fixture()
      mocks.sources.mockResolvedValue([{ ...source, [field]: "wrong" }])
      await expect(
        confirmWebCaptureAcceptance(body, options, () => {})
      ).rejects.toThrow()
    }
  )
  it.each(["uuid", "content", "media_id"])(
    "rejects version mismatch %s",
    async (field) => {
      const { body, version } = await fixture()
      mocks.version.mockResolvedValue({ ...version, [field]: "wrong" })
      await expect(
        confirmWebCaptureAcceptance(body, options, () => {})
      ).rejects.toThrow()
    }
  )
  it("rejects descriptor mismatch and deleted pin", async () => {
    const { body, version } = await fixture()
    mocks.version.mockResolvedValue({
      ...version,
      safe_metadata: { ...version.safe_metadata, clip_id: "other" }
    })
    await expect(
      confirmWebCaptureAcceptance(body, options, () => {})
    ).rejects.toThrow()
    mocks.version.mockRejectedValue(new Error("404"))
    await expect(
      confirmWebCaptureAcceptance(body, options, () => {})
    ).rejects.toThrow("404")
  })
  it("checks current owned membership and rejects superseded head", async () => {
    const { body, source, version } = await fixture()
    const { pin } = await confirmWebCaptureAcceptance(body, options, () => {})
    const local = {
      id: source.id,
      mediaId: 71,
      url: input.url,
      webCapture: pin
    } as WorkspaceSource
    await assertWebCaptureHeadCurrent(local, input.workspaceId, options)
    mocks.versions.mockResolvedValue([{ ...version, version_number: 10 }])
    await expect(
      assertWebCaptureHeadCurrent(local, input.workspaceId, options)
    ).rejects.toThrow()
    mocks.sources.mockResolvedValue([])
    await expect(
      assertWebCaptureHeadCurrent(local, input.workspaceId, options)
    ).rejects.toThrow()
  })
})

describe("acceptance edge cases", () => {
  it("snapshots preview primitives before awaiting the digest", async () => {
    const preview = { ...input }
    const pending = prepareWebCaptureAcceptance(preview)
    preview.url = "https://other.example"
    preview.workspaceId = "other"
    preview.title = "Other"
    const body = await pending
    expect(body.source_url).toBe(input.url)
    expect(body.workspace.workspace_id).toBe(input.workspaceId)
    expect(body.source_title).toBe(input.title)
  })
  it.each([
    "http://100.64.0.1",
    "http://192.0.0.8",
    "http://192.0.2.1",
    "http://198.19.0.1",
    "http://203.0.113.1",
    "http://[fc00::1]",
    "http://[::ffff:127.0.0.1]",
    "http://[2001:db8::1]",
    "http://[2002::1]",
    "http://[3fff::1]",
    "http://[64:ff9b:1::1]",
    "http://[2001:100::1]"
  ])("rejects backend non-global literal %s", async (url) => {
    await expect(
      prepareWebCaptureAcceptance({ ...input, url })
    ).rejects.toThrow()
  })
  it.each([
    "http://8.8.8.8",
    "http://192.0.0.9",
    "http://[2606:4700:4700::1111]",
    "http://[::ffff:8.8.8.8]",
    "http://[2001:3::1]",
    "http://[2001:20::1]"
  ])("preserves accepted global literal %s", async (url) => {
    expect(
      (await prepareWebCaptureAcceptance({ ...input, url })).source_url
    ).toBe(url)
  })
  it("requires owner scope and rejects cancellation before reads", async () => {
    const { body } = await fixture()
    await expect(
      confirmWebCaptureAcceptance(body, {}, () => {})
    ).rejects.toThrow()
    const controller = new AbortController()
    controller.abort()
    await expect(
      confirmWebCaptureAcceptance(
        body,
        { ...options, signal: controller.signal },
        () => {}
      )
    ).rejects.toThrow()
    expect(mocks.status).not.toHaveBeenCalled()
  })
  it.each([
    "mode",
    "requested_url",
    "captured_at",
    "content_sha256",
    "refresh_of"
  ])("rejects active descriptor disagreement for %s", async (field) => {
    const { body, version } = await fixture()
    mocks.version.mockResolvedValue({
      ...version,
      safe_metadata: {
        ...version.safe_metadata,
        capture_metadata: {
          web_capture_v1: {
            ...(body.capture_metadata.web_capture_v1 as object),
            [field]: "wrong"
          }
        }
      }
    })
    await expect(
      confirmWebCaptureAcceptance(body, options, () => {})
    ).rejects.toThrow()
  })
  it("counts body characters as Python Unicode code points", async () => {
    const text = "🐎".repeat(500_001)
    expect(
      (await prepareWebCaptureAcceptance({ ...input, text })).content
        .full_extract
    ).toBe(text)
  })
})

it("applies the backend body bound after Python strip, without clipping text", async () => {
  const text = "a".repeat(1_000_000)
  expect(
    (await prepareWebCaptureAcceptance({ ...input, text: ` ${text} ` })).content
      .full_extract
  ).toBe(text)
})

it("rejects an owner switch while membership is in flight", async () => {
  const { body, source } = await fixture()
  let changed = false
  mocks.sources.mockImplementation(async () => {
    changed = true
    return [source]
  })
  await expect(
    confirmWebCaptureAcceptance(body, options, () => {
      if (changed) throw new Error("owner changed")
    })
  ).rejects.toThrow("owner changed")
  expect(mocks.versions).not.toHaveBeenCalled()
})

it("retains the refresh identity and freezes destination before any save", async () => {
  const body = await prepareWebCaptureAcceptance({
    ...input,
    refreshOf: versionUuid
  })
  expect(body.capture_metadata.web_capture_v1.refresh_of).toBe(versionUuid)
  expect(Object.isFrozen(body.workspace)).toBe(true)
})

// Active version history and current retrieval content are separate read contracts.
it("rejects current material left behind after the newest version is deleted", async () => {
  const { body, source, version } = await fixture()
  const { pin } = await confirmWebCaptureAcceptance(body, options, () => {})
  const local = {
    id: source.id,
    mediaId: 71,
    url: input.url,
    webCapture: pin
  } as WorkspaceSource
  mocks.versions.mockResolvedValue([version])
  mocks.details.mockResolvedValue({
    media_id: 71,
    content: { text: "Later changed article" }
  })
  await expect(
    assertWebCaptureHeadCurrent(local, input.workspaceId, options)
  ).rejects.toMatchObject({ sourceId: source.id })
})
it.each([
  { media_id: 72, content: { text: "Café 🐎" } },
  { media_id: 71, content: { text: null } },
  { media_id: 71, content: { text: " Café 🐎 " } }
])(
  "rejects missing, foreign or nonexact current content %j",
  async (detail) => {
    const { body, source } = await fixture()
    const { pin } = await confirmWebCaptureAcceptance(body, options, () => {})
    mocks.details.mockResolvedValue(detail)
    await expect(
      assertWebCaptureHeadCurrent(
        {
          id: source.id,
          mediaId: 71,
          url: input.url,
          webCapture: pin
        } as WorkspaceSource,
        input.workspaceId,
        options
      )
    ).rejects.toThrow()
  }
)
it.each(["Café 🐎", "Changed article"])(
  "recovers the exact checkpoint despite a newer active version: %s",
  async (content) => {
    const { body, version } = await fixture()
    const accepted = await confirmWebCaptureAcceptance(body, options, () => {})
    const newer = {
      ...version,
      version_number: 10,
      uuid: canonicalNoteId,
      content
    }
    mocks.versions.mockResolvedValue([newer, version])
    mocks.version.mockImplementation(async (_id, number) =>
      number === 9 ? version : newer
    )
    const recovered = await confirmWebCaptureAcceptance(
      body,
      options,
      () => {},
      accepted.pin
    )
    expect(recovered.pin).toEqual(accepted.pin)
    expect(mocks.version).toHaveBeenLastCalledWith(71, 9, options)
  }
)
it("finds an older active matching acceptance before a pin exists", async () => {
  const { body, version } = await fixture()
  const newer = {
    ...version,
    version_number: 10,
    uuid: canonicalNoteId,
    content: "Changed article"
  }
  mocks.versions.mockResolvedValue([newer, version])
  mocks.version.mockImplementation(async (_id, number) =>
    number === 9 ? version : newer
  )
  expect(
    (await confirmWebCaptureAcceptance(body, options, () => {})).pin
      .versionNumber
  ).toBe(9)
})
it.each(["missing", "uuid", "body", "404"])(
  "never replaces an invalid checkpoint with a matching newer version: %s",
  async (kind) => {
    const { body, version } = await fixture()
    const { pin } = await confirmWebCaptureAcceptance(body, options, () => {})
    const newer = { ...version, version_number: 10, uuid: canonicalNoteId }
    const old = {
      ...version,
      ...(kind === "uuid" ? { uuid: body.clip_id } : {}),
      ...(kind === "body" ? { content: "Changed" } : {})
    }
    mocks.versions.mockResolvedValue(
      kind === "missing" ? [newer] : [newer, old]
    )
    mocks.version.mockImplementation(async (_id, number) => {
      if (number === 9 && kind === "404") throw Error("404")
      return number === 9 ? old : newer
    })
    await expect(
      confirmWebCaptureAcceptance(body, options, () => {}, pin)
    ).rejects.toThrow()
  }
)
