import { describe, expect, it, vi } from "vitest"
import {
  loadOwnedWorkspace,
  type OwnedWorkspaceBundle,
  type OwnedWorkspaceReader
} from "../workspace-api"

const id = "47fd3ca9-a34b-5e36-be11-ebc15ce38fe6"
const timestamp = "2026-09-13T12:00:00Z"

function fixture(): OwnedWorkspaceBundle {
  return {
    workspace: {
      id,
      name: "Recipient research",
      archived: false,
      deleted: false,
      workspace_profile: "research",
      study_materials_policy: "general",
      banner_title: "Flood study",
      banner_subtitle: "Recipient copy",
      banner_color: "#225544",
      audio_provider: "local",
      audio_model: "speech",
      audio_voice: "voice",
      audio_speed: 1.25,
      created_at: timestamp,
      last_modified: timestamp,
      version: 4,
      assistantDefaults: {
        assistantKind: "persona",
        assistantId: "persona-1",
        personaMemoryMode: "read_only",
        voice: null,
        style: null,
        toolPolicyProfileId: null
      },
      effectiveAssistantDefault: {
        status: "available",
        source: "workspace",
        assistantKind: "persona",
        assistantId: "persona-1",
        label: "Researcher",
        personaMemoryMode: "read_only",
        degradedReason: null
      }
    },
    sources: [true, false].map((selected, position) => ({
      id: `source-${position}`,
      workspace_id: id,
      media_id: position + 1,
      title: `Report ${position}`,
      source_type: "document",
      url: null,
      position,
      selected,
      added_at: timestamp,
      version: 2,
      review_state: "reviewed",
      reviewed_by_user_id: "2"
    })),
    artifacts: [
      {
        id: "artifact-1",
        workspace_id: id,
        artifact_type: "report",
        title: "Brief",
        status: "completed",
        content: "Evidence",
        total_tokens: 12,
        total_cost_usd: null,
        created_at: timestamp,
        completed_at: timestamp,
        version: 3,
        review_state: "accepted",
        producer_metadata: { producer_type: "acp" },
        source_lineage: [{ source_id: "source-0" }],
        version_metadata: { revision_reason: "Reviewed" }
      }
    ],
    notes: [
      {
        id: 3,
        workspace_id: id,
        title: "Copied note",
        content: "Independent evidence",
        keywords_json: "[]",
        created_at: timestamp,
        last_modified: timestamp,
        version: 1
      }
    ]
  }
}

function readerFor(bundle = fixture()) {
  return {
    getWorkspace: vi
      .fn<OwnedWorkspaceReader["getWorkspace"]>()
      .mockResolvedValue(bundle.workspace),
    getWorkspaceSources: vi
      .fn<OwnedWorkspaceReader["getWorkspaceSources"]>()
      .mockResolvedValue(bundle.sources),
    getWorkspaceArtifacts: vi
      .fn<OwnedWorkspaceReader["getWorkspaceArtifacts"]>()
      .mockResolvedValue(bundle.artifacts),
    getWorkspaceNotes: vi
      .fn<OwnedWorkspaceReader["getWorkspaceNotes"]>()
      .mockResolvedValue(bundle.notes)
  }
}

describe("owned workspace read-only loading", () => {
  it.each(["", "null", "{}", "[null]", '["valid",42]', "not json"])(
    "rejects corrupt note keywords %j before exposing an editable note",
    async (keywords) => {
      const bundle = fixture()
      bundle.notes[0].keywords_json = keywords
      await expect(
        loadOwnedWorkspace(id, readerFor(bundle), new AbortController().signal)
      ).rejects.toMatchObject({ reason: "invalid-response", resource: "notes" })
    }
  )

  it("retains the complete server contract, including settings and selection", async () => {
    const bundle = fixture()
    const reader = readerFor(bundle)
    const result = await loadOwnedWorkspace(
      id,
      reader,
      new AbortController().signal
    )
    expect(result).toEqual(bundle)
    expect(result.workspace.study_materials_policy).toBe("general")
    expect(result.workspace.effectiveAssistantDefault?.assistantId).toBe(
      "persona-1"
    )
    expect(result.sources.map((source) => source.selected)).toEqual([
      true,
      false
    ])
    expect(result.notes[0].content).toBe("Independent evidence")
    for (const read of Object.values(reader))
      expect(read).toHaveBeenCalledExactlyOnceWith(id)
  })

  it("accepts successful empty collections", async () => {
    const bundle = { ...fixture(), sources: [], artifacts: [], notes: [] }
    await expect(
      loadOwnedWorkspace(id, readerFor(bundle), new AbortController().signal)
    ).resolves.toEqual(bundle)
  })

  it.each([
    "getWorkspace",
    "getWorkspaceSources",
    "getWorkspaceArtifacts",
    "getWorkspaceNotes"
  ] as const)(
    "propagates %s failures rather than returning partial content",
    async (method) => {
      const reader = readerFor()
      const failure = Object.assign(new Error("Read failed"), { status: 403 })
      reader[method].mockRejectedValueOnce(failure)
      await expect(
        loadOwnedWorkspace(id, reader, new AbortController().signal)
      ).rejects.toBe(failure)
      if (method === "getWorkspace")
        expect(reader.getWorkspaceSources).not.toHaveBeenCalled()
    }
  )

  it.each(["archived", "deleted"] as const)(
    "rejects a %s workspace before reading content",
    async (field) => {
      const bundle = fixture()
      bundle.workspace[field] = true
      const reader = readerFor(bundle)
      await expect(
        loadOwnedWorkspace(id, reader, new AbortController().signal)
      ).rejects.toMatchObject({ reason: "unavailable" })
      expect(reader.getWorkspaceSources).not.toHaveBeenCalled()
      expect(reader.getWorkspaceArtifacts).not.toHaveBeenCalled()
      expect(reader.getWorkspaceNotes).not.toHaveBeenCalled()
    }
  )

  it.each(["", " "])(
    "rejects an empty target without dispatching reads",
    async (target) => {
      const reader = readerFor()
      await expect(
        loadOwnedWorkspace(target, reader, new AbortController().signal)
      ).rejects.toMatchObject({ reason: "invalid-response" })
      expect(reader.getWorkspace).not.toHaveBeenCalled()
    }
  )

  it.each([
    { id: "different-workspace" },
    { deleted: undefined },
    { archived: "false" },
    { version: -1 },
    { version: "1" },
    { study_materials_policy: undefined },
    { workspace_profile: "invalid" },
    { audio_speed: "fast" },
    { created_at: "invalid" },
    { banner_title: 42 },
    { name: {} },
    { effectiveAssistantDefault: "not-an-object" },
    { assistant_defaults: { assistant_kind: "persona", assistant_id: 12 } },
    {
      assistantDefaults: { assistantKind: "persona", assistantId: "persona-1" }
    },
    {
      effective_assistant_default: {
        status: "available",
        source: "workspace",
        label: {}
      }
    },
    { effectiveAssistantDefault: { status: "available", source: "workspace" } }
  ])("rejects invalid metadata %j", async (fields) => {
    const reader = readerFor()
    reader.getWorkspace.mockResolvedValueOnce({
      ...fixture().workspace,
      ...fields
    } as never)
    await expect(
      loadOwnedWorkspace(id, reader, new AbortController().signal)
    ).rejects.toMatchObject({ reason: "invalid-response" })
    expect(reader.getWorkspaceNotes).not.toHaveBeenCalled()
  })

  it.each([
    "getWorkspaceSources",
    "getWorkspaceArtifacts",
    "getWorkspaceNotes"
  ] as const)(
    "rejects absent, malformed, foreign or duplicate rows from %s",
    async (method) => {
      const collection = {
        getWorkspaceSources: "sources",
        getWorkspaceArtifacts: "artifacts",
        getWorkspaceNotes: "notes"
      } as const
      const row = fixture()[collection[method]][0]
      for (const value of [
        undefined,
        null,
        {},
        [null],
        [{}],
        [{ ...row, workspace_id: "foreign" }],
        [row, row]
      ]) {
        const reader = readerFor()
        reader[method].mockResolvedValueOnce(value as never)
        await expect(
          loadOwnedWorkspace(id, reader, new AbortController().signal)
        ).rejects.toMatchObject({ reason: "invalid-response" })
      }
    }
  )

  it.each([
    ["getWorkspaceSources", { selected: undefined }],
    ["getWorkspaceSources", { media_id: "1" }],
    ["getWorkspaceSources", { added_at: "invalid" }],
    ["getWorkspaceArtifacts", { content: {} }],
    ["getWorkspaceArtifacts", { version: undefined }],
    ["getWorkspaceArtifacts", { review_state: {} }],
    ["getWorkspaceArtifacts", { preview_text: {} }],
    ["getWorkspaceArtifacts", { schema_version: "1" }],
    ["getWorkspaceNotes", { content: undefined }],
    ["getWorkspaceNotes", { id: "3" }]
  ] as const)("rejects invalid fields from %s: %j", async (method, fields) => {
    const bundle = fixture()
    const reader = readerFor(bundle)
    const collection = {
      getWorkspaceSources: "sources",
      getWorkspaceArtifacts: "artifacts",
      getWorkspaceNotes: "notes"
    } as const
    reader[method].mockResolvedValueOnce([
      { ...bundle[collection[method]][0], ...fields }
    ] as never)
    await expect(
      loadOwnedWorkspace(id, reader, new AbortController().signal)
    ).rejects.toMatchObject({ reason: "invalid-response" })
  })

  it("does not dispatch when already cancelled", async () => {
    const controller = new AbortController()
    controller.abort()
    const reader = readerFor()
    await expect(
      loadOwnedWorkspace(id, reader, controller.signal)
    ).rejects.toMatchObject({ name: "AbortError" })
    expect(reader.getWorkspace).not.toHaveBeenCalled()
  })

  it("rejects cancellation while metadata is pending without starting collection reads", async () => {
    const controller = new AbortController()
    const reader = readerFor()
    let finish!: (value: OwnedWorkspaceBundle["workspace"]) => void
    reader.getWorkspace.mockImplementation(
      () =>
        new Promise((resolve) => {
          finish = resolve
        })
    )
    const pending = loadOwnedWorkspace(id, reader, controller.signal)
    const rejected = expect(pending).rejects.toMatchObject({
      name: "AbortError"
    })
    controller.abort()
    await rejected
    finish(fixture().workspace)
    await Promise.resolve()
    expect(reader.getWorkspaceSources).not.toHaveBeenCalled()
  })

  it("rejects cancellation while a collection is pending and ignores its late result", async () => {
    const controller = new AbortController()
    const reader = readerFor()
    let finish!: (value: OwnedWorkspaceBundle["notes"]) => void
    reader.getWorkspaceNotes.mockImplementation(
      () =>
        new Promise((resolve) => {
          finish = resolve
        })
    )
    const pending = loadOwnedWorkspace(id, reader, controller.signal)
    const rejected = expect(pending).rejects.toMatchObject({
      name: "AbortError"
    })
    await vi.waitFor(() => expect(reader.getWorkspaceNotes).toHaveBeenCalled())
    controller.abort()
    await rejected
    finish(fixture().notes)
  })

  it("removes abort listeners after a successful load", async () => {
    const controller = new AbortController()
    const add = vi.spyOn(controller.signal, "addEventListener")
    const remove = vi.spyOn(controller.signal, "removeEventListener")
    await loadOwnedWorkspace(id, readerFor(), controller.signal)
    for (const [event, listener] of add.mock.calls) {
      expect(remove).toHaveBeenCalledWith(event, listener)
    }
  })

  it("retains raw assistant defaults without dropping normalized fields", async () => {
    const bundle = fixture()
    bundle.workspace.assistant_defaults = {
      assistant_kind: "persona",
      assistant_id: "persona-1"
    }
    bundle.workspace.effective_assistant_default = {
      status: "available",
      source: "workspace",
      assistant_kind: "persona",
      assistant_id: "persona-1"
    }
    const result = await loadOwnedWorkspace(
      id,
      readerFor(bundle),
      new AbortController().signal
    )
    expect(result.workspace.assistant_defaults?.assistant_id).toBe("persona-1")
    expect(result.workspace.assistantDefaults?.assistantId).toBe("persona-1")
    expect(result.workspace.effective_assistant_default?.assistant_id).toBe(
      "persona-1"
    )
    expect(result.workspace.effectiveAssistantDefault?.assistantId).toBe(
      "persona-1"
    )
  })

  it.each(["raw", "normalized"] as const)(
    "rejects inconsistent %s effective-default statuses",
    async (representation) => {
      const invalid = [
        {
          status: "available",
          assistantKind: null,
          assistantId: "persona-1",
          degradedReason: null
        },
        {
          status: "available",
          assistantKind: "persona",
          assistantId: null,
          degradedReason: null
        },
        {
          status: "available",
          assistantKind: "persona",
          assistantId: "persona-1",
          degradedReason: "permission_denied"
        },
        {
          status: "unavailable",
          assistantKind: null,
          assistantId: null,
          degradedReason: null
        },
        {
          status: "none",
          assistantKind: "persona",
          assistantId: null,
          degradedReason: null
        },
        {
          status: "none",
          assistantKind: null,
          assistantId: null,
          degradedReason: "permission_denied"
        }
      ]
      for (const values of invalid) {
        const bundle = fixture()
        if (representation === "raw") {
          bundle.workspace.effective_assistant_default = {
            status: values.status,
            source: "workspace",
            assistant_kind: values.assistantKind,
            assistant_id: values.assistantId,
            degraded_reason: values.degradedReason
          } as never
        } else {
          bundle.workspace.effectiveAssistantDefault = {
            ...values,
            source: "workspace",
            label: null,
            personaMemoryMode: null
          } as never
        }
        await expect(
          loadOwnedWorkspace(
            id,
            readerFor(bundle),
            new AbortController().signal
          )
        ).rejects.toMatchObject({
          reason: "invalid-response",
          resource: "workspace"
        })
      }
    }
  )

  it.each(["none", "unavailable"] as const)(
    "accepts a valid %s effective default",
    async (status) => {
      const bundle = fixture()
      bundle.workspace.effectiveAssistantDefault = {
        status,
        source: status === "none" ? "none" : "workspace",
        assistantKind: null,
        assistantId: null,
        label: null,
        personaMemoryMode: null,
        degradedReason: status === "none" ? null : "permission_denied"
      }
      await expect(
        loadOwnedWorkspace(id, readerFor(bundle), new AbortController().signal)
      ).resolves.toEqual(bundle)
    }
  )
})
