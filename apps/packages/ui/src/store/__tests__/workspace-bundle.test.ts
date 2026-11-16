import { describe, expect, it } from "vitest"
import JSZip from "jszip"
import { workspaceArtifactDraftPayload } from "../workspace-artifact-codec"
import { buildWorkspaceBundleSnapshot, hydrateWorkspaceBundleSnapshot } from "../workspace"
import {
  LEGACY_WORKSPACE_EXPORT_BUNDLE_FORMAT,
  WORKSPACE_EXPORT_BUNDLE_FORMAT,
  WORKSPACE_EXPORT_BUNDLE_MANIFEST_FILE,
  WORKSPACE_EXPORT_BUNDLE_PAYLOAD_FILE,
  WORKSPACE_EXPORT_BUNDLE_SCHEMA_VERSION,
  WORKSPACE_EXPORT_BUNDLE_ZIP_MIME,
  createWorkspaceExportZipBlob,
  createWorkspaceExportZipFilename,
  parseWorkspaceImportFile,
  type WorkspaceExportBundle
} from "../workspace-bundle"

const createBundleFixture = (): WorkspaceExportBundle => ({
  format: WORKSPACE_EXPORT_BUNDLE_FORMAT,
  schemaVersion: WORKSPACE_EXPORT_BUNDLE_SCHEMA_VERSION,
  exportedAt: "2026-02-19T18:05:00.000Z",
  workspace: {
    name: "Alpha Research",
    tag: "workspace:alpha-research",
    createdAt: new Date("2026-02-18T09:00:00.000Z"),
    studyMaterialsPolicy: null,
    snapshot: {
      workspaceName: "Alpha Research",
      workspaceTag: "workspace:alpha-research",
      workspaceCreatedAt: new Date("2026-02-18T09:00:00.000Z"),
      studyMaterialsPolicy: null,
      sources: [
        {
          id: "source-1",
          mediaId: 101,
          title: "Alpha Source",
          type: "pdf",
          addedAt: new Date("2026-02-18T09:30:00.000Z")
        }
      ],
      selectedSourceIds: ["source-1"],
      generatedArtifacts: [
        {
          id: "artifact-1",
          type: "summary",
          title: "Alpha Summary",
          status: "completed",
          content: "Summary content",
          createdAt: new Date("2026-02-19T10:00:00.000Z"),
          completedAt: new Date("2026-02-19T10:01:00.000Z")
        }
      ],
      notes: "workspace notes",
      currentNote: {
        id: 5,
        title: "Workspace note",
        content: "content",
        keywords: ["alpha"],
        version: 1,
        isDirty: false
      },
      workspaceBanner: {
        title: "Alpha Banner",
        subtitle: "Workspace subtitle",
        image: {
          dataUrl: "data:image/webp;base64,alpha-banner",
          mimeType: "image/webp",
          width: 1400,
          height: 460,
          bytes: 24576,
          updatedAt: new Date("2026-02-19T10:05:00.000Z")
        }
      },
      leftPaneCollapsed: false,
      rightPaneCollapsed: true,
      audioSettings: {
        provider: "tldw",
        model: "kokoro",
        voice: "af_heart",
        speed: 1,
        format: "mp3"
      }
    },
    chatSession: {
      messages: [
        {
          isBot: false,
          name: "You",
          message: "hello",
          sources: []
        }
      ],
      history: [{ role: "user", content: "hello" }],
      historyId: "history-1",
      serverChatId: "server-chat-1"
    }
  }
})

const toSerializableBundle = (
  bundle: WorkspaceExportBundle
): WorkspaceExportBundle =>
  JSON.parse(JSON.stringify(bundle)) as WorkspaceExportBundle

describe("workspace bundle zip compatibility", () => {
  it("retains distinct in-memory dates that serialize to different saved data", async () => {
    const bundle = createBundleFixture()
    const artifact = bundle.workspace.snapshot.generatedArtifacts[0]
    artifact.data = { capturedAt: new Date("2026-01-01T00:00:00Z") }
    artifact.versionMetadata = { workspaceClient: { schemaVersion: 1, data: { capturedAt: new Date("2026-02-01T00:00:00Z") } } }
    const blob = await createWorkspaceExportZipBlob(bundle)
    const saved = (await parseWorkspaceImportFile(new File([blob], "dates.workspace.zip"))).workspace.snapshot.generatedArtifacts[0]
    expect(saved.data).toEqual({ capturedAt: "2026-01-01T00:00:00.000Z" })
    expect(saved.versionMetadata?.workspaceClient).toMatchObject({ data: { capturedAt: "2026-02-01T00:00:00.000Z" } })
    expect(artifact.data.capturedAt).toBeInstanceOf(Date)
  })
  it.each(["identical", "different", "unknown version", "client only"])("elides only a known identical client data projection (%s)", async mode => {
    const bundle = createBundleFixture()
    const artifact = bundle.workspace.snapshot.generatedArtifacts[0]
    const data = { sectionAnalysis: { content: "Full draft", claim_verification: { text_sha256: "a".repeat(64), evidence: { P1: { source_id: "source-1", media_id: 101 } } } } }
    artifact.data = data
    artifact.reviewMetadata = { savedCorrection: "Human review is retained." }
    artifact.versionMetadata = { retainedVersionField: "retain", workspaceClient: {
      schemaVersion: mode === "unknown version" ? 2 : 1,
      data: mode === "different" ? { distinctHistoricalData: true } : { sectionAnalysis: { claim_verification: data.sectionAnalysis.claim_verification, content: "Full draft" } },
      unknownClientField: "retain"
    } }
    if (mode === "client only") delete artifact.data
    const before = JSON.stringify(bundle)
    const blob = await createWorkspaceExportZipBlob(bundle)
    const parsed = await parseWorkspaceImportFile(new File([blob], "compact.workspace.zip"))
    const saved = parsed.workspace.snapshot.generatedArtifacts[0]
    const client = saved.versionMetadata!.workspaceClient as Record<string, unknown>
    expect(client).toMatchObject({ schemaVersion: mode === "unknown version" ? 2 : 1, unknownClientField: "retain" })
    expect(saved.versionMetadata!.retainedVersionField).toBe("retain")
    expect(saved.content).toBe(artifact.content)
    expect(saved.data).toEqual(artifact.data)
    expect(saved.reviewMetadata).toEqual(artifact.reviewMetadata)
    if (mode === "identical") {
      expect(client).not.toHaveProperty("data")
      expect(workspaceArtifactDraftPayload(saved)).toEqual(workspaceArtifactDraftPayload(artifact))
    } else expect(client.data).toEqual((artifact.versionMetadata!.workspaceClient as Record<string, unknown>).data)
    expect(JSON.stringify(bundle)).toBe(before)
  })
  it("builds sanitized zip filenames", () => {
    const filename = createWorkspaceExportZipFilename(
      "Alpha Research",
      "2026-02-19T18:05:00.000Z"
    )
    expect(filename).toBe("alpha-research-2026-02-19t18-05-00-000z.workspace.zip")
  })

  it("round-trips ZIP export/import with equivalent workspace data", async () => {
    const bundle = createBundleFixture()
    const zipBlob = await createWorkspaceExportZipBlob(bundle)
    const file = new File([zipBlob], "alpha.workspace.zip", {
      type: WORKSPACE_EXPORT_BUNDLE_ZIP_MIME
    })

    const parsed = await parseWorkspaceImportFile(file)

    expect(parsed).toEqual(toSerializableBundle(bundle))
  })

  it("round-trips workspaceBanner through zip export/import", async () => {
    const bundle = createBundleFixture()
    const zipBlob = await createWorkspaceExportZipBlob(bundle)
    const file = new File([zipBlob], "alpha.workspace.zip", {
      type: WORKSPACE_EXPORT_BUNDLE_ZIP_MIME
    })

    const parsed = await parseWorkspaceImportFile(file)

    expect(parsed.workspace.snapshot.workspaceBanner.title).toBe("Alpha Banner")
    expect(parsed.workspace.snapshot.workspaceBanner.subtitle).toBe(
      "Workspace subtitle"
    )
    expect(parsed.workspace.snapshot.workspaceBanner.image?.mimeType).toBe(
      "image/webp"
    )
  })

  it("round-trips a large completed chapter artifact through ZIP and imported snapshots without truncation", async () => {
    const bundle = createBundleFixture()
    const artifact = bundle.workspace.snapshot.generatedArtifacts[0]
    const sectionAnalysis = {
      batches: Array.from({ length: 117 }, (_, batchIndex) => ({
        status: "completed",
        evidence: Array.from({ length: 30 }, (_, passageIndex) => ({
          id: `batch-${batchIndex}-passage-${passageIndex}`,
          quote: `Exact source passage ${batchIndex}/${passageIndex}: ${"Evidence é文. ".repeat(100)}`,
          start_char: batchIndex * 30_000 + passageIndex * 1000,
          page_spans: [{ page_number: batchIndex + 34, printed_page_label: String(batchIndex + 1) }]
        }))
      }))
    }
    artifact.serverId = "canonical-large-output"
    artifact.workspaceArtifactId = "canonical-large-output"
    artifact.content = "## Completed chapter\nSummary with exact evidence [1].\n".repeat(12_000)
    artifact.data = { sectionAnalysis }
    artifact.versionMetadata = { workspaceClient: { schemaVersion: 1, data: { sectionAnalysis } } }
    const expected = toSerializableBundle(bundle).workspace.snapshot.generatedArtifacts[0]
    expect(new TextEncoder().encode(JSON.stringify(bundle)).byteLength).toBeGreaterThan(5 * 1024 * 1024)

    const zip = await createWorkspaceExportZipBlob(bundle)
    const parsed = await parseWorkspaceImportFile(new File([zip], "large.workspace.zip", {
      type: WORKSPACE_EXPORT_BUNDLE_ZIP_MIME
    }))
    const imported = hydrateWorkspaceBundleSnapshot(parsed.workspace.snapshot,
      "imported-large", "Imported large", "workspace:imported-large")
    const reexported = buildWorkspaceBundleSnapshot(imported)!

    expect(reexported.generatedArtifacts[0].content).toBe(expected.content)
    expect(reexported.generatedArtifacts[0].data).toEqual(expected.data)
    expect(reexported.generatedArtifacts[0].versionMetadata?.workspaceClient).not.toHaveProperty("data")
    expect(workspaceArtifactDraftPayload(reexported.generatedArtifacts[0])).toEqual(workspaceArtifactDraftPayload(expected))
    expect(reexported.generatedArtifacts[0].status).toBe("completed")
  })

  it("round-trips source folder organization through zip export/import", async () => {
    const bundle = createBundleFixture()
    bundle.workspace.snapshot.sourceFolders = [
      {
        id: "folder-1",
        workspaceId: "workspace-1",
        name: "Evidence",
        parentFolderId: null,
        createdAt: new Date("2026-02-18T09:35:00.000Z"),
        updatedAt: new Date("2026-02-18T09:35:00.000Z")
      }
    ]
    bundle.workspace.snapshot.sourceFolderMemberships = [
      {
        folderId: "folder-1",
        sourceId: "source-1"
      }
    ]
    bundle.workspace.snapshot.selectedSourceFolderIds = ["folder-1"]
    bundle.workspace.snapshot.activeFolderId = "folder-1"

    const zipBlob = await createWorkspaceExportZipBlob(bundle)
    const file = new File([zipBlob], "alpha.workspace.zip", {
      type: WORKSPACE_EXPORT_BUNDLE_ZIP_MIME
    })

    const parsed = await parseWorkspaceImportFile(file)

    expect(parsed.workspace.snapshot.sourceFolders).toEqual(
      toSerializableBundle(bundle).workspace.snapshot.sourceFolders
    )
    expect(parsed.workspace.snapshot.sourceFolderMemberships).toEqual(
      toSerializableBundle(bundle).workspace.snapshot.sourceFolderMemberships
    )
    expect(parsed.workspace.snapshot.selectedSourceFolderIds).toEqual(["folder-1"])
    expect(parsed.workspace.snapshot.activeFolderId).toBe("folder-1")
  })

  it("imports legacy JSON workspace bundles", async () => {
    const bundle = createBundleFixture()
    bundle.format = LEGACY_WORKSPACE_EXPORT_BUNDLE_FORMAT
    const file = new File([JSON.stringify(bundle)], "alpha.workspace.json", {
      type: "application/json"
    })

    const parsed = await parseWorkspaceImportFile(file)

    expect(parsed).toEqual(toSerializableBundle(bundle))
  })

  it("rejects ZIP bundles missing required manifest/payload files", async () => {
    const bundle = createBundleFixture()
    const zip = new JSZip()
    zip.file(
      WORKSPACE_EXPORT_BUNDLE_PAYLOAD_FILE,
      JSON.stringify(toSerializableBundle(bundle))
    )
    const blob = await zip.generateAsync({ type: "blob" })
    const file = new File([blob], "invalid.workspace.zip", {
      type: WORKSPACE_EXPORT_BUNDLE_ZIP_MIME
    })

    await expect(parseWorkspaceImportFile(file)).rejects.toThrow(
      "invalid-zip-bundle"
    )
  })

  it("rejects ZIP bundles with invalid manifest schema", async () => {
    const bundle = createBundleFixture()
    const zip = new JSZip()
    zip.file(WORKSPACE_EXPORT_BUNDLE_MANIFEST_FILE, JSON.stringify({ nope: true }))
    zip.file(
      WORKSPACE_EXPORT_BUNDLE_PAYLOAD_FILE,
      JSON.stringify(toSerializableBundle(bundle))
    )
    const blob = await zip.generateAsync({ type: "blob" })
    const file = new File([blob], "invalid-manifest.workspace.zip", {
      type: WORKSPACE_EXPORT_BUNDLE_ZIP_MIME
    })

    await expect(parseWorkspaceImportFile(file)).rejects.toThrow(
      "invalid-zip-manifest"
    )
  })

  it("rejects ZIP bundles when manifest and payload metadata mismatch", async () => {
    const bundle = createBundleFixture()
    const zip = new JSZip()
    zip.file(
      WORKSPACE_EXPORT_BUNDLE_MANIFEST_FILE,
      JSON.stringify({
        format: WORKSPACE_EXPORT_BUNDLE_FORMAT,
        schemaVersion: WORKSPACE_EXPORT_BUNDLE_SCHEMA_VERSION,
        exportedAt: bundle.exportedAt,
        workspace: {
          name: "Different Name",
          tag: bundle.workspace.tag,
          createdAt: toSerializableBundle(bundle).workspace.createdAt
        },
        payloadFile: WORKSPACE_EXPORT_BUNDLE_PAYLOAD_FILE
      })
    )
    zip.file(
      WORKSPACE_EXPORT_BUNDLE_PAYLOAD_FILE,
      JSON.stringify(toSerializableBundle(bundle))
    )
    const blob = await zip.generateAsync({ type: "blob" })
    const file = new File([blob], "mismatch.workspace.zip", {
      type: WORKSPACE_EXPORT_BUNDLE_ZIP_MIME
    })

    await expect(parseWorkspaceImportFile(file)).rejects.toThrow("schema-mismatch")
  })
})

it("export and reopen preserve all capture pin fields alongside original retrieved evidence", async () => {
  const bundle = createBundleFixture()
  const pin = {
    clipId: "clip",
    requestedUrl: "https://example.org",
    capturedAt: "2026-10-07T00:00:00Z",
    contentSha256: "digest",
    refreshOf: "prior",
    mediaId: 101,
    versionNumber: 9,
    versionUuid: "version-nine"
  }
  bundle.workspace.snapshot.sources[0].webCapture = pin
  const blob = await createWorkspaceExportZipBlob(bundle)
  const result = await parseWorkspaceImportFile(
    new File([blob], "capture.workspace.zip", {
      type: WORKSPACE_EXPORT_BUNDLE_ZIP_MIME
    })
  )
  expect(result.workspace.snapshot.sources[0].webCapture).toEqual(pin)
})
