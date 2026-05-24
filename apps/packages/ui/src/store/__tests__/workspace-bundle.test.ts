import { describe, expect, it } from "vitest"
import JSZip from "jszip"
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
    snapshot: {
      workspaceName: "Alpha Research",
      workspaceTag: "workspace:alpha-research",
      workspaceCreatedAt: new Date("2026-02-18T09:00:00.000Z"),
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
