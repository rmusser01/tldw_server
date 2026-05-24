import type { ChatHistory, Message } from "@/store/option"
import type {
  AudioGenerationSettings,
  GeneratedArtifact,
  WorkspaceBanner,
  WorkspaceCollection,
  WorkspaceNote,
  StudyMaterialsPolicy,
  WorkspaceSourceFolder,
  WorkspaceSourceFolderMembership,
  WorkspaceSource
} from "@/types/workspace"

export const WORKSPACE_EXPORT_BUNDLE_FORMAT = "tldw.research-workspace.bundle"
export const LEGACY_WORKSPACE_EXPORT_BUNDLE_FORMAT =
  "tldw.workspace-playground.bundle"
export const WORKSPACE_EXPORT_BUNDLE_SCHEMA_VERSION = 1
export const WORKSPACE_EXPORT_BUNDLE_PAYLOAD_FILE = "workspace.json"
export const WORKSPACE_EXPORT_BUNDLE_MANIFEST_FILE = "manifest.json"
export const WORKSPACE_EXPORT_BUNDLE_ZIP_MIME = "application/zip"

type ExportDateValue = string | Date | null

export interface WorkspaceBundleSnapshot {
  workspaceName: string
  workspaceTag: string
  workspaceCreatedAt: ExportDateValue
  studyMaterialsPolicy: StudyMaterialsPolicy | null
  sources: WorkspaceSource[]
  selectedSourceIds: string[]
  sourceFolders?: WorkspaceSourceFolder[]
  sourceFolderMemberships?: WorkspaceSourceFolderMembership[]
  selectedSourceFolderIds?: string[]
  activeFolderId?: string | null
  generatedArtifacts: GeneratedArtifact[]
  notes: string
  currentNote: WorkspaceNote
  workspaceBanner: WorkspaceBanner
  leftPaneCollapsed: boolean
  rightPaneCollapsed: boolean
  audioSettings: AudioGenerationSettings
}

export interface WorkspaceBundleChatSession {
  messages: Message[]
  history: ChatHistory
  historyId: string | null
  serverChatId: string | null
}

type WorkspaceExportBundleFormat =
  | typeof WORKSPACE_EXPORT_BUNDLE_FORMAT
  | typeof LEGACY_WORKSPACE_EXPORT_BUNDLE_FORMAT

export interface WorkspaceExportBundle {
  format: WorkspaceExportBundleFormat
  schemaVersion: typeof WORKSPACE_EXPORT_BUNDLE_SCHEMA_VERSION
  exportedAt: string
  workspace: {
    name: string
    tag: string
    createdAt: ExportDateValue
    studyMaterialsPolicy: StudyMaterialsPolicy | null
    collectionId?: WorkspaceCollection["id"] | null
    snapshot: WorkspaceBundleSnapshot
    chatSession?: WorkspaceBundleChatSession
  }
}

interface WorkspaceExportZipManifest {
  format: WorkspaceExportBundleFormat
  schemaVersion: typeof WORKSPACE_EXPORT_BUNDLE_SCHEMA_VERSION
  exportedAt: string
  workspace: {
    name: string
    tag: string
    createdAt: ExportDateValue
  }
  payloadFile: string
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null

export const isWorkspaceExportBundle = (
  value: unknown
): value is WorkspaceExportBundle => {
  if (!isRecord(value)) return false
  if (
    value.format !== WORKSPACE_EXPORT_BUNDLE_FORMAT &&
    value.format !== LEGACY_WORKSPACE_EXPORT_BUNDLE_FORMAT
  ) {
    return false
  }
  if (value.schemaVersion !== WORKSPACE_EXPORT_BUNDLE_SCHEMA_VERSION) return false
  if (typeof value.exportedAt !== "string") return false
  if (!isRecord(value.workspace)) return false
  if (!isRecord(value.workspace.snapshot)) return false
  return true
}

const sanitizeFilenameToken = (value: string): string =>
  value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")

export const createWorkspaceExportFilename = (
  workspaceName: string,
  exportedAt: string
): string => {
  const safeName = sanitizeFilenameToken(workspaceName) || "workspace"
  const safeDate =
    sanitizeFilenameToken(exportedAt.replace(/[.:]/g, "-")) || "export"
  return `${safeName}-${safeDate}.workspace.json`
}

export const createWorkspaceExportZipFilename = (
  workspaceName: string,
  exportedAt: string
): string => {
  const safeName = sanitizeFilenameToken(workspaceName) || "workspace"
  const safeDate =
    sanitizeFilenameToken(exportedAt.replace(/[.:]/g, "-")) || "export"
  return `${safeName}-${safeDate}.workspace.zip`
}

const parseBundleFromJson = (raw: string): WorkspaceExportBundle => {
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    throw new Error("invalid-json")
  }

  if (!isWorkspaceExportBundle(parsed)) {
    throw new Error("invalid-bundle-format")
  }

  return parsed
}

const createZipManifest = (
  bundle: WorkspaceExportBundle
): WorkspaceExportZipManifest => ({
  format: WORKSPACE_EXPORT_BUNDLE_FORMAT,
  schemaVersion: WORKSPACE_EXPORT_BUNDLE_SCHEMA_VERSION,
  exportedAt: bundle.exportedAt,
  workspace: {
    name: bundle.workspace.name,
    tag: bundle.workspace.tag,
    createdAt: bundle.workspace.createdAt
  },
  payloadFile: WORKSPACE_EXPORT_BUNDLE_PAYLOAD_FILE
})

const isValidZipManifest = (
  manifest: unknown
): manifest is WorkspaceExportZipManifest => {
  if (!isRecord(manifest)) return false
  if (
    manifest.format !== WORKSPACE_EXPORT_BUNDLE_FORMAT &&
    manifest.format !== LEGACY_WORKSPACE_EXPORT_BUNDLE_FORMAT
  ) {
    return false
  }
  if (manifest.schemaVersion !== WORKSPACE_EXPORT_BUNDLE_SCHEMA_VERSION) return false
  if (manifest.payloadFile !== WORKSPACE_EXPORT_BUNDLE_PAYLOAD_FILE) return false
  if (typeof manifest.exportedAt !== "string") return false
  if (!isRecord(manifest.workspace)) return false
  if (typeof manifest.workspace.name !== "string") return false
  if (typeof manifest.workspace.tag !== "string") return false
  if (
    manifest.workspace.createdAt !== null &&
    typeof manifest.workspace.createdAt !== "string"
  ) {
    return false
  }
  return true
}

const isZipFile = (file: File): boolean => {
  const normalizedName = file.name.toLowerCase()
  const normalizedType = (file.type || "").toLowerCase()
  return (
    normalizedName.endsWith(".workspace.zip") ||
    normalizedName.endsWith(".zip") ||
    normalizedType.includes("zip")
  )
}

export const createWorkspaceExportZipBlob = async (
  bundle: WorkspaceExportBundle
): Promise<Blob> => {
  const { default: JSZip } = await import("jszip")
  const zip = new JSZip()
  const manifest = createZipManifest(bundle)

  zip.file(
    WORKSPACE_EXPORT_BUNDLE_MANIFEST_FILE,
    JSON.stringify(manifest, null, 2)
  )
  zip.file(
    WORKSPACE_EXPORT_BUNDLE_PAYLOAD_FILE,
    JSON.stringify(bundle, null, 2)
  )

  return zip.generateAsync({
    type: "blob",
    compression: "DEFLATE",
    compressionOptions: {
      level: 6
    }
  })
}

const readFileAsArrayBuffer = async (file: File): Promise<ArrayBuffer> => {
  const fileCandidate = file as File & {
    arrayBuffer?: () => Promise<ArrayBuffer>
  }
  if (typeof fileCandidate.arrayBuffer === "function") {
    return fileCandidate.arrayBuffer()
  }

  if (typeof FileReader !== "undefined") {
    return new Promise<ArrayBuffer>((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => {
        if (reader.result instanceof ArrayBuffer) {
          resolve(reader.result)
          return
        }
        reject(new Error("invalid-zip-bundle"))
      }
      reader.onerror = () =>
        reject(reader.error || new Error("invalid-zip-bundle"))
      reader.readAsArrayBuffer(file)
    })
  }

  if (typeof Response !== "undefined") {
    return new Response(file).arrayBuffer()
  }

  throw new Error("invalid-zip-bundle")
}

/**
 * Strip serverChatId from an imported chat session to prevent
 * accidental reconnection to a server chat from a different scope.
 */
export const sanitizeImportedChatSession = <T extends Record<string, any>>(
  session: T
): Omit<T, "serverChatId"> & { serverChatId: null } => ({
  ...session,
  serverChatId: null,
})

export const parseWorkspaceImportFile = async (
  file: File
): Promise<WorkspaceExportBundle> => {
  if (!isZipFile(file)) {
    const raw = await file.text()
    return parseBundleFromJson(raw)
  }

  const { default: JSZip } = await import("jszip")
  const zip = await JSZip.loadAsync(await readFileAsArrayBuffer(file))
  const manifestFile = zip.file(WORKSPACE_EXPORT_BUNDLE_MANIFEST_FILE)
  const payloadFile = zip.file(WORKSPACE_EXPORT_BUNDLE_PAYLOAD_FILE)
  if (!manifestFile || !payloadFile) {
    throw new Error("invalid-zip-bundle")
  }

  const manifestRaw = await manifestFile.async("string")
  let manifest: unknown
  try {
    manifest = JSON.parse(manifestRaw) as unknown
  } catch {
    throw new Error("invalid-zip-manifest")
  }
  if (!isValidZipManifest(manifest)) {
    throw new Error("invalid-zip-manifest")
  }

  const bundleRaw = await payloadFile.async("string")
  const bundle = parseBundleFromJson(bundleRaw)
  if (
    bundle.format !== manifest.format ||
    bundle.schemaVersion !== manifest.schemaVersion ||
    bundle.exportedAt !== manifest.exportedAt ||
    bundle.workspace.name !== manifest.workspace.name ||
    bundle.workspace.tag !== manifest.workspace.tag ||
    bundle.workspace.createdAt !== manifest.workspace.createdAt
  ) {
    throw new Error("schema-mismatch")
  }

  return bundle
}
