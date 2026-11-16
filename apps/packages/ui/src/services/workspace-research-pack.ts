import { bgRequest, bgUpload, type BgRequestInit } from "./background-proxy"
import { tldwClient } from "./tldw/TldwApiClient"
import { requestScopeFields, servicePromptAuthorityKey } from "./tldw/domains/service-prompts"
import type { ServicePromptSnapshot } from "./service-prompts"
import { createWorkspaceExportZipFilename, type WorkspaceExportBundle } from "@/store/workspace-bundle"
import type { WorkspaceNote, GeneratedArtifact } from "@/types/workspace"
import { buildCompleteResearchPack, readBlobBytes, sha256Bytes, type CompleteResearchPack } from "@/utils/workspace-research-pack"
import { remapPortableWorkspace, remapPortableNote, type PortableWorkspaceIdentity, type VerifiedResearchImportMaps } from "@/utils/workspace-reference-remap"
import { knowledgeNoteHead, knowledgeNoteWriteFields, retainKnowledgeNoteProvenance, resolveKnowledgeNoteProvenance } from "@/utils/knowledge-note-provenance"
import { bindWorkspaceArtifactOrigin } from "@/store/workspace-artifact-codec"
import { persistWorkspaceArtifact, rememberCanonicalArtifactWorkspace } from "./workspace-artifact-persistence"
import { prepareLegacyWorkspaceImport } from "@/store/workspace-import-authority"
import type { WorkspaceSourceApiResponse } from "./tldw/domains/workspace-api"
import { bindImportedWorkspaceChat, buildWorkspaceChatImport, type WorkspaceChatImport } from "@/utils/workspace-chat-import"
import { canonicalHistoryJson } from "@/db/dexie/history-selection"
import { fetchAllServerChatMessages, mapServerChatMessagesToPlaygroundMessages } from "@/hooks/chat/useServerChatLoader"

type PackOperation = {
  scope: ServicePromptSnapshot
  isCurrent: () => boolean
  onProgress?: (message: string) => void
  pollIntervalMs?: number
}
type ImportReceipt = {
  version: 1
  fingerprint: string
  identity: PortableWorkspaceIdentity
  phase: "submitting" | "restoring" | "enrolling" | "ready"
  jobId?: string
  maps?: VerifiedResearchImportMaps
  noteBaseVersions?: Record<string, number>
}
const record = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === "object" && !Array.isArray(value)
const assertCurrent = (options: PackOperation) => {
  if (!options.isCurrent() || options.scope.scopeSignal.aborted || options.scope.scopeInvalidatedSignal.aborted)
    throw new Error("The active account or workspace changed. The research pack operation stopped.")
}
const scoped = (options: PackOperation) => {
  assertCurrent(options)
  return { ...requestScopeFields(options.scope.requestScope), abortSignal: AbortSignal.any([options.scope.scopeSignal, options.scope.scopeInvalidatedSignal]), timeoutMs: 120_000 }
}
const request = async <T,>(options: PackOperation, init: BgRequestInit): Promise<T> => {
  const fields = scoped(options)
  const value = await bgRequest<T>({ ...init, ...fields, headers: { ...init.headers, ...fields.headers } })
  assertCurrent(options)
  return value
}
const pause = (options: PackOperation) => new Promise<void>((resolve, reject) => {
  assertCurrent(options)
  const signal = AbortSignal.any([options.scope.scopeSignal, options.scope.scopeInvalidatedSignal])
  const abort = () => { clearTimeout(timer); reject(new Error("Research pack operation was interrupted.")) }
  const timer = setTimeout(() => { signal.removeEventListener("abort", abort); resolve() }, options.pollIntervalMs ?? 1200)
  signal.addEventListener("abort", abort, { once: true })
})
const pollJob = async (kind: "export" | "import", id: string, options: PackOperation) => {
  const deadline = Date.now() + 30 * 60_000
  while (Date.now() < deadline) {
    const value = await request<Record<string, unknown>>(options, { path: `/api/v1/chatbooks/${kind}/jobs/${encodeURIComponent(id)}`, method: "GET" })
    if (["completed", "failed", "cancelled", "canceled", "expired"].includes(String(value.status))) return value
    options.onProgress?.(`${kind === "export" ? "Preparing complete pack" : "Restoring sources and Notes"}${typeof value.progress_percentage === "number" ? ` (${value.progress_percentage}%)` : ""}…`)
    await pause(options)
  }
  throw new Error(kind === "import"
    ? "The import job is still running. Its receipt is retained; reopen this file to check progress."
    : `Export job ${id} is still running. Check its progress in Chatbooks before starting another export.`)
}
const keywords = (value: unknown): string[] => Array.isArray(value) ? value.flatMap(item => typeof item === "string" ? [item] : record(item) && typeof (item.keyword ?? item.name) === "string" ? [String(item.keyword ?? item.name)] : []) : []
const portableNote = (value: unknown): WorkspaceNote => {
  if (!record(value) || value.id == null || typeof value.title !== "string" || typeof value.content !== "string") throw new Error("A saved Note is unavailable or malformed.")
  return { id: String(value.id), title: value.title, content: retainKnowledgeNoteProvenance(value.content, value), keywords: keywords(value.keywords ?? (record(value.metadata) ? value.metadata.keywords : undefined)), ...(typeof value.version === "number" ? { version: value.version } : {}), ...knowledgeNoteHead(value, value.content), isDirty: false }
}

/** Exact keyword pagination, plus the current saved Note; never broad account enumeration. */
export const collectWorkspacePackNotes = async (bundle: WorkspaceExportBundle, workspaceId: string, options: PackOperation): Promise<WorkspaceNote[]> => {
  const notes = new Map<string, WorkspaceNote>()
  const tokens = [...new Set([bundle.workspace.tag, `workspace:${workspaceId}`].filter(Boolean))]
  let offset = 0
  for (let page = 0; page < 1000; page++) {
    const params = new URLSearchParams({ limit: "100", offset: String(offset), include_keywords: "true", tokens_match: "any", legacy_aliases: "false" })
    tokens.forEach(token => params.append("tokens", token))
    const response = await request<Record<string, unknown>>(options, { path: `/api/v1/notes/search/?${params.toString()}`, method: "GET" })
    const rows = Array.isArray(response.items) ? response.items : Array.isArray(response.notes) ? response.notes : Array.isArray(response.results) ? response.results : null
    if (!rows) throw new Error("Workspace Notes inventory is unavailable.")
    for (const row of rows) { const note = portableNote(row); notes.set(String(note.id), note) }
    const pagination = record(response.pagination) ? response.pagination : {}
    const hasMore = typeof pagination.has_more === "boolean" ? pagination.has_more : typeof response.total === "number" ? offset + rows.length < response.total : rows.length === 100
    if (!hasMore) break
    const next = typeof pagination.next_offset === "number" ? pagination.next_offset : offset + rows.length
    if (!rows.length || next <= offset || page === 999) throw new Error("Workspace Notes pagination did not complete; no incomplete pack was exported.")
    offset = next
  }
  const currentId = bundle.workspace.snapshot.currentNote.id
  if (currentId != null) {
    const note = portableNote(await request(options, { path: `/api/v1/notes/${encodeURIComponent(String(currentId))}`, method: "GET" }))
    if (String(note.id) !== String(currentId)) throw new Error("Current Note identity changed during export.")
    notes.set(String(note.id), note)
  }
  return [...notes.values()]
}

const validateLinkedInventory = (bundle: WorkspaceExportBundle, notes: WorkspaceNote[], identity: PortableWorkspaceIdentity) => {
  const maps = { media: Object.fromEntries(bundle.workspace.snapshot.sources.map(source => [String(source.mediaId), source.mediaId])), notes: Object.fromEntries(notes.map(note => [String(note.id), String(note.id)])) }
  // Identity maps here only validate inventory closure; they are never used to restore a workspace.
  const remapped = remapPortableWorkspace(bundle, maps, identity)
  notes.forEach(note => remapPortableNote(note, maps, identity))
  return remapped
}

export const exportWorkspaceResearchPack = async (bundle: WorkspaceExportBundle, workspaceId: string, options: PackOperation): Promise<Blob> => {
  assertCurrent(options)
  options.onProgress?.("Collecting all workspace Notes…")
  const notes = await collectWorkspacePackNotes(bundle, workspaceId, options)
  await buildWorkspaceChatImport(validateLinkedInventory(bundle, notes, { id: workspaceId, name: bundle.workspace.name, tag: bundle.workspace.tag }), workspaceId)
  assertCurrent(options)
  const media = [...new Set(bundle.workspace.snapshot.sources.map(source => String(source.mediaId)))]
  if (!media.length && !notes.length) throw new Error("Add a saved source or Note before exporting a complete research pack.")
  options.onProgress?.("Preparing original files and saved Notes…")
  const backupName = createWorkspaceExportZipFilename(bundle.workspace.name, bundle.exportedAt).slice(0, 255)
  const started = await tldwClient.exportChatbook({ name: backupName, description: "Selected Research Workspace sources and saved Notes", content_selections: { media, note: notes.map(note => String(note.id)) }, include_media: true, media_quality: "original", include_embeddings: false, include_generated_content: false, format_version: "1.1.0", async_mode: true }, { requestScope: options.scope.requestScope, signal: scoped(options).abortSignal })
  assertCurrent(options)
  if (!started.success || typeof started.job_id !== "string") throw new Error("The source backup job was not accepted. No incomplete pack was exported.")
  const status = await pollJob("export", started.job_id, options)
  if (status.status !== "completed") throw new Error("The source backup failed. No incomplete pack was exported.")
  const downloaded = await tldwClient.downloadChatbookExport(started.job_id, { requestScope: options.scope.requestScope, signal: scoped(options).abortSignal })
  assertCurrent(options)
  options.onProgress?.("Verifying original files and preparing offline reading…")
  const pack = await buildCompleteResearchPack({ bundle, notes, sourceWorkspaceId: workspaceId, chatbook: await readBlobBytes(downloaded.blob) })
  assertCurrent(options)
  return pack
}

const receiptKey = (pack: CompleteResearchPack, options: PackOperation) => `tldw:research-pack-import:v1:${servicePromptAuthorityKey(options.scope.requestScope)}:${pack.fingerprint}`
const saveReceipt = (key: string, receipt: ImportReceipt) => sessionStorage.setItem(key, JSON.stringify(receipt))
const readReceipt = (key: string, pack: CompleteResearchPack): ImportReceipt | null => {
  const raw = sessionStorage.getItem(key)
  if (!raw) return null
  const value: unknown = JSON.parse(raw)
  if (!record(value) || value.version !== 1 || value.fingerprint !== pack.fingerprint || !record(value.identity) || typeof value.identity.id !== "string" || typeof value.identity.name !== "string" || typeof value.identity.tag !== "string" || !["submitting", "restoring", "enrolling", "ready"].includes(String(value.phase)) || (value.jobId != null && typeof value.jobId !== "string")) throw new Error("The saved import receipt is invalid. Inspect this import in Chatbooks before retrying.")
  return value as ImportReceipt
}

export const restoreWorkspaceResearchPack = async (pack: CompleteResearchPack, options: PackOperation): Promise<{ bundle: WorkspaceExportBundle; identity: PortableWorkspaceIdentity }> => {
  assertCurrent(options)
  if (pack.chatbook.byteLength > 100 * 1024 * 1024) throw new Error("The embedded source backup exceeds the server's 100 MiB Chatbook upload limit. Split the workspace into smaller research packs.")
  await buildWorkspaceChatImport(validateLinkedInventory(pack.bundle, pack.notes, { id: pack.manifest.sourceWorkspaceId, name: pack.bundle.workspace.name, tag: pack.bundle.workspace.tag }), pack.manifest.sourceWorkspaceId)
  assertCurrent(options)
  const key = receiptKey(pack, options)
  let receipt = readReceipt(key, pack)
  if (!receipt) {
    const id = crypto.randomUUID()
    receipt = { version: 1, fingerprint: pack.fingerprint, identity: { id, name: `${pack.bundle.workspace.name} (Imported)`, tag: `workspace:${id}` }, phase: "submitting" }
    // Store the intent before the unsafe request; a lost response must not cause blind duplicate uploads.
    saveReceipt(key, receipt)
    options.onProgress?.("Uploading original sources and saved Notes…")
    const accepted = await bgUpload<Record<string, unknown>>({ ...scoped(options), path: "/api/v1/chatbooks/import", method: "POST", fileFieldName: "file", fields: { conflict_resolution: "rename", prefix_imported: "false", import_media: "true", import_embeddings: "false", async_mode: "true", source_format: "chatbook", content_selections: JSON.stringify({ media: pack.manifest.mediaIds, note: pack.manifest.noteIds }) }, file: { name: `research-pack-${pack.fingerprint}.zip`, type: "application/zip", data: pack.chatbook.slice().buffer } })
    if (typeof accepted.job_id === "string") { receipt.jobId = accepted.job_id; receipt.phase = "restoring"; saveReceipt(key, receipt) }
    assertCurrent(options)
    if (!receipt.jobId) throw new Error("Import submission is uncertain. Its receipt is retained; inspect Chatbooks before submitting again.")
  }
  if (!receipt.jobId) throw new Error("A previous import submission has no confirmed job ID. Inspect Chatbooks before submitting again; no duplicate upload was sent.")
  if (receipt.phase === "restoring" || !receipt.maps) {
    options.onProgress?.(`Restoring sources and Notes (job ${receipt.jobId})…`)
    const status = await pollJob("import", receipt.jobId, options)
    if (record(status.id_maps)) { receipt.maps = status.id_maps as VerifiedResearchImportMaps; saveReceipt(key, receipt) }
    if (status.status !== "completed" || Number(status.failed_items || 0) > 0) throw new Error(`Import ${receipt.jobId} ${String(status.status)}. Partial restored items and their receipt are retained in Chatbooks; the current workspace has not changed.`)
  }
  if (!receipt.maps) throw new Error("This server did not return verified restore mappings. The imported items remain in Chatbooks; the current workspace has not changed.")
  let sequence = 0
  const bundle = remapPortableWorkspace(pack.bundle, receipt.maps, receipt.identity, () => `portable-${receipt!.identity.id}-${++sequence}`)
  const chatImport = await buildWorkspaceChatImport(bundle, receipt.identity.id)
  assertCurrent(options)
  receipt.phase = "enrolling"
  saveReceipt(key, receipt)
  options.onProgress?.("Connecting the restored workspace and preserving editable reports…")
  await enrollWorkspaceSources(bundle, receipt.identity, options)
  for (const saved of pack.notes) {
    const note = remapPortableNote(saved, receipt.maps, receipt.identity)
    const path = `/api/v1/notes/${encodeURIComponent(String(note.id))}` as const
    const current = await request<Record<string, unknown>>(options, { path, method: "GET" })
    if (String(current.id) !== String(note.id) || !Number.isSafeInteger(current.version)) throw new Error("Restored Note identity or version is unavailable.")
    const desiredProvenance = resolveKnowledgeNoteProvenance(note, note.content).provenance
    const payload = { title: note.title, content: note.content, keywords: note.keywords, ...knowledgeNoteWriteFields(note.content, current, { ...(desiredProvenance ? { replacement: desiredProvenance } : {}) }) }
    const currentKeywords = keywords(current.keywords)
    const unchanged = current.title === payload.title && current.content === payload.content && JSON.stringify([...currentKeywords].sort()) === JSON.stringify([...note.keywords].sort()) && !("knowledge_provenance" in payload)
    receipt.noteBaseVersions ??= {}
    const initialVersion = receipt.noteBaseVersions[String(note.id)]
    if (!unchanged && initialVersion !== undefined && current.version !== initialVersion)
      throw new Error("A restored Note was edited after this import began. Its newer content is preserved; resolve the conflict before resuming.")
    if (initialVersion === undefined) {
      receipt.noteBaseVersions[String(note.id)] = Number(current.version)
      saveReceipt(key, receipt)
    }
    const accepted = unchanged ? current : await request<Record<string, unknown>>(options, { path, method: "PUT", headers: { "Content-Type": "application/json", "expected-version": String(current.version), "Idempotency-Key": `research-pack-${pack.fingerprint.slice(0, 24)}-${note.id}` }, body: payload })
    const resolved = record(accepted.note) ? accepted.note : accepted
    if (String(resolved.id) !== String(note.id) || !Number.isSafeInteger(resolved.version)) throw new Error("Restored Note save was not acknowledged. The restore receipt is retained.")
    if (String(bundle.workspace.snapshot.currentNote.id) === String(note.id)) {
      bundle.workspace.snapshot.currentNote = { ...bundle.workspace.snapshot.currentNote, ...knowledgeNoteHead(resolved, note.content), version: Number(resolved.version) }
    }
  }
  await enrollWorkspaceArtifacts(bundle, receipt.identity, options)
  await enrollWorkspaceChat(bundle, chatImport, options)
  receipt.phase = "ready"
  saveReceipt(key, receipt)
  return { bundle, identity: receipt.identity }
}

const enrollWorkspaceSources = async (bundle: WorkspaceExportBundle, identity: PortableWorkspaceIdentity, options: PackOperation) => {
  const destination = `/api/v1/workspaces/${encodeURIComponent(identity.id)}` as const
  const workspace = await request<Record<string, unknown>>(options, { path: destination, method: "PUT", headers: { "Content-Type": "application/json" }, body: { name: identity.name, study_materials_policy: "workspace", workspace_profile: "research" } })
  if (workspace.id !== identity.id) throw new Error("Destination workspace identity did not match the restore receipt.")
  const sourceRows = await request<WorkspaceSourceApiResponse[]>(options, { path: `${destination}/sources`, method: "GET" })
  if (!Array.isArray(sourceRows)) throw new Error("Restored workspace source inventory is unavailable.")
  for (const [position, source] of bundle.workspace.snapshot.sources.entries()) {
    const existing = sourceRows.find(row => row.id === source.id)
    if (existing && existing.media_id !== source.mediaId) throw new Error("Restored source identity conflicts with an existing destination source.")
    if (!existing) {
      const added = await request<WorkspaceSourceApiResponse>(options, { path: `${destination}/sources`, method: "POST", headers: { "Content-Type": "application/json" }, body: { id: source.id, media_id: source.mediaId, title: source.title, source_type: source.type, url: source.url || null, selected: bundle.workspace.snapshot.selectedSourceIds.includes(source.id), position } })
      if (added.id !== source.id || added.media_id !== source.mediaId) throw new Error("Restored source acknowledgement did not match the verified mapping.")
    }
  }
}
const enrollWorkspaceArtifacts = async (bundle: WorkspaceExportBundle, identity: PortableWorkspaceIdentity, options: PackOperation) => {
  rememberCanonicalArtifactWorkspace(identity.id, options.scope.scopeKey)
  const artifacts: GeneratedArtifact[] = []
  for (const artifact of bundle.workspace.snapshot.generatedArtifacts) {
    assertCurrent(options)
    artifacts.push(await persistWorkspaceArtifact(bindWorkspaceArtifactOrigin(artifact, options.scope.scopeKey, identity.id), { workspaceId: identity.id, scope: options.scope, isCurrent: options.isCurrent, deadline: performance.now() + 120_000 }))
  }
  bundle.workspace.snapshot.generatedArtifacts = artifacts
  assertCurrent(options)
}

const enrollWorkspaceChat = async (bundle: WorkspaceExportBundle, payload: WorkspaceChatImport | null, options: PackOperation) => {
  if (!payload) return
  options.onProgress?.("Saving the complete cited conversation…")
  const head = await request<Record<string, unknown>>(options, { path: "/api/v1/chats/import", method: "POST", headers: { "Content-Type": "application/json" }, body: payload })
  if (head.id !== payload.id || head.scope_type !== "workspace" || head.workspace_id !== payload.workspace_id)
    throw new Error("Imported conversation scope did not match the destination workspace.")
  const signal = scoped(options).abortSignal
  const rows = await fetchAllServerChatMessages(({ limit, offset }) => tldwClient.listChatMessages(payload.id, {
    limit, offset, include_metadata: "true", include_images: "true", include_deleted: "false", render_placeholders: "false",
  }, { signal, fresh: true, requestScope: options.scope.requestScope, scope: { type: "workspace", workspaceId: payload.workspace_id } }), { signal })
  assertCurrent(options)
  const expected = new Map(payload.messages.map(row => [row.id, row]))
  if (rows.length !== expected.size || new Set(rows.map(row => row.id)).size !== rows.length)
    throw new Error("The canonical imported conversation changed. Its saved copy is retained; the current workspace has not changed.")
  for (const row of rows) {
    const requestRow = expected.get(String(row.id))
    const expectedExtra = { ...requestRow?.metadata, ...requestRow?.metadata_extra }
    const { metadata: _generation, metadata_extra: _citations, ...requestFields } = requestRow || {}
    const expectedRow = { ...requestFields, ...(Object.keys(expectedExtra).length ? { metadata_extra: expectedExtra } : {}) }
    const timestamp = Date.parse(row.created_at)
    const actualExtra = { ...row.metadata_extra }
    // Atomic chat import records this redundant role alongside user metadata.
    // Validate it before excluding only that server-derived field from equality.
    if (actualExtra.sender_role !== undefined && actualExtra.sender_role !== row.role)
      throw new Error("The canonical imported conversation has inconsistent sender metadata.")
    delete actualExtra.sender_role
    const actual = { id: String(row.id), parent_message_id: row.parent_message_id ?? null, role: row.role,
      content: row.content, timestamp: Number.isFinite(timestamp) ? new Date(timestamp).toISOString() : null,
      images: row.images || [], ...(Object.keys(actualExtra).length ? { metadata_extra: actualExtra } : {}) }
    if (!requestRow || canonicalHistoryJson(actual) !== canonicalHistoryJson(expectedRow) ||
        !Number.isSafeInteger(row.version) || Number(row.version) < 1)
      throw new Error("The canonical imported conversation did not match its messages and citation evidence.")
  }
  const canonical = mapServerChatMessagesToPlaygroundMessages({ serverMessages: rows, assistantName: "AI", characterId: null })
  const originals = new Map(bundle.workspace.chatSession!.messages.map(row => [row.id, row]))
  bundle.workspace.chatSession = bindImportedWorkspaceChat({ ...bundle.workspace.chatSession!, historyId: null, serverChatId: payload.id,
    messages: canonical.map(row => ({ ...originals.get(row.id), ...row })),
    history: canonical.filter(row => row.message).map(row => ({ role: row.role!, content: row.message })) }, options.scope.scopeKey, payload.workspace_id)
}

/** Same-server snapshots keep Media references, but create new workspace/report authority. */
export const restoreLegacyWorkspaceBundle = async (original: WorkspaceExportBundle, options: PackOperation): Promise<{ bundle: WorkspaceExportBundle; identity: PortableWorkspaceIdentity }> => {
  assertCurrent(options)
  const fingerprint = await sha256Bytes(new TextEncoder().encode(JSON.stringify(original)))
  assertCurrent(options)
  const key = `tldw:workspace-snapshot-import:v1:${servicePromptAuthorityKey(options.scope.requestScope)}:${fingerprint}`
  const existing = sessionStorage.getItem(key)
  let identity: PortableWorkspaceIdentity
  if (existing) {
    const parsed: unknown = JSON.parse(existing)
    if (!record(parsed) || typeof parsed.id !== "string" || typeof parsed.name !== "string" || typeof parsed.tag !== "string") throw new Error("The saved workspace import receipt is invalid.")
    identity = parsed as PortableWorkspaceIdentity
  } else {
    const id = crypto.randomUUID()
    identity = { id, name: `${original.workspace.name} (Imported)`, tag: `workspace:${id}` }
    sessionStorage.setItem(key, JSON.stringify(identity))
  }
  let sequence = 0
  const bundle = prepareLegacyWorkspaceImport(original, identity, () => `snapshot-${identity.id}-${++sequence}`)
  const chatImport = await buildWorkspaceChatImport(bundle, identity.id)
  assertCurrent(options)
  options.onProgress?.("Saving the imported workspace and editable reports…")
  await enrollWorkspaceSources(bundle, identity, options)
  await enrollWorkspaceArtifacts(bundle, identity, options)
  await enrollWorkspaceChat(bundle, chatImport, options)
  return { bundle, identity }
}
