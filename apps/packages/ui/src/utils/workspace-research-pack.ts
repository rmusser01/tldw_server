import JSZip, { type JSZipObject } from "jszip"
import {
  createWorkspaceExportZipBlob,
  isWorkspaceExportBundle,
  type WorkspaceExportBundle
} from "@/store/workspace-bundle"
import { OUTPUT_TYPES, type WorkspaceNote } from "@/types/workspace"
import { renderSectionReviewExport } from "./section-artifact-review"
import { projectPdfPageSpans, stripKnowledgeNoteProvenance } from "./knowledge-note-provenance"
import { CLAIM_VERIFICATION_IMPORTED_KEY, CLAIM_VERIFICATION_KEY, readClaimVerification, renderClaimVerification } from "./claim-verification"

const PACK_FILE = "research-pack.json"
const MAX_ENTRY_BYTES = 128 * 1024 * 1024
const MAX_TOTAL_BYTES = 512 * 1024 * 1024
const MAX_FILES = 10_000
const record = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value)
type PackFile = { path: string; bytes: number; sha256: string }
export type ResearchPackManifest = {
  format: "tldw.complete-research-pack"
  schemaVersion: 1
  sourceWorkspaceId: string
  mediaIds: string[]
  noteIds: string[]
  files: PackFile[]
}
export type CompleteResearchPack = {
  manifest: ResearchPackManifest
  bundle: WorkspaceExportBundle
  notes: WorkspaceNote[]
  chatbook: Uint8Array
  fingerprint: string
}

export const readBlobBytes = async (blob: Blob): Promise<Uint8Array> => {
  if (typeof blob.arrayBuffer === "function") return new Uint8Array(await blob.arrayBuffer())
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(new Uint8Array(reader.result as ArrayBuffer))
    reader.onerror = () => reject(reader.error || new Error("Unable to read archive."))
    reader.readAsArrayBuffer(blob)
  })
}
export const sha256Bytes = async (bytes: Uint8Array): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes.slice().buffer)), value => value.toString(16).padStart(2, "0")).join("")

const safePath = (path: string) => Boolean(path && !path.startsWith("/") && !path.includes("\\") && !path.includes("\0") && !/^[A-Za-z]:/.test(path) && path.split("/").every(part => part !== "." && part !== ".."))
const declaredBytes = (entry: JSZipObject): number => {
  const size = (entry as JSZipObject & { _data?: { uncompressedSize?: number } })._data?.uncompressedSize
  if (!Number.isSafeInteger(size) || Number(size) < 0) throw new Error("Archive entry size is invalid.")
  return size as number
}
const validateZip = (zip: JSZip) => {
  let total = 0
  const names = Object.keys(zip.files)
  if (names.length > MAX_FILES) throw new Error("Research pack has too many files.")
  for (const entry of Object.values(zip.files)) {
    const original = (entry as JSZipObject & { unsafeOriginalName?: string }).unsafeOriginalName || entry.name
    if (!safePath(original) || original !== entry.name) throw new Error("Research pack contains an unsafe archive path.")
    if (entry.dir) continue
    const bytes = declaredBytes(entry)
    total += bytes
    if (bytes > MAX_ENTRY_BYTES || total > MAX_TOTAL_BYTES) throw new Error("Research pack expanded size exceeds the supported limit.")
  }
}

/** Bound actual streaming output as well as ZIP metadata, including forged declared sizes. */
export const readBoundedZipEntry = (entry: JSZipObject, limit = MAX_ENTRY_BYTES): Promise<Uint8Array> =>
  new Promise((resolve, reject) => {
    if (declaredBytes(entry) > limit) { reject(new Error("Archive entry size exceeds the supported limit.")); return }
    let size = 0
    const chunks: Uint8Array[] = []
    // JSZip exposes this streaming reader at runtime but omits it from JSZipObject's declarations.
    const stream = (entry as JSZipObject & { internalStream(type: "uint8array"): JSZip.JSZipStreamHelper<Uint8Array> }).internalStream("uint8array")
    let failed = false
    stream.on("data", (chunk: Uint8Array) => {
      if (failed) return
      size += chunk.length
      if (size > limit || size > declaredBytes(entry)) {
        failed = true
        stream.pause()
        reject(new Error("Archive entry expanded size is invalid."))
      } else chunks.push(chunk)
    }).on("error", error => { failed = true; reject(error) }).on("end", () => {
      if (failed) return
      const bytes = new Uint8Array(size)
      let offset = 0
      for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length }
      resolve(bytes)
    }).resume()
  })

const parseJson = (bytes: Uint8Array): unknown => JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes), (key, value) => {
  if (["__proto__", "prototype", "constructor"].includes(key)) throw new Error("Archive metadata has an unsafe object key.")
  return value
})
const jsonEntry = async (zip: JSZip, path: string): Promise<unknown> => {
  const entry = zip.file(path)
  if (!entry) throw new Error(`Research pack is missing ${path}.`)
  return parseJson(await readBoundedZipEntry(entry))
}
const escapeHtml = (value: unknown) => String(value ?? "").replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]!)
const htmlPage = (title: string, body: string) => `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${escapeHtml(title)}</title><style>body{font:18px/1.6 system-ui,sans-serif;max-width:78ch;margin:auto;padding:24px;color:#20242a;background:#fff}a{color:#174fb8}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}blockquote{margin:1em 0;padding:0 1em;border-left:1px solid #8b929b}summary{cursor:pointer}li{margin:.5em 0}</style></head><body><h1>${escapeHtml(title)}</h1>${body}</body></html>`
const textBlock = (text: unknown) => `<pre>${escapeHtml(text)}</pre>`
const positive = (value: unknown): value is number => Number.isSafeInteger(value) && Number(value) > 0

const validateChatbookInventory = async (zip: JSZip, mediaIds: string[], noteIds: string[]) => {
  const manifest = await jsonEntry(zip, "manifest.json")
  if (!record(manifest) || !Array.isArray(manifest.content_items) || manifest.content_items.some(item => !record(item))) throw new Error("Chatbook inventory is missing or invalid.")
  const counts = record(manifest.account_inventory_summary) && record(manifest.account_inventory_summary.counts) ? manifest.account_inventory_summary.counts : {}
  if (zip.file("json/account_profile.json") || zip.file("json/account_settings.json") ||
      ["account_profiles", "account_settings"].some(key => counts[key] != null && counts[key] !== 0))
    throw new Error("Research packs cannot restore account profiles or settings.")
  const items = manifest.content_items as Record<string, unknown>[]
  if (items.some(item => !["media", "note"].includes(String(item.type)))) throw new Error("Chatbook contains content outside this workspace selection.")
  for (const [kind, ids] of [["media", mediaIds], ["note", noteIds]] as const) {
    const actual = items.filter(item => item.type === kind).map(item => String(item.id))
    if (actual.length !== ids.length || new Set(actual).size !== actual.length || ids.some(id => !actual.includes(id))) throw new Error(`Chatbook ${kind} inventory does not match the complete research pack.`)
  }
  return items
}

export const buildCompleteResearchPack = async ({ bundle, chatbook, notes, sourceWorkspaceId }: {
  bundle: WorkspaceExportBundle
  chatbook: Uint8Array
  notes: WorkspaceNote[]
  sourceWorkspaceId: string
}): Promise<Blob> => {
  if (chatbook.byteLength > MAX_ENTRY_BYTES) throw new Error("Chatbook archive exceeds the supported size.")
  const archived = await JSZip.loadAsync(chatbook)
  validateZip(archived)
  const mediaIds = [...new Set(bundle.workspace.snapshot.sources.map(source => String(source.mediaId)))]
  const noteIds = [...new Set(notes.map(note => String(note.id)))]
  if (notes.some(note => note.id == null)) throw new Error("Saved Note identity is missing.")
  const items = await validateChatbookInventory(archived, mediaIds, noteIds)
  const base = await createWorkspaceExportZipBlob(bundle)
  const zip = await JSZip.loadAsync(await readBlobBytes(base))
  const outerManifest = await jsonEntry(zip, "manifest.json")
  zip.file("manifest.json", JSON.stringify({ ...outerManifest as object, researchPack: PACK_FILE }, null, 2))
  zip.file("chatbook.zip", chatbook)
  zip.file("notes.json", JSON.stringify(notes, null, 2))
  const sourcePaths = new Map<number, string>()
  const sourceLinks: string[] = []
  for (const source of bundle.workspace.snapshot.sources) {
    if (source.webCapture) throw new Error("Version-pinned web captures cannot yet be migrated by a complete research pack.")
    const item = items.find(item => item.type === "media" && String(item.id) === String(source.mediaId))!
    if (typeof item.file_path !== "string" || !safePath(item.file_path)) throw new Error("Chatbook Media path is invalid.")
    const media = await jsonEntry(archived, item.file_path)
    if (!record(media)) throw new Error("Chatbook Media record is invalid.")
    const files = Array.isArray(media.stored_artifacts) ? media.stored_artifacts.filter(record) : []
    let sourcePath: string | undefined
    for (const [index, file] of files.entries()) {
      if (!file.bundled || file.pointer_only || typeof file.archive_path !== "string" || !safePath(file.archive_path)) continue
      const entry = archived.file(file.archive_path)
      if (!entry) throw new Error("A bundled original source file is missing.")
      const bytes = await readBoundedZipEntry(entry)
      if ((typeof file.file_size === "number" && file.file_size !== bytes.length) || (typeof file.checksum === "string" && file.checksum && file.checksum !== await sha256Bytes(bytes))) throw new Error("Original source file integrity check failed.")
      const isPdf = file.mime_type === "application/pdf" || file.archive_path.toLowerCase().endsWith(".pdf")
      const extension = isPdf ? ".pdf" : /\.[A-Za-z0-9]{1,8}$/.exec(file.archive_path)?.[0] || ".bin"
      const path = `offline/sources/media-${source.mediaId}${index ? `-${index + 1}` : ""}${extension}`
      zip.file(path, bytes)
      if (!sourcePath && (source.type !== "pdf" || (isPdf && file.file_type === "original"))) sourcePath = path
    }
    if (source.type === "pdf" && (!sourcePath || !sourcePath.endsWith(".pdf"))) throw new Error(`The original PDF is unavailable for ${source.title}. Complete export was not created.`)
    if (!sourcePath) {
      if (typeof media.content !== "string" || !media.content.trim()) throw new Error(`Source content is unavailable for ${source.title}.`)
      sourcePath = `offline/sources/media-${source.mediaId}.txt`
      zip.file(sourcePath, media.content)
    }
    sourcePaths.set(source.mediaId, sourcePath.replace("offline/", ""))
    sourceLinks.push(`<li><a href="${escapeHtml(sourcePath.replace("offline/", ""))}">${escapeHtml(source.title)}</a></li>`)
  }
  const pageLinks = (mediaId: unknown, spans: unknown, prefix = "../") => {
    const path = sourcePaths.get(Number(mediaId))
    if (!path || !Array.isArray(spans)) return ""
    return spans.filter(record).filter(span => positive(span.page_number)).map(span => `<a href="${prefix}${escapeHtml(path)}#page=${span.page_number}">PDF page ${span.page_number}${span.printed_page_label ? ` (printed ${escapeHtml(span.printed_page_label)})` : ""}</a>`).join(", ")
  }
  const reportLinks: string[] = []
  const checksHtml = (value: unknown, text: unknown, imported = false, prefix = "../") => {
    if (typeof text !== "string" || !text.trim()) return ""
    const report = readClaimVerification(value, text)
    const links = Object.entries(imported ? {} : report?.evidence || {}).map(([id, evidence]) =>
      `<p>${escapeHtml(id)}: ${pageLinks(evidence.media_id, evidence.page_spans, prefix)}</p>`).join("")
    return `<h2>Citation support</h2>${textBlock(renderClaimVerification(value, text, imported))}${links}`
  }
  for (const [index, artifact] of bundle.workspace.snapshot.generatedArtifacts.entries()) {
    const stem = `artifact-${index + 1}`
    const imported = ["workspace_snapshot_import", "research_pack_import"].includes(String(artifact.producerMetadata?.origin))
    zip.file(`offline/reports/${stem}.md`, artifact.reviewMetadata ? renderSectionReviewExport(artifact) : artifact.content || "")
    const analysis = record(artifact.data?.sectionAnalysis) ? artifact.data.sectionAnalysis : null
    let body = textBlock(artifact.content || artifact.errorMessage || "No completed text is available.")
    if (analysis && Array.isArray(analysis.sections)) {
      const links: string[] = []
      for (const [sectionIndex, section] of analysis.sections.filter(record).entries()) {
        const path = `${stem}-section-${sectionIndex + 1}.html`
        const evidence = (Array.isArray(section.batches) ? section.batches.filter(record) : []).flatMap(batch => Array.isArray(batch.evidence) ? batch.evidence.filter(record) : [])
        const evidenceHtml = evidence.map(item => `<details><summary>${escapeHtml(item.id)} — ${escapeHtml(item.claim)}</summary><blockquote>${escapeHtml(item.quote)}</blockquote><p>${pageLinks(section.media_id, item.page_spans || section.page_spans)}</p>${textBlock([item.author, item.interpretation, item.limitations].filter(Boolean).join("\n"))}</details>`).join("")
        const batchChecks = (Array.isArray(section.batches) ? section.batches.filter(record) : []).map(batch =>
          checksHtml(batch.claim_verification, batch.verification_text, imported)).join("")
        const review = renderSectionReviewExport({ ...artifact, content: "" }, `section:${sectionIndex}`)
        zip.file(`offline/reports/${path}`, htmlPage(String(section.title || `Section ${sectionIndex + 1}`), `<p><a href="${stem}.html">Overview</a></p>${textBlock(section.content || section.error || "No completed summary.")}<p>${pageLinks(section.media_id, section.page_spans)}</p>${checksHtml(section.claim_verification, section.content, imported)}${review ? `<h2>Reviewer corrections</h2>${textBlock(review)}` : ""}<h2>Exact evidence</h2>${evidenceHtml}${batchChecks}`))
        links.push(`<li><a href="${path}">${escapeHtml(section.title || `Section ${sectionIndex + 1}`)}</a></li>`)
      }
      body = `${textBlock(analysis.synthesis || analysis.synthesis_error || "No overview is available.")}${checksHtml(analysis.synthesis_verification, analysis.synthesis, imported)}<h2>Chapters and sections</h2><ol>${links.join("")}</ol>`
    }
    if (artifact.reviewMetadata) body += `<h2>Reviewer corrections</h2>${textBlock(renderSectionReviewExport({ ...artifact, content: "" }))}<details><summary>Technical review record</summary>${textBlock(JSON.stringify(artifact.reviewMetadata, null, 2))}</details>`
    zip.file(`offline/reports/${stem}.html`, htmlPage(artifact.title, `<p><a href="../index.html">Research index</a> · <a href="${stem}.md">Full original Markdown</a></p>${body}`))
    const typeLabel = OUTPUT_TYPES.find(output => output.type === artifact.type)?.label || "Output"
    const reviewLabel = artifact.reviewStatus?.replace(/_/g, " ").replace(/^./, letter => letter.toUpperCase())
    reportLinks.push(`<li><a href="reports/${stem}.html">${escapeHtml(typeLabel)} — ${escapeHtml(artifact.title)}</a> — ${escapeHtml(artifact.status)}${reviewLabel ? ` · Review: ${escapeHtml(reviewLabel)}` : ""}</li>`)
  }
  const noteLinks = notes.map((note, index) => {
    zip.file(`offline/notes/note-${index + 1}.md`, note.content)
    zip.file(`offline/notes/note-${index + 1}.html`, htmlPage(note.title, textBlock(stripKnowledgeNoteProvenance(note.content))))
    return `<li><a href="notes/note-${index + 1}.html">${escapeHtml(note.title)}</a></li>`
  })
  if (bundle.workspace.snapshot.currentNote.isDirty) {
    zip.file("offline/current-note-draft.md", bundle.workspace.snapshot.currentNote.content)
    noteLinks.push('<li><a href="current-note-draft.md">Current unsaved Note draft</a></li>')
  }
  const messages = bundle.workspace.chatSession?.messages || []
  zip.file("offline/chat.md", messages.map(message => `## ${message.name || (message.isBot ? "Assistant" : "You")}\n\n${message.message}${message.isBot ? `\n\n${renderClaimVerification(message.metadataExtra?.[CLAIM_VERIFICATION_KEY], message.message, message.metadataExtra?.[CLAIM_VERIFICATION_IMPORTED_KEY] === true)}` : ""}`).join("\n\n"))
  zip.file("offline/chat.html", htmlPage("Saved conversation", messages.map(message => `<h2>${escapeHtml(message.name || (message.isBot ? "Assistant" : "You"))}</h2>${textBlock(message.message)}${(message.sources || []).filter(record).map(source => {
    const metadata = record(source.metadata) ? source.metadata : source
    const spans = projectPdfPageSpans(metadata.page_spans)?.map(span => ({
      page_number: span.pageNumber, printed_page_label: span.printedPageLabel
    })) || [{ page_number: metadata.page_number || metadata.pageNumber, printed_page_label: metadata.printed_page_label }]
    return `<blockquote>${escapeHtml(source.pageContent || source.content || "")}</blockquote><p>${pageLinks(metadata.media_id || metadata.mediaId, spans, "")}</p>`
  }).join("")}${message.isBot ? checksHtml(message.metadataExtra?.[CLAIM_VERIFICATION_KEY], message.message, message.metadataExtra?.[CLAIM_VERIFICATION_IMPORTED_KEY] === true, "") : ""}`).join("")))
  zip.file("offline/annotations.html", htmlPage("Source annotations", textBlock(JSON.stringify(bundle.workspace.snapshot.sourceAnnotations || {}, null, 2))))
  zip.file("offline/index.html", htmlPage(bundle.workspace.name, `<p>Offline copy. Generated drafts and recorded reviews are preserved separately. Editable migration requires the complete ZIP in Research Workspace.</p><h2>Original sources</h2><ul>${sourceLinks.join("")}</ul><h2>Reports</h2><ul>${reportLinks.join("")}</ul><h2>Saved Notes</h2><ul>${noteLinks.join("")}</ul><p><a href="chat.html">Saved conversation and citations</a> · <a href="chat.md">Conversation Markdown and checks</a> · <a href="annotations.html">Source annotations</a></p>`))
  const files: PackFile[] = []
  let total = 0
  for (const entry of Object.values(zip.files)) {
    if (entry.dir) continue
    const bytes = await entry.async("uint8array")
    total += bytes.length
    if (bytes.length > MAX_ENTRY_BYTES || total > MAX_TOTAL_BYTES) throw new Error("Complete research pack exceeds the supported size.")
    files.push({ path: entry.name, bytes: bytes.length, sha256: await sha256Bytes(bytes) })
  }
  const manifest: ResearchPackManifest = { format: "tldw.complete-research-pack", schemaVersion: 1, sourceWorkspaceId, mediaIds, noteIds, files }
  zip.file(PACK_FILE, JSON.stringify(manifest, null, 2))
  return zip.generateAsync({ type: "blob", compression: "DEFLATE", compressionOptions: { level: 6 } })
}

export const parseCompleteResearchPack = async (file: File): Promise<CompleteResearchPack | null> => {
  if (!file.name.toLowerCase().endsWith(".zip") && !file.type.includes("zip")) return null
  if (file.size > MAX_TOTAL_BYTES) throw new Error("Research pack size exceeds the supported limit.")
  const bytes = await readBlobBytes(file)
  const zip = await JSZip.loadAsync(bytes)
  validateZip(zip)
  if (!zip.file(PACK_FILE)) return null
  const manifest = await jsonEntry(zip, PACK_FILE)
  if (!record(manifest) || manifest.format !== "tldw.complete-research-pack" || manifest.schemaVersion !== 1 || typeof manifest.sourceWorkspaceId !== "string" || !Array.isArray(manifest.files) || !Array.isArray(manifest.mediaIds) || !Array.isArray(manifest.noteIds) || ![...manifest.mediaIds, ...manifest.noteIds].every(id => typeof id === "string" && id.length > 0)) throw new Error("Complete research pack manifest is invalid.")
  const paths = new Set<string>()
  for (const item of manifest.files) {
    if (!record(item) || typeof item.path !== "string" || !safePath(item.path) || paths.has(item.path) || !Number.isSafeInteger(item.bytes) || Number(item.bytes) < 0 || typeof item.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(item.sha256)) throw new Error("Research pack file inventory is invalid.")
    paths.add(item.path)
    const entry = zip.file(item.path)
    if (!entry || declaredBytes(entry) !== item.bytes || await sha256Bytes(await readBoundedZipEntry(entry, Number(item.bytes))) !== item.sha256) throw new Error("Research pack integrity verification failed.")
  }
  if (Object.values(zip.files).some(entry => !entry.dir && entry.name !== PACK_FILE && !paths.has(entry.name))) throw new Error("Research pack contains unlisted files.")
  for (const path of ["workspace.json", "chatbook.zip", "notes.json"]) if (!paths.has(path)) throw new Error(`Research pack is missing ${path}.`)
  const bundle = await jsonEntry(zip, "workspace.json")
  const notes = await jsonEntry(zip, "notes.json")
  if (!isWorkspaceExportBundle(bundle) || !Array.isArray(notes) || notes.some(note => !record(note) || typeof note.title !== "string" || typeof note.content !== "string" || note.id == null || !Array.isArray(note.keywords))) throw new Error("Research pack workspace or Notes are invalid.")
  const sourceIds = [...new Set(bundle.workspace.snapshot.sources.map(source => String(source.mediaId)))].sort()
  if (JSON.stringify(sourceIds) !== JSON.stringify([...manifest.mediaIds].sort()) || JSON.stringify(notes.map(note => String(note.id)).sort()) !== JSON.stringify([...manifest.noteIds].sort())) throw new Error("Research pack selected identities do not match its inventory.")
  const chatbook = await readBoundedZipEntry(zip.file("chatbook.zip")!)
  const archived = await JSZip.loadAsync(chatbook)
  validateZip(archived)
  await validateChatbookInventory(archived, manifest.mediaIds as string[], manifest.noteIds as string[])
  return { manifest: manifest as ResearchPackManifest, bundle, notes: notes as WorkspaceNote[], chatbook, fingerprint: await sha256Bytes(bytes) }
}
