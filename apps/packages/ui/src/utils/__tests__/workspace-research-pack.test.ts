import { Buffer } from "node:buffer"
import { webcrypto } from "node:crypto"
import JSZip from "jszip"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { portableFixture } from "./fixtures/portable-workspace"
import { buildCompleteResearchPack, parseCompleteResearchPack, readBlobBytes, readBoundedZipEntry, sha256Bytes } from "../workspace-research-pack"
import { parseWorkspaceImportFile } from "@/store/workspace-bundle"
import { projectWorkspaceRagSources, restoreWorkspaceRagSources } from "../workspace-rag-sources"
import { claimTextHash } from "../claim-verification"
import { remapPortableWorkspace } from "../workspace-reference-remap"

const pdf = Uint8Array.from(new TextEncoder().encode("%PDF-1.7 exact original bytes"))
const digest = webcrypto.subtle.digest.bind(webcrypto.subtle)
const chatbook = async (missing = false) => {
  const zip = new JSZip()
  zip.file("manifest.json", JSON.stringify({ content_items: [{ type: "media", id: "1", file_path: "content/media/media_1.json" }, { type: "note", id: "note-old", file_path: "content/notes/note_note-old.md" }] }))
  zip.file("content/media/media_1.json", JSON.stringify({ id: 1, uuid: "media-uuid", content: "Canonical text", stored_artifacts: [{ file_type: "original", archive_path: "content/media/files/original.pdf", mime_type: "application/pdf", file_size: pdf.length, checksum: await sha256Bytes(pdf), bundled: !missing, pointer_only: missing }] }))
  if (!missing) zip.file("content/media/files/original.pdf", pdf)
  zip.file("content/notes/note_note-old.md", "Reviewed text")
  return zip.generateAsync({ type: "uint8array" })
}

describe("complete research pack archive", () => {
  it("withholds historical report links after Media 1 becomes 2, retaining restored source navigation", async () => {
    const original = portableFixture()
    const draft = "Historical answer [1]"
    const report = { version: 1, text_sha256: claimTextHash(draft), verdict: "supported", provider: "openai", model: "saved-model",
      coverage: { total: 1, evaluated: 1, supported: 1, uncovered: 0, complete: true },
      findings: [{ id: "C1", text: draft, start: 0, end: draft.length, evidence_ids: ["1"], status: "supported", reason: "Saved evidence supports this." }],
      evidence: { "1": { text_sha256: "a".repeat(64), source_id: "source-old", media_id: 1, page_spans: [{ page_number: 999 }] } }, limits: {}, context: {} }
    original.workspace.chatSession!.messages[0].message = draft
    original.workspace.chatSession!.history = [{ role: "assistant", content: draft }]
    original.workspace.chatSession!.messages[0].metadataExtra = { claim_verification: report }
    const analysis = original.workspace.snapshot.generatedArtifacts[0].data!.sectionAnalysis as { synthesis?: string; synthesis_verification?: unknown; sections: Array<{ content: string; claim_verification?: unknown; batches: Array<{ verification_text?: string; claim_verification?: unknown }> }> }
    analysis.synthesis = analysis.sections[0].content = analysis.sections[0].batches[0].verification_text = draft
    analysis.synthesis_verification = analysis.sections[0].claim_verification = analysis.sections[0].batches[0].claim_verification = report
    const bundle = remapPortableWorkspace(original, { media: { "1": 2 }, notes: { "note-old": "note-old" } }, { id: "destination", name: "Imported", tag: "workspace:destination" })
    bundle.workspace.snapshot.sources.push({ id: "unrelated", mediaId: 1, title: "Unrelated destination source", type: "pdf", addedAt: new Date(0) })
    const archive = await JSZip.loadAsync(await chatbook())
    const manifest = JSON.parse(await archive.file("manifest.json")!.async("string"))
    manifest.content_items.push({ type: "media", id: "2", file_path: "content/media/media_2.json" })
    archive.file("manifest.json", JSON.stringify(manifest))
    archive.file("content/media/media_2.json", JSON.stringify({ ...JSON.parse(await archive.file("content/media/media_1.json")!.async("string")), id: 2 }))
    const blob = await buildCompleteResearchPack({ bundle, chatbook: await archive.generateAsync({ type: "uint8array" }), notes: [bundle.workspace.snapshot.currentNote], sourceWorkspaceId: "destination" })
    const zip = await JSZip.loadAsync(await readBlobBytes(blob))
    for (const path of ["offline/chat.html", "offline/reports/artifact-1.html", "offline/reports/artifact-1-section-1.html"]) {
      const doc = new DOMParser().parseFromString(await zip.file(path)!.async("string"), "text/html")
      expect(Array.from(doc.querySelectorAll("a")).some(link => link.href.includes("#page=999"))).toBe(false)
      expect(doc.body.textContent).toContain("Use the restored source citations to open evidence.")
      if (!path.endsWith("artifact-1.html")) expect(Array.from(doc.querySelectorAll("a")).some(link => link.getAttribute("href")?.endsWith("sources/media-2.pdf#page=352"))).toBe(true)
    }
    const parsed = await parseCompleteResearchPack(new File([await readBlobBytes(blob)], "reexport.zip"))
    expect(JSON.stringify(parsed!.bundle.workspace.chatSession!.messages[0].metadataExtra?.claim_verification)).toBe(JSON.stringify(report))
  })
  it.each(["text_or_unit_limit", "report_size_limit"])("retains full unchecked text and its %s notice in complete and readable exports", async reason => {
    const bundle = portableFixture()
    const draft = "Unchecked text [1]. 🌍\n".repeat(120)
    const report = { version: 1, text_sha256: claimTextHash(draft), verdict: "unverified", provider: "openai", model: "saved-model",
      coverage: { total: 1, evaluated: 0, supported: 0, uncovered: 1, complete: false },
      findings: [{ id: "tail", text: draft, start: 0, end: Array.from(draft).length, evidence_ids: [], status: "not_checked", reason }],
      evidence: {}, limits: {}, context: {} }
    bundle.workspace.chatSession!.messages[0].message = draft
    bundle.workspace.chatSession!.messages[0].metadataExtra = { claim_verification: report }
    bundle.workspace.chatSession!.history = [{ role: "assistant", content: draft }]
    const blob = await buildCompleteResearchPack({ bundle, chatbook: await chatbook(), notes: [bundle.workspace.snapshot.currentNote], sourceWorkspaceId: "old" })
    const zip = await JSZip.loadAsync(await readBlobBytes(blob))
    const restored = await parseCompleteResearchPack(new File([await readBlobBytes(blob)], "checked.zip"))
    expect(restored!.bundle.workspace.chatSession!.messages[0].metadataExtra?.claim_verification).toEqual(report)
    expect(await zip.file("offline/chat.md")!.async("string")).toContain(draft)
    const html = new DOMParser().parseFromString(await zip.file("offline/chat.html")!.async("string"), "text/html")
    expect(html.body.textContent).toContain(draft)
    expect(html.body.textContent).toContain(reason === "report_size_limit" ? "The citation report exceeded its size limit." : "This claim exceeded the check's text or claim limit")
    expect(html.body.textContent).toContain("Citation support: Not fully checked")
  })
  it("preserves citation findings and links in offline chapters, overview and conversation", async () => {
    const bundle = portableFixture()
    const report = (text: string) => ({ version: 1, text_sha256: claimTextHash(text), verdict: "needs_revision", provider: "openai", model: "saved-model",
      coverage: { total: 1, evaluated: 1, supported: 0, uncovered: 0, complete: true },
      findings: [{ id: "C1", text, start: 0, end: text.length, evidence_ids: ["1"], status: "unsupported", reason: "The cited source does not name this author <script>." }],
      evidence: { "1": { text_sha256: "a".repeat(64), source_id: "source-old", media_id: 1, page_spans: [{ page_number: 352 }] } }, limits: {}, context: {} })
    const artifact = bundle.workspace.snapshot.generatedArtifacts[0]
    const analysis = artifact.data!.sectionAnalysis as { synthesis: string; synthesis_verification?: unknown; sections: Array<{ content: string; claim_verification?: unknown; batches: Array<{ verification_text?: string; claim_verification?: unknown }> }> }
    analysis.sections[0].content += " [1]"
    analysis.synthesis = "Overview [1]"
    analysis.synthesis_verification = report(analysis.synthesis)
    analysis.sections[0].claim_verification = report(analysis.sections[0].content)
    analysis.sections[0].batches[0].verification_text = "Extracted claim [1]"
    analysis.sections[0].batches[0].claim_verification = report("Extracted claim [1]")
    const message = bundle.workspace.chatSession!.messages[0]
    message.metadataExtra = { claim_verification: report(message.message) }
    const original = JSON.stringify(bundle)
    const blob = await buildCompleteResearchPack({ bundle, chatbook: await chatbook(), notes: [bundle.workspace.snapshot.currentNote], sourceWorkspaceId: "old" })
    const zip = await JSZip.loadAsync(await readBlobBytes(blob))
    for (const path of ["offline/reports/artifact-1-section-1.html", "offline/reports/artifact-1.html", "offline/chat.html"]) {
      const document = new DOMParser().parseFromString(await zip.file(path)!.async("string"), "text/html")
      expect(document.body.textContent).toContain("Citation support: Needs review")
      expect(document.body.textContent).toContain("The cited source does not name this author <script>.")
      expect(document.querySelector("script")).toBeNull()
      expect(Array.from(document.querySelectorAll("a")).some(link => link.getAttribute("href")?.endsWith("sources/media-1.pdf#page=352"))).toBe(true)
    }
    expect(await zip.file("offline/reports/artifact-1-section-1.html")!.async("string")).toContain("Extracted claim [1]")
    expect(await zip.file("offline/chat.md")!.async("string")).toContain("Citation support: Needs review")
    expect(await zip.file("offline/reports/artifact-1.md")!.async("string")).toContain(artifact.content)
    expect(JSON.parse(await zip.file("workspace.json")!.async("string"))).toEqual(JSON.parse(original))
    expect(JSON.stringify(bundle)).toBe(original)
  })
  beforeEach(() => {
    Object.defineProperty(globalThis, "crypto", { value: webcrypto, configurable: true })
    // Node 20 rejects jsdom ArrayBuffers; bridge realms while retaining real hashing.
    vi.spyOn(webcrypto.subtle, "digest").mockImplementation((algorithm, bytes) =>
      digest(algorithm, Buffer.from(bytes as ArrayBuffer)))
  })

  it("hashes only the selected bytes with real SHA-256", async () => {
    const bytes = Uint8Array.from([0, 97, 98, 99, 255])
    expect(await sha256Bytes(bytes.subarray(1, 4))).toBe("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
  })

  it("retains originals, full >5MB artifact, Notes and metadata with usable offline page links", async () => {
    const bundle = portableFixture()
    bundle.workspace.snapshot.generatedArtifacts[0].content = "Complete text\n".repeat(400_000)
    const notes = [bundle.workspace.snapshot.currentNote]
    const blob = await buildCompleteResearchPack({ bundle, chatbook: await chatbook(), notes, sourceWorkspaceId: "old-workspace" })
    const parsed = await parseCompleteResearchPack(new File([await readBlobBytes(blob)], "complete.workspace.zip"))
    expect(parsed?.bundle).toEqual(JSON.parse(JSON.stringify(bundle)))
    expect(parsed?.notes).toEqual(JSON.parse(JSON.stringify(notes)))
    const zip = await JSZip.loadAsync(await readBlobBytes(blob))
    expect(await zip.file("offline/sources/media-1.pdf")!.async("uint8array")).toEqual(pdf)
    expect(await zip.file("offline/reports/artifact-1-section-1.html")!.async("string")).toContain('../sources/media-1.pdf#page=352')
    expect(await zip.file("offline/index.html")!.async("string")).toContain("Original PDF")
    expect(await zip.file("offline/notes/note-1.md")!.async("string")).toContain("Reviewed text")
  })

  it("distinguishes same-titled reports by readable output type and saved review status", async () => {
    const bundle = portableFixture()
    const summary = bundle.workspace.snapshot.generatedArtifacts[0]
    summary.title = "Project acceptance"
    summary.reviewStatus = "needs_revision"
    bundle.workspace.snapshot.generatedArtifacts.push({ ...summary, id: "comparison", type: "compare_sources", reviewStatus: "accepted" })
    const blob = await buildCompleteResearchPack({ bundle, chatbook: await chatbook(), notes: [bundle.workspace.snapshot.currentNote], sourceWorkspaceId: "old" })
    const zip = await JSZip.loadAsync(await readBlobBytes(blob))
    const index = new DOMParser().parseFromString(await zip.file("offline/index.html")!.async("string"), "text/html")
    const links = Array.from(index.querySelectorAll('a[href^="reports/"]'))
    expect(links.map(link => link.textContent)).toEqual(["Summary — Project acceptance", "Compare Sources — Project acceptance"])
    expect(links[0].parentElement?.textContent).toContain("Review: Needs revision")
    expect(links[1].parentElement?.textContent).toContain("Review: Accepted")
  })

  it("does not let the legacy snapshot importer bypass complete-pack identity restoration", async () => {
    const bundle = portableFixture()
    const blob = await buildCompleteResearchPack({ bundle, chatbook: await chatbook(), notes: [bundle.workspace.snapshot.currentNote], sourceWorkspaceId: "old-workspace" })
    await expect(parseWorkspaceImportFile(new File([await readBlobBytes(blob)], "complete.workspace.zip"))).rejects.toThrow(/verified restore/i)
  })

  it("fails complete export if the original PDF is only a server pointer", async () => {
    const bundle = portableFixture()
    await expect(buildCompleteResearchPack({ bundle, chatbook: await chatbook(true), notes: [bundle.workspace.snapshot.currentNote], sourceWorkspaceId: "old" })).rejects.toThrow(/original.*PDF/i)
  })

  it("refuses a derived PDF when the original is unavailable", async () => {
    const bundle = portableFixture()
    const archive = await JSZip.loadAsync(await chatbook(true))
    const media = JSON.parse(await archive.file("content/media/media_1.json")!.async("string"))
    media.stored_artifacts.push({ file_type: "derived", archive_path: "content/media/files/derived.pdf", mime_type: "application/pdf", bundled: true })
    archive.file("content/media/files/derived.pdf", "%PDF-derived")
    archive.file("content/media/media_1.json", JSON.stringify(media))
    await expect(buildCompleteResearchPack({ bundle, chatbook: await archive.generateAsync({ type: "uint8array" }), notes: [bundle.workspace.snapshot.currentNote], sourceWorkspaceId: "old" })).rejects.toThrow(/original.*PDF/i)
  })

  it.each([false, true])("links only the original PDF and retains derived attachments (derived first: %s)", async derivedFirst => {
    const bundle = portableFixture()
    const archive = await JSZip.loadAsync(await chatbook())
    const media = JSON.parse(await archive.file("content/media/media_1.json")!.async("string"))
    const derived = { file_type: "derived", archive_path: "content/media/files/derived.pdf", mime_type: "application/pdf", bundled: true }
    media.stored_artifacts = derivedFirst ? [derived, ...media.stored_artifacts] : [...media.stored_artifacts, derived]
    archive.file("content/media/files/derived.pdf", "%PDF-derived")
    archive.file("content/media/media_1.json", JSON.stringify(media))
    const blob = await buildCompleteResearchPack({ bundle, chatbook: await archive.generateAsync({ type: "uint8array" }), notes: [bundle.workspace.snapshot.currentNote], sourceWorkspaceId: "old" })
    const output = await JSZip.loadAsync(await readBlobBytes(blob))
    const originalPath = `offline/sources/media-1${derivedFirst ? "-2" : ""}.pdf`
    const attachmentPath = `offline/sources/media-1${derivedFirst ? "" : "-2"}.pdf`
    expect(await output.file(originalPath)!.async("uint8array")).toEqual(pdf)
    expect(await output.file(attachmentPath)!.async("string")).toBe("%PDF-derived")
    expect(await output.file("offline/reports/artifact-1-section-1.html")!.async("string")).toContain(`../${originalPath.replace("offline/", "")}#page=352`)
  })

  it("rejects tampered payload hashes before exposing a pack for network import", async () => {
    const bundle = portableFixture()
    const blob = await buildCompleteResearchPack({ bundle, chatbook: await chatbook(), notes: [bundle.workspace.snapshot.currentNote], sourceWorkspaceId: "old" })
    const zip = await JSZip.loadAsync(await readBlobBytes(blob))
    zip.file("workspace.json", "{}")
    await expect(parseCompleteResearchPack(new File([await zip.generateAsync({ type: "uint8array" })], "bad.zip"))).rejects.toThrow(/integrity/i)
  })

  it("rejects unsafe archive paths, and bounds actual expanded entry bytes", async () => {
    const zip = new JSZip().file("../escape.txt", "escape").file("research-pack.json", "{}")
    await expect(parseCompleteResearchPack(new File([await zip.generateAsync({ type: "uint8array" })], "bad.zip"))).rejects.toThrow(/path/i)
    const normal = await JSZip.loadAsync(await new JSZip().file("large", "x".repeat(200)).generateAsync({ type: "uint8array" }))
    await expect(readBoundedZipEntry(normal.file("large")!, 100)).rejects.toThrow(/size/i)
  })
  it("exports readable reviewer corrections alongside the unchanged draft", async () => {
    const bundle = portableFixture()
    const blob = await buildCompleteResearchPack({ bundle, chatbook: await chatbook(), notes: [bundle.workspace.snapshot.currentNote], sourceWorkspaceId: "old" })
    const zip = await JSZip.loadAsync(await readBlobBytes(blob))
    const html = await zip.file("offline/reports/artifact-1.html")!.async("string")
    expect(html).toContain("Reviewer corrections")
    expect(html).toContain("Review note: Source check")
    expect(await zip.file("offline/reports/artifact-1.md")!.async("string")).toContain("Reviewed correction")
  })

  it("rejects an embedded Chatbook with unselected content even when outer hashes match", async () => {
    const bundle = portableFixture()
    const blob = await buildCompleteResearchPack({ bundle, chatbook: await chatbook(), notes: [bundle.workspace.snapshot.currentNote], sourceWorkspaceId: "old" })
    const outer = await JSZip.loadAsync(await readBlobBytes(blob))
    const inner = await JSZip.loadAsync(await outer.file("chatbook.zip")!.async("uint8array"))
    const inventory = JSON.parse(await inner.file("manifest.json")!.async("string"))
    inventory.content_items.push({ type: "chat", id: "private-chat" })
    inner.file("manifest.json", JSON.stringify(inventory))
    const changed = await inner.generateAsync({ type: "uint8array" })
    outer.file("chatbook.zip", changed)
    const manifest = JSON.parse(await outer.file("research-pack.json")!.async("string"))
    Object.assign(manifest.files.find((file: { path: string }) => file.path === "chatbook.zip"), { bytes: changed.length, sha256: await sha256Bytes(changed) })
    outer.file("research-pack.json", JSON.stringify(manifest))
    await expect(parseCompleteResearchPack(new File([await outer.generateAsync({ type: "uint8array" })], "bad.zip"))).rejects.toThrow(/inventory|selection/i)
  })

  it("resolves offline chat citations to bundled originals", async () => {
    const bundle = portableFixture()
    const blob = await buildCompleteResearchPack({ bundle, chatbook: await chatbook(), notes: [bundle.workspace.snapshot.currentNote], sourceWorkspaceId: "old" })
    const zip = await JSZip.loadAsync(await readBlobBytes(blob))
    const html = await zip.file("offline/chat.html")!.async("string")
    const href = /href="([^"]+#page=352)"/.exec(html)![1]
    const target = new URL(href, "https://offline.test/offline/chat.html")
    expect(zip.file(target.pathname.slice(1))).not.toBeNull()
  })

  it.each(["live", "restored"])("links every PDF page in %s chat citations to the exact bundled original", async mode => {
    const bundle = portableFixture()
    const source = { name: "PDF", type: "pdf", pageContent: "Exact quote across pages", metadata: {
      media_id: 1, page_number: 871, citation_index: 6, chunk_id: "chunk-2843",
      page_spans: [
        { page_number: 871, start_char: 100, end_char: 120, printed_page_label: "838" },
        { page_number: 872, start_char: 122, end_char: 140, printed_page_label: "839" }
      ]
    } }
    bundle.workspace.chatSession!.messages[0].sources = mode === "live" ? [source]
      : restoreWorkspaceRagSources(JSON.parse(JSON.stringify(projectWorkspaceRagSources([source]))))
    const blob = await buildCompleteResearchPack({ bundle, chatbook: await chatbook(), notes: [bundle.workspace.snapshot.currentNote], sourceWorkspaceId: "old" })
    const zip = await JSZip.loadAsync(await readBlobBytes(blob))
    const document = new DOMParser().parseFromString(await zip.file("offline/chat.html")!.async("string"), "text/html")
    const links = Array.from(document.querySelectorAll("a"))
    expect(links.map(link => link.textContent)).toEqual(["PDF page 871 (printed 838)", "PDF page 872 (printed 839)"])
    expect(document.querySelector("blockquote")?.textContent).toBe(source.pageContent)
    for (const [index, link] of links.entries()) {
      const target = new URL(link.getAttribute("href")!, "https://offline.test/offline/chat.html")
      expect(target.hash).toBe(`#page=${871 + index}`)
      expect(await zip.file(target.pathname.slice(1))!.async("uint8array")).toEqual(pdf)
    }
    expect((await parseCompleteResearchPack(new File([await readBlobBytes(blob)], "complete.zip")))?.bundle).toEqual(JSON.parse(JSON.stringify(bundle)))
  })

  it("keeps legacy fallback for malformed chat spans without linking unbundled media or interpreting HTML", async () => {
    const bundle = portableFixture()
    const source = { name: "PDF", pageContent: "<script>untrusted excerpt</script>", metadata: {
      media_id: 1, page_number: 871, printed_page_label: '<img src=x onerror="alert(1)">',
      page_spans: [{ page_number: 872, start_char: 140, end_char: 100 }]
    } }
    bundle.workspace.chatSession!.messages[0].sources = [source, { ...source, metadata: {
      media_id: 999, page_number: 872,
      page_spans: [{ page_number: 872, start_char: 100, end_char: 140 }]
    } }]
    const blob = await buildCompleteResearchPack({ bundle, chatbook: await chatbook(), notes: [bundle.workspace.snapshot.currentNote], sourceWorkspaceId: "old" })
    const zip = await JSZip.loadAsync(await readBlobBytes(blob))
    const document = new DOMParser().parseFromString(await zip.file("offline/chat.html")!.async("string"), "text/html")
    expect(Array.from(document.querySelectorAll("a")).map(link => link.getAttribute("href"))).toEqual(["sources/media-1.pdf#page=871"])
    expect(document.querySelector("a")?.textContent).toBe(`PDF page 871 (printed ${source.metadata.printed_page_label})`)
    expect(document.querySelector("blockquote")?.textContent).toBe(source.pageContent)
    expect(document.querySelector("script, img")).toBeNull()
  })

  it.each(["file", "inventory"])("rejects nested account restoration via %s", async mode => {
    const bundle = portableFixture()
    const blob = await buildCompleteResearchPack({ bundle, chatbook: await chatbook(), notes: [bundle.workspace.snapshot.currentNote], sourceWorkspaceId: "old" })
    const outer = await JSZip.loadAsync(await readBlobBytes(blob))
    const inner = await JSZip.loadAsync(await outer.file("chatbook.zip")!.async("uint8array"))
    if (mode === "file") inner.file("json/account_settings.json", JSON.stringify({ overrides: { custom: "untrusted" } }))
    else {
      const inventory = JSON.parse(await inner.file("manifest.json")!.async("string"))
      inventory.account_inventory_summary = { counts: { account_profiles: 1, account_settings: 1 } }
      inner.file("manifest.json", JSON.stringify(inventory))
    }
    const changed = await inner.generateAsync({ type: "uint8array" })
    outer.file("chatbook.zip", changed)
    const manifest = JSON.parse(await outer.file("research-pack.json")!.async("string"))
    Object.assign(manifest.files.find((file: { path: string }) => file.path === "chatbook.zip"), { bytes: changed.length, sha256: await sha256Bytes(changed) })
    outer.file("research-pack.json", JSON.stringify(manifest))
    await expect(parseCompleteResearchPack(new File([await outer.generateAsync({ type: "uint8array" })], "bad.zip"))).rejects.toThrow(/account/i)
  })

})
