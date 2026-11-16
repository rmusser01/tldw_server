import { webcrypto } from "node:crypto"
import JSZip from "jszip"
import { beforeEach, describe, expect, it } from "vitest"
import { portableFixture } from "./fixtures/portable-workspace"
import { buildCompleteResearchPack, parseCompleteResearchPack, readBlobBytes, readBoundedZipEntry, sha256Bytes } from "../workspace-research-pack"
import { parseWorkspaceImportFile } from "@/store/workspace-bundle"

const pdf = Uint8Array.from(new TextEncoder().encode("%PDF-1.7 exact original bytes"))
const chatbook = async (missing = false) => {
  const zip = new JSZip()
  zip.file("manifest.json", JSON.stringify({ content_items: [{ type: "media", id: "1", file_path: "content/media/media_1.json" }, { type: "note", id: "note-old", file_path: "content/notes/note_note-old.md" }] }))
  zip.file("content/media/media_1.json", JSON.stringify({ id: 1, uuid: "media-uuid", content: "Canonical text", stored_artifacts: [{ file_type: "original", archive_path: "content/media/files/original.pdf", mime_type: "application/pdf", file_size: pdf.length, checksum: await sha256Bytes(pdf), bundled: !missing, pointer_only: missing }] }))
  if (!missing) zip.file("content/media/files/original.pdf", pdf)
  zip.file("content/notes/note_note-old.md", "Reviewed text")
  return zip.generateAsync({ type: "uint8array" })
}

describe("complete research pack archive", () => {
  beforeEach(() => Object.defineProperty(globalThis, "crypto", { value: webcrypto, configurable: true }))

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
