import { describe, expect, it } from "vitest"
import { remapPortableWorkspace, remapPortableNote, type VerifiedResearchImportMaps } from "../workspace-reference-remap"
import { readKnowledgeNoteProvenance } from "../knowledge-note-provenance"
import { portableFixture } from "./fixtures/portable-workspace"

const identity = { id: "destination", name: "Imported research", tag: "workspace:destination" }
const maps = { media: { "1": 42 }, notes: { "note-old": "note-new" } }

describe("verified portable workspace remapping", () => {
  it("remaps typed citations, sections, source membership and Note markers without changing quoted text", () => {
    const original = portableFixture()
    let sequence = 0
    const result = remapPortableWorkspace(original, maps, identity, () => `new-${++sequence}`)
    const snapshot = result.workspace.snapshot
    const source = snapshot.sources[0]
    expect(source.mediaId).toBe(42)
    expect(source.id).not.toBe("source-old")
    expect(snapshot.selectedSourceIds).toEqual([source.id])
    expect(snapshot.sourceAnnotations?.[source.id]).toEqual(original.workspace.snapshot.sourceAnnotations?.["source-old"])
    const section = (snapshot.generatedArtifacts[0].data!.sectionAnalysis as { sections: Array<Record<string, unknown> & { batches: Array<{ evidence: unknown[] }> }> }).sections[0]
    expect(section).toMatchObject({ source_id: source.id, media_id: 42, node_id: "old-node", navigation_version: "old-nav", content: "Original section" })
    expect(section.batches[0].evidence[0]).toEqual({ id: "E1", quote: "Exact quote", start_char: 1, end_char: 10 })
    expect(result.workspace.chatSession?.messages[0].sources[0]).toMatchObject({ metadata: { media_id: 42, page_number: 352, printed_page_label: "319" } })
    expect(result.workspace.chatSession?.messages[0].sources[0].metadata).not.toHaveProperty("chunk_id")
    expect(readKnowledgeNoteProvenance(snapshot.currentNote.content)?.sources?.[0]).toMatchObject({ mediaId: 42, originalId: 42, excerpt: "Exact quote", pageNumber: 352, printedPageLabel: "319" })
    expect(snapshot.currentNote.id).toBe("note-new")
    expect(original.workspace.snapshot.sources[0].mediaId).toBe(1)
  })

  it("detaches old jobs, canonical mutation IDs and chat identity while retaining the historical review", () => {
    const original = portableFixture()
    const result = remapPortableWorkspace(original, maps, identity)
    const artifact = result.workspace.snapshot.generatedArtifacts[0]
    expect(artifact.content).toBe("Unchanged draft")
    expect(artifact.reviewMetadata).toEqual(original.workspace.snapshot.generatedArtifacts[0].reviewMetadata)
    expect(artifact.producerMetadata?.origin).toBe("research_pack_import")
    expect(artifact.producerMetadata).not.toHaveProperty("job_id")
    expect(artifact).not.toHaveProperty("workspaceArtifactId")
    expect(artifact).not.toHaveProperty("serverId")
    expect(result.workspace.snapshot.currentNote).not.toHaveProperty("pendingNoteWriteKey")
    expect(result.workspace.snapshot.currentNote).not.toHaveProperty("version")
    expect(result.workspace.chatSession).toMatchObject({ historyId: null, serverChatId: null, history: original.workspace.chatSession!.history })
  })

  it.each([undefined, null, 0, -1, "42"])("rejects an unusable Media mapping (%s), never reusing old numeric identity", value => {
    expect(() => remapPortableWorkspace(portableFixture(), { ...maps, media: { "1": value } } as unknown as VerifiedResearchImportMaps, identity)).toThrow(/mapping/i)
  })

  it("requires mappings for cited Media outside source rows and for saved Notes", () => {
    const bundle = portableFixture()
    ;(bundle.workspace.chatSession!.messages[0].sources[0].metadata as Record<string, unknown>).media_id = 9
    expect(() => remapPortableWorkspace(bundle, maps, identity)).toThrow(/9/)
    expect(() => remapPortableWorkspace(portableFixture(), { media: { "1": 42 }, notes: {} }, identity)).toThrow(/note-old/)
  })

  it("maps Note evidence identities and numeric Media citation aliases by their source type", () => {
    const bundle = portableFixture()
    const evidence = bundle.workspace.snapshot.currentNote.knowledge_provenance!.sources![0]
    Object.assign(evidence, { mediaId: null, originalId: "note-old", sourceType: "notes_db" })
    const sources = bundle.workspace.chatSession!.messages[0].sources as Array<Record<string, unknown>>
    sources.push({ source_id: "1", source: "media", content: "Literal source 1 stays text" })
    const result = remapPortableWorkspace(bundle, maps, identity)
    expect(result.workspace.snapshot.currentNote.knowledge_provenance!.sources![0].originalId).toBe("note-new")
    expect(result.workspace.chatSession!.messages[0].sources[1]).toMatchObject({ source_id: "42", content: "Literal source 1 stays text" })
    sources.push({ source_id: "unmapped-source", source: "media" })
    expect(() => remapPortableWorkspace(bundle, maps, identity)).toThrow(/mapping|reference/i)
  })

  it("does not resurrect a removed canonical Note history from its retained marker", () => {
    const note = portableFixture().workspace.snapshot.currentNote
    note.knowledge_provenance_state = "deleted"
    note.knowledge_provenance = null
    const result = remapPortableNote(note, maps, identity)
    expect(readKnowledgeNoteProvenance(result.content)).toBeNull()
    expect(result.knowledge_provenance).toBeNull()
  })

  it("retains a section checkpoint stored only in the canonical client projection", () => {
    const bundle = portableFixture()
    const artifact = bundle.workspace.snapshot.generatedArtifacts[0]
    artifact.versionMetadata = { workspaceClient: { schemaVersion: 1, data: artifact.data } }
    delete artifact.data
    const result = remapPortableWorkspace(bundle, maps, identity)
    expect(result.workspace.snapshot.generatedArtifacts[0].data?.sectionAnalysis).toMatchObject({ fingerprint: "original-fingerprint", sections: [{ media_id: 42 }] })
  })
})
