import { expect, it } from "vitest"
import {
  reconcileCapturedNoteProvenance,
  type KnowledgeNoteHead,
  type KnowledgeNoteProvenance,
  type KnowledgeNoteSource
} from "../knowledge-note-provenance"

const source = (id: string): KnowledgeNoteSource => ({
  originalId: id, excerpt: "Original excerpt", mediaId: 71, title: id,
  type: "website", sourceType: "web_capture", originalVersion: 9
})
const history: KnowledgeNoteProvenance = {
  origin: "knowledge_qa", question: "Original question", thread_id: "owned-thread",
  trust_state: "uncited_degraded_answer", trust_reason_codes: ["missing_citations"],
  scope: { include_note_ids: ["owned-note"], enable_web_fallback: false },
  sources: [source("A")]
}
const local = {
  knowledge_provenance_state: "active" as const,
  knowledge_provenance: history,
  pendingKnowledgeProvenance: { ...history, sources: [...history.sources!, source("C")] }
}
it.each(["deleted", "unsupported"] as const)(
  "reconciliation never restores a %s head from pending or portable history",
  (state) => {
    expect(reconcileCapturedNoteProvenance(local, {
      knowledge_provenance_state: state, knowledge_provenance: history
    })).toBeUndefined()
  }
)
it("deduplicates only new local additions and keeps remote qualifications and intentional removals", () => {
  const remote: KnowledgeNoteHead = {
    knowledge_provenance_state: "active",
    knowledge_provenance: { ...history, question: "Remote question", sources: [source("B"), source("C")] }
  }
  expect(reconcileCapturedNoteProvenance(local, remote)).toBeUndefined()
  expect(local.pendingKnowledgeProvenance.sources).toEqual([source("A"), source("C")])
  expect(remote.knowledge_provenance?.sources).toEqual([source("B"), source("C")])
})
it("a second fresh head does not restore references removed since the first rebase", () => {
  const first = { knowledge_provenance_state: "active" as const,
    knowledge_provenance: { ...history, sources: [source("A"), source("B")] } }
  const pending = reconcileCapturedNoteProvenance(local, first)
  const second = { ...first, knowledge_provenance: { ...history, sources: [source("D")] } }
  expect(reconcileCapturedNoteProvenance({ ...first, pendingKnowledgeProvenance: pending }, second))
    .toEqual({ ...history, sources: [source("D"), source("C")] })
})
it("an absent canonical head retains only new capture references, not removed QA history", () => {
  expect(reconcileCapturedNoteProvenance(local, {
    knowledge_provenance_state: "absent", knowledge_provenance_version: 0
  })).toEqual({ origin: "reviewed_sources", sources: [source("C")] })
})
it("a new draft without a canonical base retains its original qualifications", () => {
  expect(reconcileCapturedNoteProvenance({ pendingKnowledgeProvenance: history }, {})).toEqual(history)
})
it("reconciliation enforces the existing whole-history source bound", () => {
  expect(() => reconcileCapturedNoteProvenance(local, {
    knowledge_provenance_state: "active",
    knowledge_provenance: { ...history, sources: Array.from({ length: 100 }, (_, i) => source(`remote-${i}`)) }
  })).toThrow(/invalid or too large/)
})
