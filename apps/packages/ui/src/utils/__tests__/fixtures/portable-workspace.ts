import { retainKnowledgeNoteProvenance } from "../../knowledge-note-provenance"
import type { WorkspaceExportBundle } from "@/store/workspace-bundle"

const provenance = {
  origin: "knowledge_qa" as const,
  sources: [{ originalId: 1, mediaId: 1, title: "Source", type: "pdf" as const, sourceType: "pdf", excerpt: "Exact quote", pageNumber: 352, printedPageLabel: "319", chunkId: "old-chunk" }]
}
export const portableFixture = (): WorkspaceExportBundle => ({
  format: "tldw.research-workspace.bundle", schemaVersion: 1, exportedAt: "2026-11-15T12:00:00Z",
  workspace: {
    name: "Research", tag: "workspace:old", createdAt: null, studyMaterialsPolicy: null,
    snapshot: {
      workspaceName: "Research", workspaceTag: "workspace:old", workspaceCreatedAt: null, studyMaterialsPolicy: null,
      sources: [{ id: "source-old", mediaId: 1, title: "Original PDF", type: "pdf", addedAt: new Date(0), statusDetails: { job: { id: 9, uuid: null, status: "completed", jobType: null, progressPercent: null, progressMessage: null, errorMessage: null } } }],
      selectedSourceIds: ["source-old"], sourceAnnotations: { "source-old": [{ id: "annotation", quote: "Exact quote", note: "Review note", createdAt: 1, updatedAt: 1 }] },
      generatedArtifacts: [{ id: "artifact-old", workspaceArtifactId: "server-artifact", workspaceArtifactVersion: 3, serverId: 7,
        type: "summary", title: "Chapter summary", status: "completed", content: "Unchanged draft", createdAt: new Date(0),
        producerMetadata: { origin: "research_workspace_output_job", job_id: 9, settings: { sections: [{ source_id: "source-old", node_id: "old-node" }] } },
        sourceLineage: [{ sourceId: "source-old", mediaId: 1 }], reviewStatus: "reviewed",
        reviewMetadata: { sectionReview: { schemaVersion: 1, entries: [{ targetId: "section:0", title: "Chapter", savedAt: "2026-11-15T00:00:00Z", evidenceIds: [], provenance: "user_review", targetFingerprint: "historical-fingerprint", correction: "Reviewed correction", reason: "Source check" }] } },
        data: { sectionAnalysis: { schemaVersion: 1, fingerprint: "original-fingerprint", sections: [{ source_id: "source-old", media_id: 1, node_id: "old-node", navigation_version: "old-nav", title: "Chapter", content: "Original section", page_spans: [{ page_number: 352, printed_page_label: "319", start_char: 1, end_char: 10 }], batches: [{ evidence: [{ id: "E1", quote: "Exact quote", start_char: 1, end_char: 10 }] }] }] } }
      }], notes: "Legacy draft", currentNote: { id: "note-old", title: "Note", content: retainKnowledgeNoteProvenance("Reviewed text", { knowledge_provenance: provenance }), knowledge_provenance: JSON.parse(JSON.stringify(provenance)), knowledge_provenance_state: "active", knowledge_provenance_version: 3, pendingNoteWriteKey: "old-authority-write", keywords: ["workspace:old"], version: 5, isDirty: true },
      workspaceBanner: null, leftPaneCollapsed: false, rightPaneCollapsed: false, audioSettings: { provider: "tldw", model: "test", voice: "test", speed: 1, format: "mp3" }
    },
    chatSession: { historyId: "old-local", serverChatId: "old-server", history: [{ role: "assistant", content: "Answer [1]" }], messages: [{ id: "message", isBot: true, name: "Assistant", message: "Answer [1]", createdAt: 1, sources: [{ name: "PDF", pageContent: "Exact quote", metadata: { media_id: 1, chunk_id: "old-chunk", page_number: 352, printed_page_label: "319" } }] }] }
  }
})
