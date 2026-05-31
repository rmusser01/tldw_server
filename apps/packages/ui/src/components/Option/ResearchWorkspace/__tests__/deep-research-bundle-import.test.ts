import { describe, expect, it } from "vitest"
import type {
  ArtifactSourceCoverage,
  GeneratedArtifact
} from "@/types/workspace"
import {
  DeepResearchBundleImportError,
  MAX_IMPORT_LIST_ITEMS,
  buildDeepResearchBundleArtifactPayload
} from "../deep-research-bundle-import"

const returnContext = {
  sourceWorkspaceId: "workspace-1",
  sourceArtifactId: "gap-artifact",
  sourceArtifactTemplate: "corpus_gap_finder",
  sourceArtifactTitle: "Corpus Gap Finder",
  researchRunId: "research-run-7"
}

const sourceCoverage: ArtifactSourceCoverage = {
  selectedSourceIds: ["source-a", "source-b"],
  usableSources: [
    { sourceId: "source-a", mediaId: 101, title: "Paper A" },
    { sourceId: "source-b", mediaId: 102, title: "Paper B" }
  ],
  skippedSources: [],
  truncatedSources: [],
  sourceContextCharLimit: { perSource: 6000, total: 18000 },
  minimumUsableSourcesMet: true
}

const sourceArtifact = {
  id: "gap-artifact",
  type: "data_table",
  title: "Corpus Gap Finder",
  status: "completed",
  templateId: "corpus_gap_finder",
  sourceCoverage,
  sourceLineage: [
    { sourceId: "source-a", mediaId: 101, title: "Paper A", citationCount: 2 },
    { sourceId: "source-b", mediaId: 102, title: "Paper B", citationCount: 1 }
  ],
  content: "Gap artifact content",
  createdAt: new Date("2026-05-30T12:00:00.000Z"),
  completedAt: new Date("2026-05-30T12:01:00.000Z")
} satisfies GeneratedArtifact

const bundle = {
  question: "Which intervention gaps remain?",
  report_markdown: "# Deep Report\n\nThe evidence supports follow-up work.",
  claims: [
    {
      text: "Claim one",
      citations: [{ source_id: "src_1", title: "Paper A" }],
      support_level: "strong"
    }
  ],
  source_inventory: [
    { source_id: "src_1", title: "Paper A" },
    { source_id: "src_2", title: "Paper B" }
  ],
  unresolved_questions: ["What changes in field settings?"],
  verification_summary: {
    supported_claim_count: 1,
    unsupported_claim_count: 0
  },
  unsupported_claims: [],
  contradictions: [],
  source_trust: [{ source_id: "src_1", snapshot_policy: "full_artifact" }]
}

describe("Deep Research bundle import", () => {
  it("builds a completed Research Workspace artifact from a bundle", () => {
    const artifact = buildDeepResearchBundleArtifactPayload({
      bundle,
      returnContext,
      sourceArtifact
    })

    expect(artifact.type).toBe("report")
    expect(artifact.status).toBe("completed")
    expect(artifact.title).toBe("Deep Research: Corpus Gap Finder")
    expect(artifact.templateId).toBeUndefined()
    expect(artifact.content).toContain("# Deep Report")
    expect(artifact.content).toContain("Imported from Deep Research run")
    expect(artifact.sourceCoverage).toEqual(sourceCoverage)
    expect(artifact.sourceLineage).toEqual(sourceArtifact.sourceLineage)
    expect(artifact.producerMetadata).toMatchObject({
      producerType: "deep_research_bundle_import",
      runId: "research-run-7",
      templateId: "corpus_gap_finder"
    })
    expect(artifact.data?.deepResearch).toMatchObject({
      runId: "research-run-7",
      question: "Which intervention gaps remain?",
      sourceArtifact: {
        id: "gap-artifact",
        template: "corpus_gap_finder",
        title: "Corpus Gap Finder"
      },
      verificationSummary: {
        supported_claim_count: 1,
        unsupported_claim_count: 0
      },
      sourceTrust: [{ source_id: "src_1", snapshot_policy: "full_artifact" }]
    })
  })

  it("derives fallback source coverage from the bundle source inventory", () => {
    const artifact = buildDeepResearchBundleArtifactPayload({
      bundle,
      returnContext
    })

    expect(artifact.sourceCoverage).toEqual({
      selectedSourceIds: ["src_1", "src_2"],
      usableSources: [
        { sourceId: "src_1", title: "Paper A" },
        { sourceId: "src_2", title: "Paper B" }
      ],
      skippedSources: [],
      truncatedSources: [],
      minimumUsableSourcesMet: true
    })
    expect(artifact.sourceLineage).toEqual([
      { sourceId: "src_1", title: "Paper A", citationCount: 1 },
      { sourceId: "src_2", title: "Paper B", citationCount: 0 }
    ])
  })

  it("rejects malformed bundles without a usable report", () => {
    expect(() =>
      buildDeepResearchBundleArtifactPayload({
        bundle: { question: "Question only" },
        returnContext
      })
    ).toThrow(DeepResearchBundleImportError)
  })

  it("bounds imported readable content", () => {
    const longBundle = {
      ...bundle,
      report_markdown: `# Deep Report\n\n${"Long finding. ".repeat(2000)}`
    }

    const artifact = buildDeepResearchBundleArtifactPayload({
      bundle: longBundle,
      returnContext
    })

    expect(artifact.content?.length).toBeLessThan(9000)
    expect(artifact.content).toContain(
      "[Deep Research report truncated for workspace import.]"
    )
  })

  it("bounds imported bundle lists before persisting artifact metadata", () => {
    const unboundedList = Array.from({ length: MAX_IMPORT_LIST_ITEMS + 10 }, (_, index) => ({
      source_id: `src_${index}`,
      title: `Paper ${index}`,
      citations: [{ source_id: `src_${index}` }]
    }))
    const longBundle = {
      ...bundle,
      claims: unboundedList,
      source_inventory: unboundedList,
      unresolved_questions: Array.from(
        { length: MAX_IMPORT_LIST_ITEMS + 10 },
        (_, index) => `Question ${index}`
      ),
      unsupported_claims: unboundedList,
      contradictions: unboundedList,
      source_trust: unboundedList
    }

    const artifact = buildDeepResearchBundleArtifactPayload({
      bundle: longBundle,
      returnContext
    })
    const deepResearch = artifact.data?.deepResearch as {
      claims: unknown[]
      sourceInventory: unknown[]
      unresolvedQuestions: unknown[]
      unsupportedClaims: unknown[]
      contradictions: unknown[]
      sourceTrust: unknown[]
    }

    expect(deepResearch.claims).toHaveLength(MAX_IMPORT_LIST_ITEMS)
    expect(deepResearch.sourceInventory).toHaveLength(MAX_IMPORT_LIST_ITEMS)
    expect(deepResearch.unresolvedQuestions).toHaveLength(MAX_IMPORT_LIST_ITEMS)
    expect(deepResearch.unsupportedClaims).toHaveLength(MAX_IMPORT_LIST_ITEMS)
    expect(deepResearch.contradictions).toHaveLength(MAX_IMPORT_LIST_ITEMS)
    expect(deepResearch.sourceTrust).toHaveLength(MAX_IMPORT_LIST_ITEMS)
    expect(artifact.sourceCoverage.selectedSourceIds).toHaveLength(
      MAX_IMPORT_LIST_ITEMS
    )
    expect(artifact.sourceLineage).toHaveLength(MAX_IMPORT_LIST_ITEMS)
    expect(artifact.content).toContain(`Source inventory: ${MAX_IMPORT_LIST_ITEMS}`)
  })
})
