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
      sourceTrust: [{ sourceId: "src_1", snapshotPolicy: "full_artifact" }]
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

  it("persists selected imported sources and recovery details from returned bundles", () => {
    const richBundle = {
      ...bundle,
      unsupported_claims: [
        { text: "Unsupported claim one", reason: "No matching citation" }
      ],
      contradictions: [
        { text: "Contradiction one", source_id: "src_2" }
      ],
      skipped_sources: [
        { source_id: "src_3", title: "Paper C", reason: "paywalled" }
      ],
      failed_sources: [
        { source_id: "src_4", title: "Paper D", reason: "extraction failed" }
      ]
    }

    const artifact = buildDeepResearchBundleArtifactPayload({
      bundle: richBundle,
      returnContext,
      sourceArtifact
    })
    const deepResearch = artifact.data?.deepResearch as {
      selectedImportedSources: unknown[]
      skippedSources: unknown[]
      failedSources: unknown[]
    }

    expect(deepResearch.selectedImportedSources).toEqual([
      {
        sourceId: "source-a",
        mediaId: 101,
        title: "Paper A",
        status: "selected"
      },
      {
        sourceId: "source-b",
        mediaId: 102,
        title: "Paper B",
        status: "selected"
      }
    ])
    expect(deepResearch.skippedSources).toEqual([
      { sourceId: "src_3", title: "Paper C", reason: "paywalled" }
    ])
    expect(deepResearch.failedSources).toEqual([
      { sourceId: "src_4", title: "Paper D", reason: "extraction failed" }
    ])
    expect(artifact.content).toContain("## Selected Imported Sources")
    expect(artifact.content).toContain("- Paper A (source-a, media #101) - selected")
    expect(artifact.content).toContain("## Skipped Sources")
    expect(artifact.content).toContain("- Paper C (src_3): paywalled")
    expect(artifact.content).toContain("## Failed Sources")
    expect(artifact.content).toContain("- Paper D (src_4): extraction failed")
    expect(artifact.content).toContain("## Unsupported Claims")
    expect(artifact.content).toContain("- Unsupported claim one")
    expect(artifact.content).toContain("## Contradictions")
    expect(artifact.content).toContain("- Contradiction one")
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
      skipped_sources: unboundedList,
      failed_sources: unboundedList,
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
      skippedSources: unknown[]
      failedSources: unknown[]
      sourceTrust: unknown[]
    }

    expect(deepResearch.claims).toHaveLength(MAX_IMPORT_LIST_ITEMS)
    expect(deepResearch.sourceInventory).toHaveLength(MAX_IMPORT_LIST_ITEMS)
    expect(deepResearch.unresolvedQuestions).toHaveLength(MAX_IMPORT_LIST_ITEMS)
    expect(deepResearch.unsupportedClaims).toHaveLength(MAX_IMPORT_LIST_ITEMS)
    expect(deepResearch.contradictions).toHaveLength(MAX_IMPORT_LIST_ITEMS)
    expect(deepResearch.skippedSources).toHaveLength(MAX_IMPORT_LIST_ITEMS)
    expect(deepResearch.failedSources).toHaveLength(MAX_IMPORT_LIST_ITEMS)
    expect(deepResearch.sourceTrust).toHaveLength(MAX_IMPORT_LIST_ITEMS)
    expect(artifact.sourceCoverage.selectedSourceIds).toHaveLength(
      MAX_IMPORT_LIST_ITEMS
    )
    expect(artifact.sourceLineage).toHaveLength(MAX_IMPORT_LIST_ITEMS)
    expect(artifact.content).toContain(
      `Source inventory: ${MAX_IMPORT_LIST_ITEMS + 10} (${MAX_IMPORT_LIST_ITEMS} shown)`
    )
  })

  it("normalizes retained bundle metadata before persisting artifact data", () => {
    const oversizedText = `Oversized field ${"x".repeat(1200)}`
    const oversizedSourceId = `src_${"x".repeat(300)}`
    const noisyBundle = {
      ...bundle,
      claims: [
        {
          claim_id: "claim-1",
          text: oversizedText,
          focus_area: "background",
          source_ids: ["src_1"],
          citations: [{ source_id: "src_1", quote: oversizedText }],
          metadata: { nested: oversizedText }
        }
      ],
      source_inventory: [
        {
          source_id: oversizedSourceId,
          title: oversizedText,
          provider: "local_corpus",
          metadata: { nested: oversizedText },
          excerpt: oversizedText
        }
      ],
      unsupported_claims: [
        {
          claim_id: "claim-2",
          text: oversizedText,
          focus_area: "background",
          reason: "No supporting notes",
          source_ids: ["src_1"],
          citations: [{ source_id: "src_1", quote: oversizedText }],
          metadata: { nested: oversizedText }
        }
      ],
      contradictions: [
        {
          note_id: "note-1",
          source_id: "src_1",
          focus_area: "background",
          marker: "however",
          text: oversizedText,
          metadata: { nested: oversizedText }
        }
      ],
      skipped_sources: [
        {
          source_id: oversizedSourceId,
          title: oversizedText,
          reason: oversizedText,
          metadata: { nested: oversizedText }
        }
      ],
      failed_sources: [
        {
          source_id: oversizedSourceId,
          title: oversizedText,
          reason: oversizedText,
          metadata: { nested: oversizedText }
        }
      ],
      source_trust: [
        {
          source_id: "src_1",
          title: oversizedText,
          provider: "local_corpus",
          source_type: "web_result",
          trust_tier: "internal",
          trust_label: "local_corpus",
          snapshot_policy: "full_artifact",
          warnings: ["full_source_snapshot_unavailable"],
          metadata: { nested: oversizedText },
          excerpt: oversizedText
        }
      ]
    }

    const artifact = buildDeepResearchBundleArtifactPayload({
      bundle: noisyBundle,
      returnContext
    })
    const deepResearch = artifact.data?.deepResearch as {
      claims: unknown[]
      sourceInventory: unknown[]
      skippedSources: unknown[]
      failedSources: unknown[]
      unsupportedClaims: unknown[]
      contradictions: unknown[]
      sourceTrust: unknown[]
    }
    const serializedMetadata = JSON.stringify(deepResearch)

    expect(deepResearch.sourceInventory).toEqual([
      { sourceId: expect.any(String), title: expect.any(String) }
    ])
    expect(deepResearch.sourceInventory[0]).toMatchObject({
      sourceId: expect.stringMatching(/^src_/)
    })
    expect(String((deepResearch.sourceInventory[0] as { sourceId: string }).sourceId))
      .toHaveLength(128)
    expect(deepResearch.skippedSources[0]).toMatchObject({
      sourceId: expect.stringMatching(/^src_/),
      reason: expect.any(String)
    })
    expect(
      String((deepResearch.skippedSources[0] as { sourceId: string }).sourceId)
    ).toHaveLength(128)
    expect(
      String((deepResearch.skippedSources[0] as { reason: string }).reason).length
    ).toBeLessThanOrEqual(500)
    expect(
      String((deepResearch.failedSources[0] as { reason: string }).reason).length
    ).toBeLessThanOrEqual(500)
    expect(deepResearch.sourceTrust).toEqual([
      {
        sourceId: "src_1",
        title: expect.any(String),
        provider: "local_corpus",
        sourceType: "web_result",
        trustTier: "internal",
        trustLabel: "local_corpus",
        snapshotPolicy: "full_artifact",
        warnings: ["full_source_snapshot_unavailable"]
      }
    ])
    expect(deepResearch.unsupportedClaims[0]).toMatchObject({
      claimId: "claim-2",
      text: expect.any(String),
      focusArea: "background",
      reason: "No supporting notes",
      sourceIds: ["src_1"]
    })
    expect(deepResearch.contradictions[0]).toMatchObject({
      noteId: "note-1",
      sourceId: "src_1",
      focusArea: "background",
      marker: "however",
      text: expect.any(String)
    })
    expect(deepResearch.claims[0]).toMatchObject({
      claimId: "claim-1",
      text: expect.any(String),
      focusArea: "background",
      sourceIds: ["src_1"]
    })
    expect(serializedMetadata).not.toContain("metadata")
    expect(serializedMetadata).not.toContain("excerpt")
    expect(serializedMetadata).not.toContain("citations")
    expect(serializedMetadata).not.toContain(oversizedText)
    expect(serializedMetadata).not.toContain(oversizedSourceId)
  })

  it("bounds provenance copied from an existing source artifact", () => {
    const unboundedSources = Array.from(
      { length: MAX_IMPORT_LIST_ITEMS + 10 },
      (_, index) => ({
        sourceId: `source-${index}`,
        mediaId: index + 1,
        title: `Paper ${index}`
      })
    )
    const unboundedSourceArtifact = {
      ...sourceArtifact,
      sourceCoverage: {
        selectedSourceIds: unboundedSources.map((source) => source.sourceId),
        usableSources: unboundedSources,
        skippedSources: unboundedSources.map((source) => ({
          ...source,
          reason: "context_limit" as const
        })),
        truncatedSources: unboundedSources,
        minimumUsableSourcesMet: true
      },
      sourceLineage: unboundedSources.map((source) => ({
        ...source,
        citationCount: 1,
        citationSpans: Array.from(
          { length: MAX_IMPORT_LIST_ITEMS + 10 },
          (_, index) => ({ index })
        ),
        evidenceIds: Array.from(
          { length: MAX_IMPORT_LIST_ITEMS + 10 },
          (_, index) => `evidence-${index}`
        ),
        oversizedUnknownField: Array.from(
          { length: MAX_IMPORT_LIST_ITEMS + 10 },
          (_, index) => ({ index })
        )
      }))
    } satisfies GeneratedArtifact

    const artifact = buildDeepResearchBundleArtifactPayload({
      bundle,
      returnContext,
      sourceArtifact: unboundedSourceArtifact
    })

    expect(artifact.sourceCoverage.selectedSourceIds).toHaveLength(
      MAX_IMPORT_LIST_ITEMS
    )
    expect(artifact.sourceCoverage.usableSources).toHaveLength(
      MAX_IMPORT_LIST_ITEMS
    )
    expect(artifact.sourceCoverage.skippedSources).toHaveLength(
      MAX_IMPORT_LIST_ITEMS
    )
    expect(artifact.sourceCoverage.truncatedSources).toHaveLength(
      MAX_IMPORT_LIST_ITEMS
    )
    expect(artifact.sourceLineage).toHaveLength(MAX_IMPORT_LIST_ITEMS)
    expect(artifact.sourceLineage[0].citationSpans).toHaveLength(
      MAX_IMPORT_LIST_ITEMS
    )
    expect(artifact.sourceLineage[0].evidenceIds).toHaveLength(
      MAX_IMPORT_LIST_ITEMS
    )
    expect(artifact.sourceLineage[0]).not.toHaveProperty("oversizedUnknownField")
  })

  it("treats missing source provenance arrays as empty during import", () => {
    const partialSourceArtifact = {
      ...sourceArtifact,
      sourceCoverage: {
        minimumUsableSourcesMet: false
      },
      sourceLineage: "not-a-list"
    } as unknown as GeneratedArtifact

    const artifact = buildDeepResearchBundleArtifactPayload({
      bundle,
      returnContext,
      sourceArtifact: partialSourceArtifact
    })

    expect(artifact.sourceCoverage).toMatchObject({
      selectedSourceIds: [],
      usableSources: [],
      skippedSources: [],
      truncatedSources: [],
      minimumUsableSourcesMet: false
    })
    expect(artifact.sourceLineage).toEqual([])
  })
})
