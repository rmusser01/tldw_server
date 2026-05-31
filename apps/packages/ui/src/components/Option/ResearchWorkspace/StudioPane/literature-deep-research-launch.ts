import { buildResearchLaunchPath } from "@/routes/route-paths"
import type {
  ArtifactSourceCoverage,
  ArtifactSourceCoverageEntry,
  ArtifactSkippedSource,
  GeneratedArtifact
} from "@/types/workspace"
import type { NormalizedEvidenceBoundHypothesis } from "./literature-workproducts"

const LITERATURE_DEEP_RESEARCH_TEMPLATE_IDS = new Set([
  "literature_matrix",
  "corpus_gap_finder",
  "evidence_bound_hypotheses",
  "research_proposal_pack"
])

const LITERATURE_DEEP_RESEARCH_FOLLOW_UP_TEMPLATE_IDS = new Set([
  "evidence_bound_hypotheses",
  "research_proposal_pack"
])

const ARTIFACT_EXCERPT_LIMIT = 1400
const RESEARCH_QUERY_LIMIT = 2600
const TRUNCATION_SUFFIX = "\n\n[Truncated for Deep Research launch context.]"
const FOLLOW_UP_CLAIM_LIMIT = 900
const FOLLOW_UP_QUESTION_LIMIT = 1800

type ResearchFollowUpSeed = {
  question: string
  background: {
    question: string
    outline: Array<{
      title: string
      focus_area?: string | null
    }>
    key_claims: Array<{
      claim_id: string
      text: string
    }>
    unresolved_questions: string[]
    verification_summary: {
      supported_claim_count: number
      unsupported_claim_count: number
    }
    source_trust_summary: {
      high_trust_count: number
      low_trust_count: number
    }
  }
}

type LiteratureDeepResearchLaunchOptions = {
  workspaceId?: string | null
}

const truncateForLaunch = (value: string, limit: number): string => {
  const trimmed = value.trim()
  if (limit <= 0) {
    return ""
  }
  if (trimmed.length <= limit) {
    return trimmed
  }
  if (limit <= TRUNCATION_SUFFIX.length) {
    return TRUNCATION_SUFFIX.slice(0, limit)
  }
  return `${trimmed.slice(0, limit - TRUNCATION_SUFFIX.length).trimEnd()}${TRUNCATION_SUFFIX}`
}

const formatCoverageEntry = (source: ArtifactSourceCoverageEntry): string =>
  source.title || source.sourceId

const formatSkippedSource = (source: ArtifactSkippedSource): string =>
  `${formatCoverageEntry(source)} (${source.reason})`

const formatCoverageEntryList = (
  sources: ArtifactSourceCoverageEntry[] | undefined
): string => sources?.map(formatCoverageEntry).join(", ") || "none"

const formatSkippedSourceList = (
  sources: ArtifactSkippedSource[] | undefined
): string => sources?.map(formatSkippedSource).join(", ") || "none"

const formatSourceCoverageClaim = (
  sourceCoverage: ArtifactSourceCoverage
): string =>
  [
    "Source coverage from Research Workspace artifact:",
    `selected IDs: ${(sourceCoverage.selectedSourceIds ?? []).join(", ") || "none"}`,
    `usable sources: ${formatCoverageEntryList(sourceCoverage.usableSources)}`,
    `skipped sources: ${formatSkippedSourceList(sourceCoverage.skippedSources)}`,
    `truncated sources: ${formatCoverageEntryList(sourceCoverage.truncatedSources)}`
  ].join(" ")

const formatSourceCoverage = (sourceCoverage: ArtifactSourceCoverage): string =>
  [
    `Selected source IDs: ${(sourceCoverage.selectedSourceIds ?? []).join(", ") || "none"}`,
    `Usable sources: ${formatCoverageEntryList(sourceCoverage.usableSources)}`,
    `Skipped sources: ${formatSkippedSourceList(sourceCoverage.skippedSources)}`,
    `Truncated sources: ${formatCoverageEntryList(sourceCoverage.truncatedSources)}`
  ].join("\n")

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const readStringList = (value: unknown): string[] =>
  Array.isArray(value)
    ? value
        .map((entry) => String(entry ?? "").trim())
        .filter(
          (entry, index, all) =>
            entry.length > 0 && all.indexOf(entry) === index
        )
    : []

const normalizeHypotheses = (
  artifact: GeneratedArtifact
): NormalizedEvidenceBoundHypothesis[] => {
  const rawHypotheses = isRecord(artifact.data) ? artifact.data.hypotheses : null
  if (!Array.isArray(rawHypotheses)) {
    return []
  }

  return rawHypotheses
    .filter(isRecord)
    .map((entry) => ({
      hypothesis: String(entry.hypothesis ?? "").trim(),
      supportingFindings: readStringList(entry.supportingFindings),
      supportingSources: readStringList(entry.supportingSources),
      prediction: String(entry.prediction ?? "").trim(),
      suggestedMethodology: String(entry.suggestedMethodology ?? "").trim(),
      threatsToValidity: readStringList(entry.threatsToValidity),
      whatWouldFalsifyIt: String(entry.whatWouldFalsifyIt ?? "").trim(),
      confidence: String(entry.confidence ?? "").trim()
    }))
    .filter((entry) => entry.hypothesis.length > 0)
}

const makeClaimId = (artifact: GeneratedArtifact, suffix: string): string => {
  const rawId = `${artifact.id}:${suffix}`
  return rawId.length <= 128 ? rawId : rawId.slice(0, 128)
}

const buildCommonUnresolvedQuestions = (subject: string): string[] => [
  `Which parts of the ${subject} are evidence-supported versus proposed work?`,
  "Which selected sources confirm, weaken, or contradict the proposed work?",
  "Which skipped or truncated sources need review before trusting this follow-up?"
]

const buildSourceTrustSummary = (sourceCoverage: ArtifactSourceCoverage) => ({
  high_trust_count: sourceCoverage.usableSources?.length ?? 0,
  low_trust_count:
    (sourceCoverage.skippedSources?.length ?? 0) +
    (sourceCoverage.truncatedSources?.length ?? 0)
})

const buildCoverageClaim = (
  artifact: GeneratedArtifact,
  sourceCoverage: ArtifactSourceCoverage,
  suffix = "source-coverage"
) => ({
  claim_id: makeClaimId(artifact, suffix),
  text: truncateForLaunch(formatSourceCoverageClaim(sourceCoverage), FOLLOW_UP_CLAIM_LIMIT)
})

const buildHypothesesFollowUp = (
  artifact: GeneratedArtifact,
  sourceCoverage: ArtifactSourceCoverage
): ResearchFollowUpSeed | null => {
  const hypotheses = normalizeHypotheses(artifact)
  if (hypotheses.length === 0) {
    return null
  }

  const headline = hypotheses
    .slice(0, 2)
    .map((entry) => entry.hypothesis)
    .join(" ")
  const question = truncateForLaunch(
    `Verify and expand these Research Workspace evidence-bound hypotheses: ${headline}`,
    FOLLOW_UP_QUESTION_LIMIT
  )
  const hypothesisClaims = hypotheses.slice(0, 4).map((entry, index) => {
    const finding = entry.supportingFindings[0] || "No supporting finding listed."
    const sources = entry.supportingSources.join(", ") || "unknown sources"
    const proposedWork = [
      entry.prediction ? `Prediction: ${entry.prediction}` : "",
      entry.suggestedMethodology
        ? `Proposed method: ${entry.suggestedMethodology}`
        : ""
    ]
      .filter(Boolean)
      .join(" ")
    return {
      claim_id: makeClaimId(artifact, `hypothesis-${index + 1}`),
      text: truncateForLaunch(
        [
          `Evidence-supported finding for hypothesis ${index + 1}: ${finding}`,
          `Sources: ${sources}.`,
          proposedWork
        ]
          .filter(Boolean)
          .join(" "),
        FOLLOW_UP_CLAIM_LIMIT
      )
    }
  })

  return {
    question,
    background: {
      question,
      outline: hypotheses.slice(0, 7).map((_entry, index) => ({
        title: `Hypothesis ${index + 1}`,
        focus_area: `hypothesis_${index + 1}`
      })),
      key_claims: [
        ...hypothesisClaims.slice(0, 4),
        buildCoverageClaim(artifact, sourceCoverage)
      ].slice(0, 5),
      unresolved_questions: buildCommonUnresolvedQuestions("hypothesis"),
      verification_summary: {
        supported_claim_count: Math.min(hypotheses.length, 4),
        unsupported_claim_count: 1
      },
      source_trust_summary: buildSourceTrustSummary(sourceCoverage)
    }
  }
}

const extractMarkdownSection = (content: string, headings: RegExp[]): string => {
  const lines = content.split(/\r?\n/)
  let collecting = false
  let startLevel = 0
  const collected: string[] = []

  for (const line of lines) {
    const headingMatch = line.match(/^(#{1,6})\s+/)
    if (headingMatch) {
      const headingLevel = headingMatch[1].length
      if (collecting) {
        if (headingLevel <= startLevel) {
          break
        }
      } else if (headings.some((heading) => heading.test(line))) {
        collecting = true
        startLevel = headingLevel
        continue
      }
    }
    if (collecting) {
      collected.push(line)
    }
  }

  return collected.join("\n").trim()
}

const buildProposalFollowUp = (
  artifact: GeneratedArtifact,
  sourceCoverage: ArtifactSourceCoverage
): ResearchFollowUpSeed | null => {
  const content = artifact.content?.trim()
  if (!content) {
    return null
  }

  const evidenceExcerpt =
    extractMarkdownSection(content, [/^#{1,6}\s+Literature Overview\b/i]) ||
    extractMarkdownSection(content, [/^#{1,6}\s+Source Audit\b/i]) ||
    content
  const proposedHypothesis = extractMarkdownSection(content, [
    /^#{1,6}\s+Proposed Hypothesis\b/i
  ])
  const methodology = extractMarkdownSection(content, [/^#{1,6}\s+Methodology\b/i])
  const proposedExcerpt = [proposedHypothesis, methodology]
    .filter(Boolean)
    .join("\n\n")
  const question = truncateForLaunch(
    `Verify and expand this Research Workspace proposal, separating evidence-supported claims from proposed work: ${artifact.title}`,
    FOLLOW_UP_QUESTION_LIMIT
  )

  return {
    question,
    background: {
      question,
      outline: [
        {
          title: "Literature evidence",
          focus_area: "evidence_supported_claims"
        },
        {
          title: "Proposed work",
          focus_area: "proposed_work"
        }
      ],
      key_claims: [
        {
          claim_id: makeClaimId(artifact, "evidence-supported-excerpt"),
          text: truncateForLaunch(
            `Evidence-supported proposal excerpt: ${evidenceExcerpt}`,
            FOLLOW_UP_CLAIM_LIMIT
          )
        },
        {
          claim_id: makeClaimId(artifact, "proposed-work-excerpt"),
          text: truncateForLaunch(
            `Proposed-work excerpt: ${proposedExcerpt || content}`,
            FOLLOW_UP_CLAIM_LIMIT
          )
        },
        buildCoverageClaim(artifact, sourceCoverage)
      ],
      unresolved_questions: buildCommonUnresolvedQuestions("proposal"),
      verification_summary: {
        supported_claim_count: evidenceExcerpt ? 1 : 0,
        unsupported_claim_count: proposedExcerpt ? 1 : 0
      },
      source_trust_summary: buildSourceTrustSummary(sourceCoverage)
    }
  }
}

const buildLiteratureDeepResearchFollowUp = (
  artifact: GeneratedArtifact,
  sourceCoverage: ArtifactSourceCoverage
): ResearchFollowUpSeed | null => {
  if (artifact.templateId === "evidence_bound_hypotheses") {
    return buildHypothesesFollowUp(artifact, sourceCoverage)
  }
  if (artifact.templateId === "research_proposal_pack") {
    return buildProposalFollowUp(artifact, sourceCoverage)
  }
  return null
}

export const isDeepResearchLaunchableLiteratureArtifact = (
  artifact: GeneratedArtifact
): boolean => {
  if (artifact.status !== "completed") {
    return false
  }
  if (
    !artifact.templateId ||
    !LITERATURE_DEEP_RESEARCH_TEMPLATE_IDS.has(artifact.templateId)
  ) {
    return false
  }
  if (!artifact.sourceCoverage?.minimumUsableSourcesMet) {
    return false
  }
  if ((artifact.sourceCoverage.usableSources?.length ?? 0) < 2) {
    return false
  }
  if (
    artifact.templateId &&
    LITERATURE_DEEP_RESEARCH_FOLLOW_UP_TEMPLATE_IDS.has(artifact.templateId)
  ) {
    return (
      buildLiteratureDeepResearchFollowUp(artifact, artifact.sourceCoverage) !== null
    )
  }
  return typeof artifact.content === "string" && artifact.content.trim().length > 0
}

export const buildLiteratureDeepResearchLaunchQuery = (
  artifact: GeneratedArtifact
): string | null => {
  if (!isDeepResearchLaunchableLiteratureArtifact(artifact)) {
    return null
  }

  const sourceCoverage = artifact.sourceCoverage
  if (!sourceCoverage) {
    return null
  }
  const followUp = buildLiteratureDeepResearchFollowUp(artifact, sourceCoverage)
  if (
    artifact.templateId &&
    LITERATURE_DEEP_RESEARCH_FOLLOW_UP_TEMPLATE_IDS.has(artifact.templateId)
  ) {
    if (!followUp) {
      return null
    }
    const artifactExcerpt = truncateForLaunch(
      artifact.content || "",
      ARTIFACT_EXCERPT_LIMIT
    )
    return truncateForLaunch(
      [
        followUp.question,
        "",
        `Artifact: ${artifact.title}`,
        `Artifact template: ${artifact.templateId}`,
        "",
        "Source coverage from the artifact:",
        formatSourceCoverage(sourceCoverage),
        "",
        "Artifact excerpt:",
        artifactExcerpt
      ].join("\n"),
      RESEARCH_QUERY_LIMIT
    )
  }

  const artifactKind =
    artifact.templateId === "corpus_gap_finder"
      ? "Corpus Gap Finder"
      : "Literature Matrix"
  const artifactExcerpt = truncateForLaunch(
    artifact.content || "",
    ARTIFACT_EXCERPT_LIMIT
  )
  const query = [
    `Run Deep Research from this Research Workspace ${artifactKind} artifact.`,
    "",
    `Artifact: ${artifact.title}`,
    `Artifact template: ${artifact.templateId}`,
    "",
    "Source coverage from the artifact:",
    formatSourceCoverage(sourceCoverage),
    "",
    "Research task:",
    artifact.templateId === "corpus_gap_finder"
      ? "Verify, expand, and prioritize the identified corpus gaps. Look for additional evidence, counterexamples, and practical follow-up research questions."
      : "Verify, expand, and stress-test the matrix findings. Look for missing evidence, contradictions, and follow-up questions across the source set.",
    "",
    "Artifact excerpt:",
    artifactExcerpt
  ].join("\n")

  return truncateForLaunch(query, RESEARCH_QUERY_LIMIT)
}

export const buildLiteratureDeepResearchLaunchPath = (
  artifact: GeneratedArtifact,
  options: LiteratureDeepResearchLaunchOptions = {}
): string | null => {
  const query = buildLiteratureDeepResearchLaunchQuery(artifact)
  if (!query) {
    return null
  }
  const sourceCoverage = artifact.sourceCoverage
  if (!sourceCoverage) {
    return null
  }

  return buildResearchLaunchPath({
    query,
    sourcePolicy: "local_first",
    autonomyMode: "checkpointed",
    from: "research-workspace",
    sourceWorkspaceId: options.workspaceId,
    sourceArtifactId: artifact.id,
    sourceArtifactTemplate: artifact.templateId,
    sourceArtifactTitle: artifact.title,
    followUp: buildLiteratureDeepResearchFollowUp(artifact, sourceCoverage)
  })
}
