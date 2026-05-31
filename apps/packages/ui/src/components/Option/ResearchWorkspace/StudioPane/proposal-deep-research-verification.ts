import type { GeneratedArtifact } from "@/types/workspace"

const VERIFIED_PROPOSAL_SECTION_HEADINGS = new Set([
  "literature overview",
  "proposed hypothesis",
  "methodology",
  "source audit"
])

export interface ProposalDeepResearchVerificationSummary {
  runId: string
  question?: string
  supportedClaimCount: number
  unsupportedClaimCount: number
  unresolvedQuestions: string[]
  contradictionCount: number
  sourceTrustCount: number
}

export interface ProposalDeepResearchVerificationSection {
  heading: string
  level: number
  body: string
  verification?: ProposalDeepResearchVerificationSummary
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const asRecord = (value: unknown): Record<string, unknown> =>
  isRecord(value) ? value : {}

const readString = (value: unknown): string =>
  typeof value === "string" ? value.trim() : ""

const readCount = (value: unknown): number =>
  typeof value === "number" && Number.isFinite(value) ? value : 0

const readRecordList = (value: unknown): Array<Record<string, unknown>> =>
  Array.isArray(value) ? value.filter(isRecord) : []

const readStringList = (value: unknown): string[] =>
  Array.isArray(value)
    ? value.map(readString).filter((entry) => entry.length > 0)
    : []

const getDeepResearchData = (
  artifact: GeneratedArtifact
): Record<string, unknown> | null => {
  if (
    artifact.status !== "completed" ||
    artifact.producerMetadata?.producerType !== "deep_research_bundle_import"
  ) {
    return null
  }
  const deepResearch = isRecord(artifact.data)
    ? artifact.data.deepResearch
    : null
  return isRecord(deepResearch) ? deepResearch : null
}

export const findProposalDeepResearchVerificationArtifact = (
  proposalArtifact: GeneratedArtifact,
  artifacts: GeneratedArtifact[]
): GeneratedArtifact | null => {
  if (
    proposalArtifact.templateId !== "research_proposal_pack" ||
    proposalArtifact.status !== "completed"
  ) {
    return null
  }

  for (const artifact of artifacts) {
    const deepResearch = getDeepResearchData(artifact)
    if (!deepResearch) continue

    const sourceArtifact = asRecord(deepResearch.sourceArtifact)
    const sourceTemplate = readString(sourceArtifact.template)
    if (
      readString(sourceArtifact.id) === proposalArtifact.id &&
      (!sourceTemplate || sourceTemplate === "research_proposal_pack")
    ) {
      return artifact
    }
  }

  return null
}

const buildVerificationSummary = (
  verificationArtifact: GeneratedArtifact
): ProposalDeepResearchVerificationSummary | undefined => {
  const deepResearch = getDeepResearchData(verificationArtifact)
  if (!deepResearch) return undefined

  const verificationSummary = asRecord(deepResearch.verificationSummary)
  const unsupportedClaims = readRecordList(deepResearch.unsupportedClaims)
  const runId =
    readString(deepResearch.runId) ||
    readString(verificationArtifact.producerMetadata?.runId)
  if (!runId) return undefined

  return {
    runId,
    question: readString(deepResearch.question) || undefined,
    supportedClaimCount: readCount(verificationSummary.supported_claim_count),
    unsupportedClaimCount:
      readCount(verificationSummary.unsupported_claim_count) ||
      unsupportedClaims.length,
    unresolvedQuestions: readStringList(deepResearch.unresolvedQuestions),
    contradictionCount: readRecordList(deepResearch.contradictions).length,
    sourceTrustCount: readRecordList(deepResearch.sourceTrust).length
  }
}

const normalizeHeading = (heading: string): string =>
  heading.trim().toLowerCase()

const shouldShowVerificationForHeading = (heading: string): boolean =>
  VERIFIED_PROPOSAL_SECTION_HEADINGS.has(normalizeHeading(heading))

const fallbackSection = (content: string): ProposalDeepResearchVerificationSection[] =>
  content.trim()
    ? [
        {
          heading: "Research Proposal Pack",
          level: 1,
          body: content.trim()
        }
      ]
    : []

const parseProposalSections = (
  content: string
): ProposalDeepResearchVerificationSection[] => {
  const lines = content.split(/\r?\n/)
  const sections: ProposalDeepResearchVerificationSection[] = []
  let current:
    | {
        heading: string
        level: number
        body: string[]
      }
    | null = null

  const flush = () => {
    if (!current) return
    sections.push({
      heading: current.heading,
      level: current.level,
      body: current.body.join("\n").trim()
    })
  }

  for (const line of lines) {
    const headingMatch = line.match(/^(#{1,6})\s+(.+?)\s*$/)
    if (headingMatch) {
      flush()
      current = {
        heading: headingMatch[2].trim(),
        level: headingMatch[1].length,
        body: []
      }
      continue
    }

    if (current) {
      current.body.push(line)
    }
  }

  flush()
  return sections.length > 0 ? sections : fallbackSection(content)
}

export const buildProposalDeepResearchVerificationSections = (
  proposalArtifact: GeneratedArtifact,
  verificationArtifact?: GeneratedArtifact | null
): ProposalDeepResearchVerificationSection[] => {
  const sections = parseProposalSections(proposalArtifact.content ?? "")
  const verification = verificationArtifact
    ? buildVerificationSummary(verificationArtifact)
    : undefined

  if (!verification) {
    return sections
  }

  return sections.map((section) =>
    shouldShowVerificationForHeading(section.heading)
      ? { ...section, verification }
      : section
  )
}
