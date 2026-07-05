import type {
  ArtifactSourceCoverage,
  ArtifactSourceCoverageEntry,
  ArtifactSourceLineage,
  GeneratedArtifact
} from "@/types/workspace"
import type { ResearchWorkspaceDeepResearchReturnContext } from "./research-workspace-route-state"

const IMPORTED_REPORT_LIMIT = 7_600
const PREVIEW_LIMIT = 280
export const MAX_IMPORT_LIST_ITEMS = 50
const IMPORT_TRUNCATION_SUFFIX =
  "\n\n[Deep Research report truncated for workspace import.]"

type DeepResearchBundleImportOptions = {
  bundle: unknown
  returnContext: ResearchWorkspaceDeepResearchReturnContext
  sourceArtifact?: GeneratedArtifact | null
}

type GeneratedArtifactPayload = Omit<GeneratedArtifact, "id" | "createdAt">
type DeepResearchImportedSource = ArtifactSourceCoverageEntry & {
  status: "selected" | "inventory"
}
type DeepResearchSourceDetail = {
  sourceId: string
  title?: string
  reason?: string
}

export class DeepResearchBundleImportError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "DeepResearchBundleImportError"
  }
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const asRecord = (value: unknown): Record<string, unknown> =>
  isRecord(value) ? value : {}

const readString = (value: unknown): string =>
  typeof value === "string" ? value.trim() : ""

const capImportList = <T,>(items: T[]): T[] =>
  items.slice(0, MAX_IMPORT_LIST_ITEMS)

const readRecordList = (value: unknown): Array<Record<string, unknown>> =>
  Array.isArray(value) ? capImportList(value.filter(isRecord)) : []

const readStringList = (value: unknown): string[] =>
  Array.isArray(value) ? capImportList(value.map(readString).filter(Boolean)) : []

const readQuestion = (bundle: Record<string, unknown>): string => {
  const directQuestion = readString(bundle.question)
  if (directQuestion) return directQuestion

  return readString(asRecord(bundle.brief).query)
}

const truncateForImport = (value: string, limit: number): string => {
  const trimmed = value.trim()
  if (trimmed.length <= limit) return trimmed
  if (limit <= IMPORT_TRUNCATION_SUFFIX.length) {
    return IMPORT_TRUNCATION_SUFFIX.slice(0, limit)
  }

  return `${trimmed
    .slice(0, limit - IMPORT_TRUNCATION_SUFFIX.length)
    .trimEnd()}${IMPORT_TRUNCATION_SUFFIX}`
}

const formatCount = (label: string, value: unknown): string | null =>
  typeof value === "number" && Number.isFinite(value)
    ? `${label}: ${value}`
    : null

const normalizeSourceInventory = (
  bundle: Record<string, unknown>
): ArtifactSourceCoverageEntry[] =>
  readRecordList(bundle.source_inventory)
    .map((entry): ArtifactSourceCoverageEntry | null => {
      const sourceId = readString(entry.source_id ?? entry.id)
      if (!sourceId) return null

      const title = readString(entry.title ?? entry.label)
      return {
        sourceId,
        ...(title ? { title } : {})
      }
    })
    .filter((entry): entry is ArtifactSourceCoverageEntry => Boolean(entry))

const buildSelectedImportedSources = (
  sourceCoverage: ArtifactSourceCoverage,
  sourceInventory: ArtifactSourceCoverageEntry[]
): DeepResearchImportedSource[] => {
  const selectedSources = sourceCoverage.usableSources.length
    ? sourceCoverage.usableSources
    : sourceInventory
  const status: DeepResearchImportedSource["status"] =
    sourceCoverage.usableSources.length ? "selected" : "inventory"

  return capImportList(selectedSources).map((source) => ({
    sourceId: source.sourceId,
    ...(source.mediaId !== undefined ? { mediaId: source.mediaId } : {}),
    ...(source.title ? { title: source.title } : {}),
    status
  }))
}

const normalizeBundleSourceDetails = (
  value: unknown
): DeepResearchSourceDetail[] =>
  readRecordList(value)
    .map((entry): DeepResearchSourceDetail | null => {
      const sourceId = readString(entry.source_id ?? entry.sourceId ?? entry.id)
      if (!sourceId) return null

      const title = readString(entry.title ?? entry.label)
      const reason = readString(
        entry.reason ?? entry.error ?? entry.message ?? entry.status_message
      )
      return {
        sourceId,
        ...(title ? { title } : {}),
        ...(reason ? { reason } : {})
      }
    })
    .filter((entry): entry is DeepResearchSourceDetail => Boolean(entry))

const normalizeCoverageSkippedSources = (
  sourceCoverage: ArtifactSourceCoverage
): DeepResearchSourceDetail[] =>
  capImportList(sourceCoverage.skippedSources).map((source) => ({
    sourceId: source.sourceId,
    ...(source.title ? { title: source.title } : {}),
    reason: source.reason
  }))

const buildFallbackSourceCoverage = (
  sources: ArtifactSourceCoverageEntry[]
): ArtifactSourceCoverage => ({
  selectedSourceIds: sources.map((source) => source.sourceId),
  usableSources: sources,
  skippedSources: [],
  truncatedSources: [],
  minimumUsableSourcesMet: sources.length > 0
})

const sanitizeSourceCoverage = (
  sourceCoverage: ArtifactSourceCoverage
): ArtifactSourceCoverage => ({
  selectedSourceIds: capImportList(
    sourceCoverage.selectedSourceIds.map(readString).filter(Boolean)
  ),
  usableSources: capImportList(sourceCoverage.usableSources),
  skippedSources: capImportList(sourceCoverage.skippedSources),
  truncatedSources: capImportList(sourceCoverage.truncatedSources),
  ...(sourceCoverage.sourceContextCharLimit
    ? { sourceContextCharLimit: sourceCoverage.sourceContextCharLimit }
    : {}),
  minimumUsableSourcesMet: sourceCoverage.minimumUsableSourcesMet
})

const sanitizeSourceLineage = (
  sourceLineage: ArtifactSourceLineage[]
): ArtifactSourceLineage[] =>
  capImportList(sourceLineage)
    .map((lineage): ArtifactSourceLineage | null => {
      const sourceId = readString(lineage.sourceId)
      if (!sourceId) return null

      return {
        sourceId,
        ...(readString(lineage.sourceType)
          ? { sourceType: readString(lineage.sourceType) }
          : {}),
        ...(typeof lineage.mediaId === "number" && Number.isFinite(lineage.mediaId)
          ? { mediaId: lineage.mediaId }
          : {}),
        ...(readString(lineage.title) ? { title: readString(lineage.title) } : {}),
        ...(readString(lineage.label) ? { label: readString(lineage.label) } : {}),
        ...(typeof lineage.citationCount === "number" &&
        Number.isFinite(lineage.citationCount)
          ? { citationCount: lineage.citationCount }
          : {}),
        ...(Array.isArray(lineage.citationSpans)
          ? { citationSpans: capImportList(lineage.citationSpans) }
          : {}),
        ...(Array.isArray(lineage.evidenceIds)
          ? {
              evidenceIds: capImportList(lineage.evidenceIds)
                .map(readString)
                .filter(Boolean)
            }
          : {}),
        ...(readString(lineage.coverageNotes)
          ? { coverageNotes: readString(lineage.coverageNotes) }
          : {})
      }
    })
    .filter((lineage): lineage is ArtifactSourceLineage => Boolean(lineage))

const buildCitationCounts = (
  claims: Array<Record<string, unknown>>
): Map<string, number> => {
  const counts = new Map<string, number>()
  for (const claim of claims) {
    for (const citation of readRecordList(claim.citations)) {
      const sourceId = readString(citation.source_id ?? citation.id)
      if (!sourceId) continue
      counts.set(sourceId, (counts.get(sourceId) ?? 0) + 1)
    }
  }
  return counts
}

const buildFallbackSourceLineage = (
  sources: ArtifactSourceCoverageEntry[],
  claims: Array<Record<string, unknown>>
): ArtifactSourceLineage[] => {
  const citationCounts = buildCitationCounts(claims)
  return sources.map((source) => ({
    sourceId: source.sourceId,
    ...(source.mediaId !== undefined ? { mediaId: source.mediaId } : {}),
    ...(source.title ? { title: source.title } : {}),
    citationCount: citationCounts.get(source.sourceId) ?? 0
  }))
}

const buildSourceArtifactMetadata = (
  returnContext: ResearchWorkspaceDeepResearchReturnContext,
  sourceArtifact?: GeneratedArtifact | null
) => ({
  id: sourceArtifact?.id ?? returnContext.sourceArtifactId,
  template: sourceArtifact?.templateId ?? returnContext.sourceArtifactTemplate,
  title: sourceArtifact?.title ?? returnContext.sourceArtifactTitle
})

const formatSourceEntry = (
  source: ArtifactSourceCoverageEntry,
  options?: { status?: string }
): string => {
  const label = source.title || source.sourceId
  const details = [
    source.sourceId,
    source.mediaId !== undefined ? `media #${source.mediaId}` : null
  ].filter(Boolean)
  const detailSuffix = details.length ? ` (${details.join(", ")})` : ""
  const statusSuffix = options?.status ? ` - ${options.status}` : ""
  return `- ${label}${detailSuffix}${statusSuffix}`
}

const formatSourceDetailEntry = (source: DeepResearchSourceDetail): string => {
  const label = source.title || source.sourceId
  const reasonSuffix = source.reason ? `: ${source.reason}` : ""
  return `- ${label} (${source.sourceId})${reasonSuffix}`
}

const readRecordSummary = (
  record: Record<string, unknown>,
  fallback: string
): string =>
  readString(
    record.text ??
      record.claim ??
      record.summary ??
      record.title ??
      record.reason ??
      record.source_id ??
      record.id
  ) || fallback

const buildRecordSummaryLines = (
  records: Array<Record<string, unknown>>
): string[] =>
  records.map((record, index) => `- ${readRecordSummary(record, `Item ${index + 1}`)}`)

const buildSection = (title: string, lines: string[]): string[] =>
  lines.length ? ["", `## ${title}`, "", ...lines] : []

const buildImportedContent = (options: {
  question: string
  reportMarkdown: string
  returnContext: ResearchWorkspaceDeepResearchReturnContext
  sourceTitle: string
  sourceCount: number
  selectedImportedSources: DeepResearchImportedSource[]
  sourceInventory: ArtifactSourceCoverageEntry[]
  skippedSources: DeepResearchSourceDetail[]
  failedSources: DeepResearchSourceDetail[]
  unsupportedClaims: Array<Record<string, unknown>>
  contradictions: Array<Record<string, unknown>>
  verificationSummary: Record<string, unknown>
  unresolvedQuestions: string[]
}): string => {
  const verificationBits = [
    formatCount(
      "supported claims",
      options.verificationSummary.supported_claim_count
    ),
    formatCount(
      "unsupported claims",
      options.verificationSummary.unsupported_claim_count
    )
  ].filter(Boolean)
  const unresolvedQuestions = options.unresolvedQuestions.length
    ? [
        "",
        "## Unresolved Questions",
        "",
        ...options.unresolvedQuestions.map((question) => `- ${question}`)
      ]
    : []
  const selectedImportedSources = buildSection(
    "Selected Imported Sources",
    options.selectedImportedSources.map((source) =>
      formatSourceEntry(source, { status: source.status })
    )
  )
  const sourceInventory = buildSection(
    "Source Inventory",
    options.sourceInventory.map((source) => formatSourceEntry(source))
  )
  const skippedSources = buildSection(
    "Skipped Sources",
    options.skippedSources.map(formatSourceDetailEntry)
  )
  const failedSources = buildSection(
    "Failed Sources",
    options.failedSources.map(formatSourceDetailEntry)
  )
  const unsupportedClaims = buildSection(
    "Unsupported Claims",
    buildRecordSummaryLines(options.unsupportedClaims)
  )
  const contradictions = buildSection(
    "Contradictions",
    buildRecordSummaryLines(options.contradictions)
  )

  return [
    `# Deep Research Import: ${options.sourceTitle}`,
    "",
    `Imported from Deep Research run ${options.returnContext.researchRunId}.`,
    `Source artifact: ${options.sourceTitle}`,
    `Question: ${options.question}`,
    `Source inventory: ${options.sourceCount}`,
    verificationBits.length ? `Verification: ${verificationBits.join(", ")}` : "",
    ...selectedImportedSources,
    ...sourceInventory,
    ...skippedSources,
    ...failedSources,
    ...unsupportedClaims,
    ...contradictions,
    "",
    "## Report",
    "",
    truncateForImport(options.reportMarkdown, IMPORTED_REPORT_LIMIT),
    ...unresolvedQuestions
  ]
    .filter((line) => line !== "")
    .join("\n")
}

export const buildDeepResearchBundleArtifactPayload = ({
  bundle,
  returnContext,
  sourceArtifact = null
}: DeepResearchBundleImportOptions): GeneratedArtifactPayload => {
  if (!isRecord(bundle)) {
    throw new DeepResearchBundleImportError(
      "Deep Research bundle could not be imported because it was not valid JSON."
    )
  }

  const question = readQuestion(bundle)
  const reportMarkdown = readString(bundle.report_markdown)
  if (!question) {
    throw new DeepResearchBundleImportError(
      "Deep Research bundle could not be imported because it is missing a question."
    )
  }
  if (!reportMarkdown) {
    throw new DeepResearchBundleImportError(
      "Deep Research bundle could not be imported because it is missing report markdown."
    )
  }

  const claims = readRecordList(bundle.claims)
  const sourceInventory = normalizeSourceInventory(bundle)
  const sourceCoverage = sanitizeSourceCoverage(
    sourceArtifact?.sourceCoverage ?? buildFallbackSourceCoverage(sourceInventory)
  )
  const sourceLineage = sanitizeSourceLineage(
    sourceArtifact?.sourceLineage ??
      buildFallbackSourceLineage(sourceInventory, claims)
  )
  const selectedImportedSources = buildSelectedImportedSources(
    sourceCoverage,
    sourceInventory
  )
  const skippedSources = capImportList([
    ...normalizeCoverageSkippedSources(sourceCoverage),
    ...normalizeBundleSourceDetails(bundle.skipped_sources)
  ])
  const failedSources = normalizeBundleSourceDetails(bundle.failed_sources)
  const unsupportedClaims = readRecordList(bundle.unsupported_claims)
  const contradictions = readRecordList(bundle.contradictions)
  const sourceArtifactMetadata = buildSourceArtifactMetadata(
    returnContext,
    sourceArtifact
  )
  const sourceTitle =
    sourceArtifactMetadata.title ||
    sourceArtifactMetadata.id ||
    returnContext.researchRunId
  const verificationSummary = asRecord(bundle.verification_summary)
  const unresolvedQuestions = readStringList(bundle.unresolved_questions)
  const content = buildImportedContent({
    question,
    reportMarkdown,
    returnContext,
    sourceTitle,
    sourceCount: sourceInventory.length,
    selectedImportedSources,
    sourceInventory,
    skippedSources,
    failedSources,
    unsupportedClaims,
    contradictions,
    verificationSummary,
    unresolvedQuestions
  })

  return {
    type: "report",
    title: `Deep Research: ${sourceTitle}`,
    status: "completed",
    reviewStatus: "draft",
    sourceLineage,
    sourceCoverage,
    reviewChecklist: [
      {
        id: "deep-research-source-inventory",
        label: "Review imported source inventory against workspace sources",
        checked: false
      },
      {
        id: "deep-research-unsupported-claims",
        label: "Check unsupported claims and contradictions before reuse",
        checked: false
      },
      {
        id: "deep-research-provenance",
        label: "Confirm Deep Research run provenance matches the source artifact",
        checked: false
      }
    ],
    exportTargets: ["markdown"],
    schemaVersion: 1,
    producerMetadata: {
      producerType: "deep_research_bundle_import",
      producerId: "deep_research",
      runId: returnContext.researchRunId,
      templateId: sourceArtifactMetadata.template ?? undefined
    },
    contentType: "text/markdown",
    previewText: truncateForImport(reportMarkdown, PREVIEW_LIMIT),
    summary: `Imported Deep Research bundle for: ${question}`,
    content,
    data: {
      deepResearch: {
        runId: returnContext.researchRunId,
        question,
        sourceArtifact: sourceArtifactMetadata,
        claims,
        sourceInventory: readRecordList(bundle.source_inventory),
        selectedImportedSources,
        skippedSources,
        failedSources,
        unresolvedQuestions,
        verificationSummary,
        unsupportedClaims,
        contradictions,
        sourceTrust: readRecordList(bundle.source_trust)
      }
    },
    completedAt: new Date()
  }
}
