import type {
  ArtifactSourceCoverage,
  ArtifactSourceCoverageEntry,
  ArtifactSkippedSource,
  ArtifactSourceLineage,
  GeneratedArtifact
} from "@/types/workspace"
import type { ResearchWorkspaceDeepResearchReturnContext } from "./research-workspace-route-state"

const IMPORTED_REPORT_LIMIT = 7_600
const PREVIEW_LIMIT = 280
const METADATA_TEXT_LIMIT = 500
const METADATA_ID_LIMIT = 128
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
type DeepResearchRecordSummary = {
  text: string
  claimId?: string
  noteId?: string
  sourceId?: string
  sourceIds?: string[]
  focusArea?: string
  reason?: string
  marker?: string
  supportLevel?: string
}
type DeepResearchSourceTrustSummary = {
  sourceId: string
  title?: string
  provider?: string
  sourceType?: string
  trustTier?: string
  trustLabel?: string
  snapshotPolicy?: string
  warnings?: string[]
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

const truncateMetadataText = (
  value: string,
  limit = METADATA_TEXT_LIMIT
): string => {
  if (value.length <= limit) return value
  if (limit <= 3) return value.slice(0, limit)
  return `${value.slice(0, limit - 3).trimEnd()}...`
}

const readBoundedString = (
  value: unknown,
  limit = METADATA_TEXT_LIMIT
): string => truncateMetadataText(readString(value), limit)

const readOptionalBoundedString = (
  value: unknown,
  limit = METADATA_TEXT_LIMIT
): string | null => readBoundedString(value, limit) || null

const capImportList = <T,>(items: T[]): T[] =>
  items.slice(0, MAX_IMPORT_LIST_ITEMS)

const readRecordList = (value: unknown): Array<Record<string, unknown>> =>
  Array.isArray(value) ? capImportList(value.filter(isRecord)) : []

const countRecords = (value: unknown): number =>
  Array.isArray(value) ? value.filter(isRecord).length : 0

const readStringList = (value: unknown): string[] =>
  Array.isArray(value) ? capImportList(value.map(readString).filter(Boolean)) : []

const readBoundedStringList = (
  value: unknown,
  limit = METADATA_ID_LIMIT
): string[] =>
  Array.isArray(value)
    ? capImportList(value.map((item) => readBoundedString(item, limit)).filter(Boolean))
    : []

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
      const sourceId = readBoundedString(
        entry.source_id ?? entry.sourceId ?? entry.id,
        METADATA_ID_LIMIT
      )
      if (!sourceId) return null

      const title = readBoundedString(entry.title ?? entry.label)
      const mediaId =
        typeof entry.media_id === "number" && Number.isFinite(entry.media_id)
          ? entry.media_id
          : typeof entry.mediaId === "number" && Number.isFinite(entry.mediaId)
            ? entry.mediaId
            : undefined
      return {
        sourceId,
        ...(mediaId !== undefined ? { mediaId } : {}),
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
    ...(source.title ? { title: readBoundedString(source.title) } : {}),
    status
  }))
}

const normalizeBundleSourceDetails = (
  value: unknown
): DeepResearchSourceDetail[] =>
  readRecordList(value)
    .map((entry): DeepResearchSourceDetail | null => {
      const sourceId = readBoundedString(
        entry.source_id ?? entry.sourceId ?? entry.id,
        METADATA_ID_LIMIT
      )
      if (!sourceId) return null

      const title = readBoundedString(entry.title ?? entry.label)
      const reason = readBoundedString(
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
    ...(source.title ? { title: readBoundedString(source.title) } : {}),
    reason: readBoundedString(source.reason)
  }))

const normalizeCoverageEntry = (
  source: ArtifactSourceCoverageEntry
): ArtifactSourceCoverageEntry | null => {
  const sourceId = readBoundedString(source.sourceId, METADATA_ID_LIMIT)
  if (!sourceId) return null

  return {
    sourceId,
    ...(typeof source.mediaId === "number" && Number.isFinite(source.mediaId)
      ? { mediaId: source.mediaId }
      : {}),
    ...(source.title ? { title: readBoundedString(source.title) } : {})
  }
}

const normalizeSkippedCoverageEntry = (
  source: ArtifactSkippedSource
): ArtifactSkippedSource | null => {
  const normalized = normalizeCoverageEntry(source)
  if (!normalized) return null
  const reason =
    source.reason === "missing_text" ||
    source.reason === "unready" ||
    source.reason === "context_limit" ||
    source.reason === "unknown"
      ? source.reason
      : "unknown"

  return {
    ...normalized,
    reason
  }
}

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
    sourceCoverage.selectedSourceIds
      .map((sourceId) => readBoundedString(sourceId, METADATA_ID_LIMIT))
      .filter(Boolean)
  ),
  usableSources: capImportList(sourceCoverage.usableSources)
    .map(normalizeCoverageEntry)
    .filter((source): source is ArtifactSourceCoverageEntry => Boolean(source)),
  skippedSources: capImportList(sourceCoverage.skippedSources)
    .map(normalizeSkippedCoverageEntry)
    .filter((source): source is ArtifactSkippedSource => Boolean(source)),
  truncatedSources: capImportList(sourceCoverage.truncatedSources)
    .map(normalizeCoverageEntry)
    .filter((source): source is ArtifactSourceCoverageEntry => Boolean(source)),
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
        ...(readBoundedString(lineage.sourceType)
          ? { sourceType: readBoundedString(lineage.sourceType) }
          : {}),
        ...(typeof lineage.mediaId === "number" && Number.isFinite(lineage.mediaId)
          ? { mediaId: lineage.mediaId }
          : {}),
        ...(readBoundedString(lineage.title)
          ? { title: readBoundedString(lineage.title) }
          : {}),
        ...(readBoundedString(lineage.label)
          ? { label: readBoundedString(lineage.label) }
          : {}),
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
                .map((evidenceId) =>
                  readBoundedString(evidenceId, METADATA_ID_LIMIT)
                )
                .filter(Boolean)
            }
          : {}),
        ...(readBoundedString(lineage.coverageNotes)
          ? { coverageNotes: readBoundedString(lineage.coverageNotes) }
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
  id: readOptionalBoundedString(
    sourceArtifact?.id ?? returnContext.sourceArtifactId,
    METADATA_ID_LIMIT
  ),
  template: readOptionalBoundedString(
    sourceArtifact?.templateId ?? returnContext.sourceArtifactTemplate,
    METADATA_ID_LIMIT
  ),
  title: readOptionalBoundedString(
    sourceArtifact?.title ?? returnContext.sourceArtifactTitle
  )
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

const readRecordSummaryText = (
  record: Record<string, unknown>,
  fallback: string
): string =>
  readBoundedString(
    record.text ??
      record.claim ??
      record.summary ??
      record.title ??
      record.reason ??
      record.source_id ??
      record.id
  ) || fallback

const buildRecordSummaryLines = (
  records: DeepResearchRecordSummary[]
): string[] =>
  records.map((record, index) => `- ${record.text || `Item ${index + 1}`}`)

const normalizeResearchRecords = (
  value: unknown
): DeepResearchRecordSummary[] =>
  readRecordList(value).map((record, index) => {
    const claimId = readBoundedString(
      record.claim_id ?? record.claimId ?? record.id,
      METADATA_ID_LIMIT
    )
    const noteId = readBoundedString(record.note_id ?? record.noteId, METADATA_ID_LIMIT)
    const sourceId = readBoundedString(
      record.source_id ?? record.sourceId,
      METADATA_ID_LIMIT
    )
    const sourceIds = readBoundedStringList(record.source_ids ?? record.sourceIds)
    const focusArea = readBoundedString(record.focus_area ?? record.focusArea)
    const reason = readBoundedString(record.reason)
    const marker = readBoundedString(record.marker)
    const supportLevel = readBoundedString(
      record.support_level ?? record.supportLevel
    )

    return {
      text: readRecordSummaryText(record, `Item ${index + 1}`),
      ...(claimId ? { claimId } : {}),
      ...(noteId ? { noteId } : {}),
      ...(sourceId ? { sourceId } : {}),
      ...(sourceIds.length ? { sourceIds } : {}),
      ...(focusArea ? { focusArea } : {}),
      ...(reason ? { reason } : {}),
      ...(marker ? { marker } : {}),
      ...(supportLevel ? { supportLevel } : {})
    }
  })

const normalizeSourceTrust = (value: unknown): DeepResearchSourceTrustSummary[] =>
  readRecordList(value)
    .map((entry): DeepResearchSourceTrustSummary | null => {
      const sourceId = readBoundedString(
        entry.source_id ?? entry.sourceId ?? entry.id,
        METADATA_ID_LIMIT
      )
      if (!sourceId) return null

      const title = readBoundedString(entry.title ?? entry.label)
      const provider = readBoundedString(entry.provider)
      const sourceType = readBoundedString(entry.source_type ?? entry.sourceType)
      const trustTier = readBoundedString(entry.trust_tier ?? entry.trustTier)
      const trustLabel = readBoundedString(entry.trust_label ?? entry.trustLabel)
      const snapshotPolicy = readBoundedString(
        entry.snapshot_policy ?? entry.snapshotPolicy
      )
      const warnings = readBoundedStringList(entry.warnings)

      return {
        sourceId,
        ...(title ? { title } : {}),
        ...(provider ? { provider } : {}),
        ...(sourceType ? { sourceType } : {}),
        ...(trustTier ? { trustTier } : {}),
        ...(trustLabel ? { trustLabel } : {}),
        ...(snapshotPolicy ? { snapshotPolicy } : {}),
        ...(warnings.length ? { warnings } : {})
      }
    })
    .filter((entry): entry is DeepResearchSourceTrustSummary => Boolean(entry))

const formatSourceInventoryCount = (
  sourceCount: number,
  shownSourceCount: number
): string =>
  sourceCount > shownSourceCount
    ? `${sourceCount} (${shownSourceCount} shown)`
    : String(sourceCount)

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
  unsupportedClaims: DeepResearchRecordSummary[]
  contradictions: DeepResearchRecordSummary[]
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
    `Source inventory: ${formatSourceInventoryCount(
      options.sourceCount,
      options.sourceInventory.length
    )}`,
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

  const rawClaims = readRecordList(bundle.claims)
  const claims = normalizeResearchRecords(bundle.claims)
  const sourceInventory = normalizeSourceInventory(bundle)
  const sourceInventoryCount = countRecords(bundle.source_inventory)
  const sourceCoverage = sanitizeSourceCoverage(
    sourceArtifact?.sourceCoverage ?? buildFallbackSourceCoverage(sourceInventory)
  )
  const sourceLineage = sanitizeSourceLineage(
    sourceArtifact?.sourceLineage ??
      buildFallbackSourceLineage(sourceInventory, rawClaims)
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
  const unsupportedClaims = normalizeResearchRecords(bundle.unsupported_claims)
  const contradictions = normalizeResearchRecords(bundle.contradictions)
  const sourceTrust = normalizeSourceTrust(bundle.source_trust)
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
    sourceCount: sourceInventoryCount,
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
        sourceInventory,
        selectedImportedSources,
        skippedSources,
        failedSources,
        unresolvedQuestions,
        verificationSummary,
        unsupportedClaims,
        contradictions,
        sourceTrust
      }
    },
    completedAt: new Date()
  }
}
