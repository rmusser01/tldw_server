import { buildResearchLaunchPath } from "@/routes/route-paths"
import type {
  ArtifactSourceCoverage,
  ArtifactSourceCoverageEntry,
  ArtifactSkippedSource,
  GeneratedArtifact
} from "@/types/workspace"

const LITERATURE_DEEP_RESEARCH_TEMPLATE_IDS = new Set([
  "literature_matrix",
  "corpus_gap_finder"
])

const ARTIFACT_EXCERPT_LIMIT = 1400
const RESEARCH_QUERY_LIMIT = 2600
const TRUNCATION_SUFFIX = "\n\n[Truncated for Deep Research launch context.]"

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

const formatSourceCoverage = (sourceCoverage: ArtifactSourceCoverage): string =>
  [
    `Selected source IDs: ${(sourceCoverage.selectedSourceIds ?? []).join(", ") || "none"}`,
    `Usable sources: ${
      (sourceCoverage.usableSources ?? []).map(formatCoverageEntry).join(", ") ||
      "none"
    }`,
    `Skipped sources: ${
      (sourceCoverage.skippedSources ?? []).map(formatSkippedSource).join(", ") ||
      "none"
    }`,
    `Truncated sources: ${
      (sourceCoverage.truncatedSources ?? []).map(formatCoverageEntry).join(", ") ||
      "none"
    }`
  ].join("\n")

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
  artifact: GeneratedArtifact
): string | null => {
  const query = buildLiteratureDeepResearchLaunchQuery(artifact)
  if (!query) {
    return null
  }

  return buildResearchLaunchPath({
    query,
    sourcePolicy: "local_first",
    autonomyMode: "checkpointed",
    from: "research-workspace"
  })
}
