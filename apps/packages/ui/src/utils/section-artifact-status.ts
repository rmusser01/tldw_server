import type { GeneratedArtifact } from "@/types/workspace"

export const INCOMPLETE_SECTIONS_MESSAGE = "Some sections are incomplete. Retry to continue from saved portions."

const record = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value)

/** Derive missing warnings and refine old generic warnings without replacing specific errors or saved evidence. */
export const getSectionArtifactErrorMessage = (artifact: GeneratedArtifact): string | undefined => {
  if (artifact.status !== "failed" || (artifact.errorMessage && artifact.errorMessage !== INCOMPLETE_SECTIONS_MESSAGE) ||
      artifact.producerMetadata?.error) return artifact.errorMessage

  const client = artifact.versionMetadata?.workspaceClient
  const restoredData = record(client) && record(client.data) ? client.data : undefined
  const analysis = artifact.data?.sectionAnalysis ?? restoredData?.sectionAnalysis
  if (!record(analysis) || analysis.schemaVersion !== 1 ||
      analysis.stopped_error || analysis.stopped_error_code ||
      !Array.isArray(analysis.sections) || analysis.sections.length === 0 ||
      !analysis.sections.every(section => record(section) &&
        typeof section.title === "string" && typeof section.status === "string" &&
        ["unread", "generating", "completed", "failed"].includes(section.status))) return artifact.errorMessage

  if (analysis.sections.some(section => section.status !== "completed"))
    return artifact.errorMessage || INCOMPLETE_SECTIONS_MESSAGE
  if (typeof analysis.synthesis_error !== "string" || !analysis.synthesis_error.trim()) return artifact.errorMessage

  return "All selected sections are complete. The overview could not be completed. Retry to rebuild the overview from saved evidence."
}
