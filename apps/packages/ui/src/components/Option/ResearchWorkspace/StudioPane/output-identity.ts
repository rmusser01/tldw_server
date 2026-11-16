import type { GeneratedArtifact } from "@/types/workspace"

const record = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value)
const text = (value: unknown): string | undefined =>
  typeof value === "string" && value.trim() ? value.trim() : undefined
const sectionKey = (row: Record<string, unknown>) => JSON.stringify([row.source_id, row.node_id, row.navigation_version])
const selections = (value: unknown): Record<string, unknown>[] | undefined => {
  if (!Array.isArray(value) || value.length === 0 || value.length > 500 ||
      !value.every(row => record(row) && text(row.source_id) && text(row.node_id) && text(row.navigation_version))) return undefined
  return new Set(value.map(sectionKey)).size === value.length ? value : undefined
}

/** Read saved generation identity only; current model choices confer no history. */
export function getArtifactRunIdentity(artifact: GeneratedArtifact): {
  model?: string; provider?: string; createdAt?: Date; imported: boolean;
  sectionCount?: number; sectionTitles: string[]
} {
  let producer: Record<string, unknown> | undefined = record(artifact.producerMetadata) ? artifact.producerMetadata : undefined
  let imported = false
  const seen = new Set<Record<string, unknown>>()
  while (producer?.origin === "workspace_snapshot_import") {
    imported = true
    if (seen.has(producer) || seen.size >= 5) { producer = undefined; break }
    seen.add(producer)
    producer = record(producer.importedProducer) ? producer.importedProducer : undefined
  }
  const settings = record(producer?.settings) ? producer.settings : undefined
  const modelMetadata = producer?.origin === "research_workspace_output_job" ? settings : producer
  const client = artifact.versionMetadata?.workspaceClient
  const restoredData = record(client) && client.schemaVersion === 1 && record(client.data) ? client.data : undefined
  const analysis = artifact.data?.sectionAnalysis ?? restoredData?.sectionAnalysis
  const rows = record(analysis) && analysis.schemaVersion === 1 ? selections(analysis.sections) : undefined
  const scope = selections(settings?.sections) ?? rows
  const titles = new Map((rows ?? []).map(row => [sectionKey(row), text(row.title)]))
  const rawDate: unknown = artifact.createdAt
  const date = rawDate instanceof Date ? new Date(rawDate.getTime())
    : typeof rawDate === "string" && /^\d{4}-\d{2}-\d{2}T/.test(rawDate) ? new Date(rawDate) : undefined

  return {
    model: text(modelMetadata?.model), provider: text(modelMetadata?.provider), imported,
    createdAt: date && Number.isFinite(date.getTime()) ? date : undefined,
    sectionCount: scope?.length,
    sectionTitles: (scope ?? []).map(row => titles.get(sectionKey(row))).filter((title): title is string => Boolean(title)).slice(0, 3)
  }
}
