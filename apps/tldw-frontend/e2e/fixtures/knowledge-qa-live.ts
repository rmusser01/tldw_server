import { existsSync, readFileSync } from "node:fs"
import path from "node:path"

export type KnowledgeQaLiveSourceKey =
  | "cited_media"
  | "distractor_media"
  | "scoped_note"

export interface KnowledgeQaLiveSource {
  key: KnowledgeQaLiveSourceKey
  title: string
  body: string
  source_type: "media_db" | "notes"
  id: string | number | null
}

export interface KnowledgeQaLiveManifest {
  schemaVersion: number
  queries: {
    cited: string
    noMatch: string
    scopedIncluded: string
    degradedUncited: string
  }
  expected: {
    citedAnswerPhrase: string
    distractorPhrase: string
    scopedExcludedPhrase: string
    scopedIncludedPhrase: string
    degradedUncitedAnswer: string
  }
  sources: Record<KnowledgeQaLiveSourceKey, KnowledgeQaLiveSource>
}

const MANIFEST_ENV = "TLDW_KNOWLEDGE_QA_FIXTURE_MANIFEST"

const requiredSourceKeys: KnowledgeQaLiveSourceKey[] = [
  "cited_media",
  "distractor_media",
  "scoped_note",
]

export const getKnowledgeQaLiveManifestPath = (): string => {
  const manifestPath = process.env[MANIFEST_ENV]?.trim()
  if (!manifestPath) {
    throw new Error(
      `${MANIFEST_ENV} is required for Knowledge QA live backend UAT. ` +
        "Run Helper_Scripts/seed_knowledge_qa_uat.py and point this env var at the generated manifest."
    )
  }
  return path.resolve(manifestPath)
}

export const loadKnowledgeQaLiveManifest = (): KnowledgeQaLiveManifest => {
  const manifestPath = getKnowledgeQaLiveManifestPath()
  if (!existsSync(manifestPath)) {
    throw new Error(
      `Knowledge QA live fixture manifest was not found at ${manifestPath}. ` +
        "Seed the backend before running this release gate."
    )
  }

  const parsed = JSON.parse(readFileSync(manifestPath, "utf8")) as unknown
  assertKnowledgeQaLiveManifest(parsed, manifestPath)
  return parsed
}

export const getRequiredKnowledgeQaLiveSourceId = (
  manifest: KnowledgeQaLiveManifest,
  key: KnowledgeQaLiveSourceKey
): string | number => {
  const id = manifest.sources[key]?.id
  if (id === null || typeof id === "undefined" || String(id).trim() === "") {
    throw new Error(
      `Knowledge QA live fixture source ${key} has no seeded id. ` +
        "Run Helper_Scripts/seed_knowledge_qa_uat.py without --dry-run before browser UAT."
    )
  }
  return id
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const assertString = (
  value: unknown,
  label: string,
  manifestPath: string
): asserts value is string => {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${manifestPath} has invalid ${label}; expected a non-empty string.`)
  }
}

function assertKnowledgeQaLiveManifest(
  value: unknown,
  manifestPath: string
): asserts value is KnowledgeQaLiveManifest {
  if (!isRecord(value)) {
    throw new Error(`${manifestPath} is not a Knowledge QA live fixture manifest object.`)
  }
  if (value.schemaVersion !== 1) {
    throw new Error(`${manifestPath} has unsupported schemaVersion ${String(value.schemaVersion)}.`)
  }
  if (!isRecord(value.queries) || !isRecord(value.expected) || !isRecord(value.sources)) {
    throw new Error(`${manifestPath} is missing queries, expected, or sources sections.`)
  }

  for (const [queryKey, queryValue] of Object.entries(value.queries)) {
    assertString(queryValue, `queries.${queryKey}`, manifestPath)
  }
  for (const [expectedKey, expectedValue] of Object.entries(value.expected)) {
    assertString(expectedValue, `expected.${expectedKey}`, manifestPath)
  }
  for (const key of requiredSourceKeys) {
    const source = value.sources[key]
    if (!isRecord(source)) {
      throw new Error(`${manifestPath} is missing sources.${key}.`)
    }
    assertString(source.key, `sources.${key}.key`, manifestPath)
    assertString(source.title, `sources.${key}.title`, manifestPath)
    assertString(source.body, `sources.${key}.body`, manifestPath)
    assertString(source.source_type, `sources.${key}.source_type`, manifestPath)
    if (source.key !== key) {
      throw new Error(`${manifestPath} source key mismatch: expected ${key}.`)
    }
    if (source.source_type !== "media_db" && source.source_type !== "notes") {
      throw new Error(`${manifestPath} sources.${key}.source_type is not supported.`)
    }
  }
}
