import type { TestType } from "@playwright/test"
import { existsSync, readFileSync } from "node:fs"
import path from "node:path"
import {
  launchWithExtension,
  type LaunchWithExtensionResult
} from "./extension"
import { launchWithBuiltExtension } from "./extension-build"

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

const KNOWLEDGE_QA_MANIFEST_ENV = "TLDW_KNOWLEDGE_QA_FIXTURE_MANIFEST"
const KNOWLEDGE_QA_SOURCE_KEYS: KnowledgeQaLiveSourceKey[] = [
  "cited_media",
  "distractor_media",
  "scoped_note"
]

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const normalizeServerUrl = (value: string): string => {
  const trimmed = value.trim()
  if (!trimmed) return ""
  const withProtocol = /^https?:\/\//i.test(trimmed) ? trimmed : `http://${trimmed}`
  return withProtocol.replace(/\/$/, "")
}

/**
 * Read real tldw_server config for E2E tests.
 *
 * Tests that rely on a real server should call this at the top of the test
 * body. If the required env vars are not set, the test is skipped with a
 * clear message instead of attempting to spin up a mock server.
 *
 * Required env vars:
 * - TLDW_E2E_SERVER_URL  (e.g. http://127.0.0.1:3001)
 * - TLDW_E2E_API_KEY     (API key accepted by that server)
 */
export const requireRealServerConfig = (
  test: TestType<any, any>
): { serverUrl: string; apiKey: string } => {
  const serverUrl = process.env.TLDW_E2E_SERVER_URL
  const apiKey = process.env.TLDW_E2E_API_KEY

  if (!serverUrl || !apiKey) {
    test.skip(
      true,
      "Set TLDW_E2E_SERVER_URL and TLDW_E2E_API_KEY to run real-server E2E tests."
    )
    return { serverUrl: "", apiKey: "" }
  }

  return { serverUrl: serverUrl!, apiKey: apiKey! }
}

export const requireRealServerConfigStrict = (): { serverUrl: string; apiKey: string } => {
  const serverUrl = normalizeServerUrl(process.env.TLDW_E2E_SERVER_URL || "")
  const apiKey = String(process.env.TLDW_E2E_API_KEY || "").trim()

  if (!serverUrl || !apiKey) {
    throw new Error(
      "Knowledge QA extension live UAT requires TLDW_E2E_SERVER_URL and TLDW_E2E_API_KEY. " +
        "These tests are release gates and must fail rather than skip when live-server config is missing."
    )
  }

  return { serverUrl, apiKey }
}

export const assertRealServerHealth = async (
  serverUrl: string,
  apiKey: string
): Promise<void> => {
  const normalizedServerUrl = normalizeServerUrl(serverUrl)
  const response = await fetch(`${normalizedServerUrl}/api/v1/health`, {
    headers: { "X-API-KEY": apiKey }
  }).catch((error) => {
    throw new Error(
      `Knowledge QA extension live UAT could not reach ${normalizedServerUrl}/api/v1/health: ${String(error)}`
    )
  })

  if (!response.ok) {
    const body = await response.text().catch(() => "")
    throw new Error(
      `Knowledge QA extension live UAT backend health failed at ${normalizedServerUrl}/api/v1/health ` +
        `(HTTP ${response.status}). ${body.slice(0, 500)}`
    )
  }
}

export const getKnowledgeQaLiveManifestPath = (): string => {
  const manifestPath = String(process.env[KNOWLEDGE_QA_MANIFEST_ENV] || "").trim()
  if (!manifestPath) {
    throw new Error(
      `${KNOWLEDGE_QA_MANIFEST_ENV} is required for Knowledge QA extension live UAT. ` +
        "Run Helper_Scripts/seed_knowledge_qa_uat.py and point this env var at the generated manifest."
    )
  }
  return path.resolve(manifestPath)
}

const assertManifestString = (
  value: unknown,
  label: string,
  manifestPath: string
): asserts value is string => {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${manifestPath} has invalid ${label}; expected a non-empty string.`)
  }
}

const assertKnowledgeQaLiveManifest = (
  value: unknown,
  manifestPath: string
): asserts value is KnowledgeQaLiveManifest => {
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
    assertManifestString(queryValue, `queries.${queryKey}`, manifestPath)
  }
  for (const [expectedKey, expectedValue] of Object.entries(value.expected)) {
    assertManifestString(expectedValue, `expected.${expectedKey}`, manifestPath)
  }
  for (const key of KNOWLEDGE_QA_SOURCE_KEYS) {
    const source = value.sources[key]
    if (!isRecord(source)) {
      throw new Error(`${manifestPath} is missing sources.${key}.`)
    }
    assertManifestString(source.key, `sources.${key}.key`, manifestPath)
    assertManifestString(source.title, `sources.${key}.title`, manifestPath)
    assertManifestString(source.body, `sources.${key}.body`, manifestPath)
    assertManifestString(source.source_type, `sources.${key}.source_type`, manifestPath)
    if (source.key !== key) {
      throw new Error(`${manifestPath} source key mismatch: expected ${key}.`)
    }
    if (source.source_type !== "media_db" && source.source_type !== "notes") {
      throw new Error(`${manifestPath} sources.${key}.source_type is not supported.`)
    }
  }
}

export const loadKnowledgeQaLiveManifest = (): KnowledgeQaLiveManifest => {
  const manifestPath = getKnowledgeQaLiveManifestPath()
  if (!existsSync(manifestPath)) {
    throw new Error(
      `Knowledge QA extension live fixture manifest was not found at ${manifestPath}. ` +
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
      `Knowledge QA extension live fixture source ${key} has no seeded id. ` +
        "Run Helper_Scripts/seed_knowledge_qa_uat.py without --dry-run before browser UAT."
    )
  }
  return id
}

/**
 * Launch the extension for real-server E2E tests with a bounded startup timeout.
 *
 * If the browser/extension cannot start in the current environment, the test is
 * skipped with a clear message instead of timing out for the full test duration.
 */
export const launchWithExtensionOrSkip = async (
  test: TestType<any, any>,
  extensionPath: string,
  options: Parameters<typeof launchWithExtension>[1] = {}
): Promise<LaunchWithExtensionResult> => {
  try {
    return await launchWithExtension(extensionPath, {
      ...(options || {})
    })
  } catch (error) {
    test.skip(
      true,
      `Extension launch unavailable in this environment (${String(error)}).`
    )
    return undefined as never
  }
}

export const launchWithBuiltExtensionOrSkip = async (
  test: TestType<any, any>,
  options: Parameters<typeof launchWithBuiltExtension>[0] = {}
): Promise<Awaited<ReturnType<typeof launchWithBuiltExtension>>> => {
  try {
    return await launchWithBuiltExtension({
      ...(options || {})
    })
  } catch (error) {
    test.skip(
      true,
      `Extension launch unavailable in this environment (${String(error)}).`
    )
    return undefined as never
  }
}

export const launchWithBuiltExtensionForLiveUat = async (
  options: Parameters<typeof launchWithBuiltExtension>[0] = {}
): Promise<Awaited<ReturnType<typeof launchWithBuiltExtension>>> => {
  try {
    return await launchWithBuiltExtension({
      ...(options || {})
    })
  } catch (error) {
    throw new Error(
      "Knowledge QA extension live UAT could not launch the packaged extension route. " +
        "This is release-blocking for extension coverage; inspect WXT build output and extension target availability.",
      { cause: error }
    )
  }
}
