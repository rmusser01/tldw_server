import type { ArtifactType } from "@/types/workspace"

export const RESEARCH_STUDIO_CAPABILITY_IDS = [
  "source_browse",
  "chat",
  "artifact_text_generation",
  "slides_generation",
  "audio_summary",
  "export_download",
  "sync_share"
] as const

export type ResearchStudioCapabilityId =
  (typeof RESEARCH_STUDIO_CAPABILITY_IDS)[number]
export type ResearchStudioCapabilityStatus =
  | "ready"
  | "degraded"
  | "unavailable"
  | "unknown"
export type ResearchStudioCapabilityMode = "allow" | "warn" | "block"

export type ResearchStudioCapability = {
  status: ResearchStudioCapabilityStatus
  mode: ResearchStudioCapabilityMode
  dependencies: string[]
  reason_code?: string | null
}

export type ResearchStudioCapabilitiesResponse = {
  status: ResearchStudioCapabilityStatus
  ttl_seconds: number
  capabilities: Record<ResearchStudioCapabilityId, ResearchStudioCapability>
  timestamp: string
}

const VALID_STATUSES = new Set<ResearchStudioCapabilityStatus>([
  "ready",
  "degraded",
  "unavailable",
  "unknown"
])

const VALID_MODES = new Set<ResearchStudioCapabilityMode>([
  "allow",
  "warn",
  "block"
])

const DEFAULT_TTL_SECONDS = 30

export function buildUnknownResearchStudioCapabilities(
  reasonCode = "capability_health_unknown"
): ResearchStudioCapabilitiesResponse {
  const capabilities = Object.fromEntries(
    RESEARCH_STUDIO_CAPABILITY_IDS.map((id) => [
      id,
      buildUnknownCapability(reasonCode)
    ])
  ) as Record<ResearchStudioCapabilityId, ResearchStudioCapability>

  return {
    status: "unknown",
    ttl_seconds: DEFAULT_TTL_SECONDS,
    capabilities,
    timestamp: new Date().toISOString()
  }
}

export function normalizeResearchStudioCapabilities(
  raw: unknown
): ResearchStudioCapabilitiesResponse {
  const record = isRecord(raw) ? raw : {}
  const rawCapabilities = isRecord(record.capabilities)
    ? record.capabilities
    : {}
  const fallback = buildUnknownResearchStudioCapabilities()
  const capabilities = { ...fallback.capabilities }

  for (const id of RESEARCH_STUDIO_CAPABILITY_IDS) {
    capabilities[id] = normalizeCapability(rawCapabilities[id])
  }

  return {
    status: isCapabilityStatus(record.status) ? record.status : "unknown",
    ttl_seconds: normalizeTtl(record.ttl_seconds),
    capabilities,
    timestamp:
      typeof record.timestamp === "string" && record.timestamp.trim()
        ? record.timestamp
        : fallback.timestamp
  }
}

export function isResearchStudioCapabilitiesStale(
  payload: Pick<ResearchStudioCapabilitiesResponse, "ttl_seconds"> | null | undefined,
  fetchedAtMs: number | null | undefined,
  nowMs = Date.now()
): boolean {
  if (!payload || typeof fetchedAtMs !== "number" || !Number.isFinite(fetchedAtMs)) {
    return true
  }
  return nowMs - fetchedAtMs > normalizeTtl(payload.ttl_seconds) * 1000
}

export function getCapability(
  payload: ResearchStudioCapabilitiesResponse | null | undefined,
  id: ResearchStudioCapabilityId
): ResearchStudioCapability {
  return payload?.capabilities[id] ?? buildUnknownCapability()
}

export function getArtifactCapabilityId(
  type: ArtifactType
): ResearchStudioCapabilityId {
  if (type === "slides") return "slides_generation"
  if (type === "audio_overview") return "audio_summary"
  return "artifact_text_generation"
}

export function getCapabilityCopy(
  capability: ResearchStudioCapability,
  actionLabel: string
): string | null {
  if (capability.mode === "allow") return null
  const label = actionLabel.trim() || "This action"
  if (capability.mode === "block") {
    return `${label} is unavailable while required services are offline.`
  }
  return `${label} may be degraded. You can still try it.`
}

function normalizeCapability(raw: unknown): ResearchStudioCapability {
  if (!isRecord(raw)) return buildUnknownCapability()
  const status = isCapabilityStatus(raw.status) ? raw.status : "unknown"
  const mode = isCapabilityMode(raw.mode) ? raw.mode : "warn"
  const dependencies = Array.isArray(raw.dependencies)
    ? raw.dependencies.filter((item): item is string => typeof item === "string")
    : []
  const reason_code =
    typeof raw.reason_code === "string" && raw.reason_code.trim()
      ? raw.reason_code
      : null

  if (!isCapabilityStatus(raw.status) || !isCapabilityMode(raw.mode)) {
    return buildUnknownCapability()
  }

  return {
    status,
    mode,
    dependencies,
    reason_code
  }
}

function buildUnknownCapability(
  reasonCode = "capability_health_unknown"
): ResearchStudioCapability {
  return {
    status: "unknown",
    mode: "warn",
    dependencies: [],
    reason_code: reasonCode
  }
}

function normalizeTtl(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? value
    : DEFAULT_TTL_SECONDS
}

function isCapabilityStatus(value: unknown): value is ResearchStudioCapabilityStatus {
  return typeof value === "string" && VALID_STATUSES.has(value as ResearchStudioCapabilityStatus)
}

function isCapabilityMode(value: unknown): value is ResearchStudioCapabilityMode {
  return typeof value === "string" && VALID_MODES.has(value as ResearchStudioCapabilityMode)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value)
}
