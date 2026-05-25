import type { ArtifactType } from "@/types/workspace"

export const RESEARCH_WORKSPACE_CAPABILITY_IDS = [
  "source_browse",
  "chat",
  "artifact_text_generation",
  "slides_generation",
  "audio_summary",
  "export_download",
  "sync_share"
] as const

export type ResearchWorkspaceCapabilityId =
  (typeof RESEARCH_WORKSPACE_CAPABILITY_IDS)[number]
export type ResearchWorkspaceCapabilityStatus =
  | "ready"
  | "degraded"
  | "unavailable"
  | "unknown"
export type ResearchWorkspaceCapabilityMode = "allow" | "warn" | "block"

export type ResearchWorkspaceCapability = {
  status: ResearchWorkspaceCapabilityStatus
  mode: ResearchWorkspaceCapabilityMode
  dependencies: string[]
  reason_code?: string | null
}

export type ResearchWorkspaceCapabilitiesResponse = {
  status: ResearchWorkspaceCapabilityStatus
  ttl_seconds: number
  capabilities: Record<ResearchWorkspaceCapabilityId, ResearchWorkspaceCapability>
  timestamp: string
}

const VALID_STATUSES = new Set<ResearchWorkspaceCapabilityStatus>([
  "ready",
  "degraded",
  "unavailable",
  "unknown"
])

const VALID_MODES = new Set<ResearchWorkspaceCapabilityMode>([
  "allow",
  "warn",
  "block"
])

const DEFAULT_TTL_SECONDS = 30

export function buildUnknownResearchWorkspaceCapabilities(
  reasonCode = "capability_health_unknown"
): ResearchWorkspaceCapabilitiesResponse {
  const capabilities = Object.fromEntries(
    RESEARCH_WORKSPACE_CAPABILITY_IDS.map((id) => [
      id,
      buildUnknownCapability(reasonCode)
    ])
  ) as Record<ResearchWorkspaceCapabilityId, ResearchWorkspaceCapability>

  return {
    status: "unknown",
    ttl_seconds: DEFAULT_TTL_SECONDS,
    capabilities,
    timestamp: new Date().toISOString()
  }
}

export function normalizeResearchWorkspaceCapabilities(
  raw: unknown
): ResearchWorkspaceCapabilitiesResponse {
  const record = isRecord(raw) ? raw : {}
  const rawCapabilities = isRecord(record.capabilities)
    ? record.capabilities
    : {}
  const fallback = buildUnknownResearchWorkspaceCapabilities()
  const capabilities = { ...fallback.capabilities }

  for (const id of RESEARCH_WORKSPACE_CAPABILITY_IDS) {
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

export function isResearchWorkspaceCapabilitiesStale(
  payload: Pick<ResearchWorkspaceCapabilitiesResponse, "ttl_seconds"> | null | undefined,
  fetchedAtMs: number | null | undefined,
  nowMs = Date.now()
): boolean {
  if (!payload || typeof fetchedAtMs !== "number" || !Number.isFinite(fetchedAtMs)) {
    return true
  }
  return nowMs - fetchedAtMs > normalizeTtl(payload.ttl_seconds) * 1000
}

export function getCapability(
  payload: ResearchWorkspaceCapabilitiesResponse | null | undefined,
  id: ResearchWorkspaceCapabilityId
): ResearchWorkspaceCapability {
  return payload?.capabilities[id] ?? buildUnknownCapability()
}

export function getArtifactCapabilityId(
  type: ArtifactType
): ResearchWorkspaceCapabilityId {
  if (type === "slides") return "slides_generation"
  if (type === "audio_overview") return "audio_summary"
  return "artifact_text_generation"
}

export function getCapabilityCopy(
  capability: ResearchWorkspaceCapability,
  actionLabel: string
): string | null {
  if (capability.mode === "allow") return null
  const label = actionLabel.trim() || "This action"
  if (capability.mode === "block") {
    return `${label} is unavailable while required services are offline.`
  }
  return `${label} may be degraded. You can still try it.`
}

function normalizeCapability(raw: unknown): ResearchWorkspaceCapability {
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
): ResearchWorkspaceCapability {
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

function isCapabilityStatus(value: unknown): value is ResearchWorkspaceCapabilityStatus {
  return typeof value === "string" && VALID_STATUSES.has(value as ResearchWorkspaceCapabilityStatus)
}

function isCapabilityMode(value: unknown): value is ResearchWorkspaceCapabilityMode {
  return typeof value === "string" && VALID_MODES.has(value as ResearchWorkspaceCapabilityMode)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value)
}
