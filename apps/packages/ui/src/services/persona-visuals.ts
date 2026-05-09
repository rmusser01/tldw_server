import { tldwClient } from "@/services/tldw/TldwApiClient"
import type {
  PersonaVisualAsset,
  PersonaVisualAssetRole,
  PersonaVisualCandidate,
  PersonaVisualCandidateListResponse,
  PersonaVisualCandidateReviewRequest,
  PersonaVisualDeactivateResponse,
  PersonaVisualGenerationJobResponse,
  PersonaVisualGenerationRequest,
  PersonaVisualManifestUpdate,
  PersonaVisualPack,
  PersonaVisualPackCreate,
  PersonaVisualPackListResponse
} from "@/types/persona-visuals"

type PersonaVisualFetchInit = {
  method?: string
  body?: BodyInit | object | null
  headers?: Record<string, string>
}

export class PersonaVisualApiError extends Error {
  status?: number
  detail?: unknown

  constructor(
    message: string,
    options: { status?: number; detail?: unknown } = {}
  ) {
    super(message)
    this.name = "PersonaVisualApiError"
    this.status = options.status
    this.detail = options.detail
  }
}

const personaVisualPath = (
  personaId: string,
  suffix = ""
): `/api/v1/persona/profiles/${string}${string}` =>
  `/api/v1/persona/profiles/${encodeURIComponent(personaId)}${suffix}`

const packPath = (
  personaId: string,
  packId: string,
  suffix = ""
): `/api/v1/persona/profiles/${string}/visual-packs/${string}${string}` =>
  personaVisualPath(
    personaId,
    `/visual-packs/${encodeURIComponent(packId)}${suffix}`
  ) as `/api/v1/persona/profiles/${string}/visual-packs/${string}${string}`

const normalizeBody = (
  body: PersonaVisualFetchInit["body"],
  headers: Record<string, string>
): BodyInit | undefined => {
  if (body == null) return undefined
  if (body instanceof FormData) return body
  if (typeof body === "string") return body
  headers["Content-Type"] = headers["Content-Type"] || "application/json"
  return JSON.stringify(body)
}

export async function fetchPersonaVisualJson<T>(
  path: string,
  init: PersonaVisualFetchInit = {}
): Promise<T> {
  const headers = { ...(init.headers || {}) }
  const response = await tldwClient.fetchWithAuth(path as any, {
    method: init.method || "GET",
    headers,
    body: normalizeBody(init.body, headers)
  })
  const payload = await response.json().catch(() => null)
  if (!response.ok) {
    const detail =
      payload && typeof payload === "object" && "detail" in payload
        ? (payload as { detail?: unknown }).detail
        : payload
    const message =
      response.error ||
      (detail && typeof detail === "object" && "message" in detail
        ? String((detail as { message?: unknown }).message || "")
        : "") ||
      "Persona visual API request failed"
    throw new PersonaVisualApiError(message, {
      status: response.status,
      detail
    })
  }
  return payload as T
}

export const normalizePersonaVisualPackList = (
  payload: PersonaVisualPack[] | PersonaVisualPackListResponse
): PersonaVisualPackListResponse => {
  if (Array.isArray(payload)) {
    const activePack = payload.find((pack) => pack.status === "active") ?? null
    return {
      packs: payload,
      active_pack: activePack
    }
  }
  return {
    packs: Array.isArray(payload?.packs) ? payload.packs : [],
    active_pack: payload?.active_pack ?? null
  }
}

export async function listPersonaVisualPacks(
  personaId: string
): Promise<PersonaVisualPackListResponse> {
  const payload = await fetchPersonaVisualJson<
    PersonaVisualPack[] | PersonaVisualPackListResponse
  >(personaVisualPath(personaId, "/visual-packs"))
  return normalizePersonaVisualPackList(payload)
}

export async function getPersonaVisualPack(
  personaId: string,
  packId: string
): Promise<PersonaVisualPack> {
  return fetchPersonaVisualJson<PersonaVisualPack>(packPath(personaId, packId))
}

export async function createPersonaVisualPack(
  personaId: string,
  payload: PersonaVisualPackCreate
): Promise<PersonaVisualPack> {
  return fetchPersonaVisualJson<PersonaVisualPack>(
    personaVisualPath(personaId, "/visual-packs"),
    {
      method: "POST",
      body: payload
    }
  )
}

export async function updatePersonaVisualManifest(
  personaId: string,
  packId: string,
  payload: PersonaVisualManifestUpdate
): Promise<PersonaVisualPack> {
  return fetchPersonaVisualJson<PersonaVisualPack>(
    packPath(personaId, packId, "/manifest"),
    {
      method: "PATCH",
      body: payload
    }
  )
}

export async function uploadPersonaVisualAsset(
  personaId: string,
  packId: string,
  file: File,
  role: PersonaVisualAssetRole
): Promise<PersonaVisualAsset> {
  const formData = new FormData()
  formData.append("file", file)
  formData.append("asset_role", role)
  return fetchPersonaVisualJson<PersonaVisualAsset>(
    packPath(personaId, packId, "/assets"),
    {
      method: "POST",
      body: formData
    }
  )
}

export async function activatePersonaVisualPack(
  personaId: string,
  packId: string
): Promise<PersonaVisualPack> {
  return fetchPersonaVisualJson<PersonaVisualPack>(
    packPath(personaId, packId, "/activate"),
    {
      method: "POST"
    }
  )
}

export async function deactivatePersonaVisualPack(
  personaId: string
): Promise<PersonaVisualDeactivateResponse> {
  return fetchPersonaVisualJson<PersonaVisualDeactivateResponse>(
    personaVisualPath(personaId, "/visual-packs/deactivate"),
    {
      method: "POST"
    }
  )
}

export async function listPersonaVisualCandidates(
  personaId: string,
  packId: string
): Promise<PersonaVisualCandidateListResponse> {
  const payload = await fetchPersonaVisualJson<
    PersonaVisualCandidate[] | PersonaVisualCandidateListResponse
  >(packPath(personaId, packId, "/generated-candidates"))
  return Array.isArray(payload) ? { candidates: payload } : payload
}

export async function createPersonaVisualGenerationJob(
  personaId: string,
  packId: string,
  payload: PersonaVisualGenerationRequest
): Promise<PersonaVisualGenerationJobResponse> {
  return fetchPersonaVisualJson<PersonaVisualGenerationJobResponse>(
    packPath(personaId, packId, "/generation-jobs"),
    {
      method: "POST",
      body: payload
    }
  )
}

export async function reviewPersonaVisualCandidate(
  personaId: string,
  packId: string,
  candidateId: string,
  payload: PersonaVisualCandidateReviewRequest
): Promise<PersonaVisualCandidate> {
  return fetchPersonaVisualJson<PersonaVisualCandidate>(
    packPath(
      personaId,
      packId,
      `/candidates/${encodeURIComponent(candidateId)}/review`
    ),
    {
      method: "POST",
      body: payload
    }
  )
}
