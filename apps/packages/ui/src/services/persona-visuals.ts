import { tldwClient } from "@/services/tldw/TldwApiClient"
import { downloadBlob } from "@/utils/download-blob"
import type {
  PersonaVisualAsset,
  PersonaVisualAssetRole,
  PersonaVisualCandidate,
  PersonaVisualCandidateListResponse,
  PersonaVisualCandidateReviewRequest,
  PersonaVisualDeactivateResponse,
  PersonaVisualDuplicateTarget,
  PersonaVisualGenerationJobResponse,
  PersonaVisualGenerationRequest,
  PersonaVisualGenerationReadinessResponse,
  PersonaVisualImportCommitRequest,
  PersonaVisualImportCommitStartResponse,
  PersonaVisualImportPreviewResponse,
  PersonaVisualImportPreviewStartResponse,
  PersonaVisualLibraryDeleteResponse,
  PersonaVisualLibraryItem,
  PersonaVisualLibraryListResponse,
  PersonaVisualLibrarySaveRequest,
  PersonaVisualLibraryUpdateRequest,
  PersonaVisualLibraryUseRequest,
  PersonaVisualManifestUpdate,
  PersonaVisualPack,
  PersonaVisualPackCreate,
  PersonaVisualPackDuplicateRequest,
  PersonaVisualPackExportRequest,
  PersonaVisualPackExportResponse,
  PersonaVisualPackListResponse,
  PersonaVisualPortabilityJobResponse,
  PersonaVisualRendererCapabilitiesResponse
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

const visualLibraryPath = (
  suffix = ""
): `/api/v1/persona/visual-library${string}` =>
  `/api/v1/persona/visual-library${suffix}`

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

export async function getPersonaVisualRendererCapabilities(): Promise<
  PersonaVisualRendererCapabilitiesResponse
> {
  const payload = await fetchPersonaVisualJson<PersonaVisualRendererCapabilitiesResponse>(
    "/api/v1/persona/visual-renderers"
  )
  return {
    renderers: Array.isArray(payload?.renderers) ? payload.renderers : []
  }
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

export async function duplicatePersonaVisualPack(
  sourcePersonaId: string,
  packId: string,
  payload: PersonaVisualPackDuplicateRequest
): Promise<PersonaVisualPack> {
  return fetchPersonaVisualJson<PersonaVisualPack>(
    packPath(sourcePersonaId, packId, "/duplicate"),
    {
      method: "POST",
      body: payload
    }
  )
}

export async function listPersonaVisualLibraryItems(): Promise<
  PersonaVisualLibraryListResponse
> {
  const payload = await fetchPersonaVisualJson<
    PersonaVisualLibraryItem[] | PersonaVisualLibraryListResponse
  >(visualLibraryPath())
  return Array.isArray(payload)
    ? { items: payload }
    : { items: Array.isArray(payload?.items) ? payload.items : [] }
}

export async function savePersonaVisualPackToLibrary(
  personaId: string,
  packId: string,
  payload: PersonaVisualLibrarySaveRequest
): Promise<PersonaVisualLibraryItem> {
  return fetchPersonaVisualJson<PersonaVisualLibraryItem>(
    packPath(personaId, packId, "/library"),
    {
      method: "POST",
      body: payload
    }
  )
}

export async function updatePersonaVisualLibraryItem(
  itemId: string,
  payload: PersonaVisualLibraryUpdateRequest
): Promise<PersonaVisualLibraryItem> {
  return fetchPersonaVisualJson<PersonaVisualLibraryItem>(
    visualLibraryPath(`/${encodeURIComponent(itemId)}`),
    {
      method: "PATCH",
      body: payload
    }
  )
}

export async function deletePersonaVisualLibraryItem(
  itemId: string
): Promise<PersonaVisualLibraryDeleteResponse> {
  return fetchPersonaVisualJson<PersonaVisualLibraryDeleteResponse>(
    visualLibraryPath(`/${encodeURIComponent(itemId)}`),
    {
      method: "DELETE"
    }
  )
}

export async function usePersonaVisualLibraryItem(
  itemId: string,
  payload: PersonaVisualLibraryUseRequest
): Promise<PersonaVisualPack> {
  return fetchPersonaVisualJson<PersonaVisualPack>(
    visualLibraryPath(`/${encodeURIComponent(itemId)}/use`),
    {
      method: "POST",
      body: payload
    }
  )
}

export async function listPersonaVisualDuplicateTargets(): Promise<
  PersonaVisualDuplicateTarget[]
> {
  const payload = await fetchPersonaVisualJson<unknown>("/api/v1/persona/catalog")
  if (!Array.isArray(payload)) return []
  return payload
    .map((item): PersonaVisualDuplicateTarget | null => {
      if (!item || typeof item !== "object") return null
      const candidate = item as { id?: unknown; name?: unknown }
      const id = String(candidate.id || "").trim()
      if (!id) return null
      return {
        id,
        name:
          typeof candidate.name === "string" && candidate.name.trim()
            ? candidate.name
            : null
      }
    })
    .filter((item): item is PersonaVisualDuplicateTarget => item !== null)
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

export async function getPersonaVisualGenerationReadiness(
  personaId: string,
  packId: string,
  backend?: string | null
): Promise<PersonaVisualGenerationReadinessResponse> {
  const query = backend?.trim()
    ? `?backend=${encodeURIComponent(backend.trim())}`
    : ""
  return fetchPersonaVisualJson<PersonaVisualGenerationReadinessResponse>(
    packPath(personaId, packId, `/generation-readiness${query}`)
  )
}

export async function startPersonaVisualPackExport(
  personaId: string,
  packId: string,
  payload: PersonaVisualPackExportRequest = {}
): Promise<PersonaVisualPackExportResponse> {
  return fetchPersonaVisualJson<PersonaVisualPackExportResponse>(
    packPath(personaId, packId, "/export"),
    {
      method: "POST",
      body: payload
    }
  )
}

export async function getPersonaVisualPackExportJob(
  personaId: string,
  packId: string,
  jobId: string
): Promise<PersonaVisualPortabilityJobResponse> {
  return fetchPersonaVisualJson<PersonaVisualPortabilityJobResponse>(
    packPath(personaId, packId, `/exports/${encodeURIComponent(jobId)}`)
  )
}

export async function downloadPersonaVisualPackExportArchive(
  personaId: string,
  packId: string,
  jobId: string,
  filename = "persona-visual-pack.tldw-persona-vpack"
): Promise<void> {
  const response = await tldwClient.fetchWithAuth(
    packPath(personaId, packId, `/exports/${encodeURIComponent(jobId)}/download`) as any,
    {
      method: "GET",
      responseType: "arrayBuffer"
    }
  )
  if (!response.ok) {
    throw new PersonaVisualApiError(
      response.error || "Persona visual export download failed",
      {
        status: response.status,
        detail: response.data
      }
    )
  }
  const data = response.data
  let blobPart: BlobPart | null = null
  if (data instanceof ArrayBuffer) {
    blobPart = data
  } else if (ArrayBuffer.isView(data)) {
    if (data.buffer instanceof ArrayBuffer) {
      blobPart = new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
    } else {
      const view = new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
      blobPart = Uint8Array.from(view)
    }
  } else if (Array.isArray(data)) {
    blobPart = Uint8Array.from(data)
  }
  if (!blobPart) {
    throw new PersonaVisualApiError("Persona visual export download was empty", {
      status: response.status,
      detail: data
    })
  }
  downloadBlob(
    new Blob([blobPart], {
      type: "application/vnd.tldw.persona.visual-pack+zip"
    }),
    filename
  )
}

export async function createPersonaVisualImportPreview(
  personaId: string,
  file: File
): Promise<PersonaVisualImportPreviewStartResponse> {
  const formData = new FormData()
  formData.append("archive", file)
  return fetchPersonaVisualJson<PersonaVisualImportPreviewStartResponse>(
    personaVisualPath(personaId, "/visual-packs/import-previews"),
    {
      method: "POST",
      body: formData
    }
  )
}

export async function getPersonaVisualImportPreview(
  personaId: string,
  previewId: string
): Promise<PersonaVisualImportPreviewResponse> {
  return fetchPersonaVisualJson<PersonaVisualImportPreviewResponse>(
    personaVisualPath(
      personaId,
      `/visual-packs/import-previews/${encodeURIComponent(previewId)}`
    )
  )
}

export async function startPersonaVisualImportCommit(
  personaId: string,
  previewId: string,
  payload: PersonaVisualImportCommitRequest = {}
): Promise<PersonaVisualImportCommitStartResponse> {
  return fetchPersonaVisualJson<PersonaVisualImportCommitStartResponse>(
    personaVisualPath(
      personaId,
      `/visual-packs/import-previews/${encodeURIComponent(previewId)}/commit`
    ),
    {
      method: "POST",
      body: payload
    }
  )
}

export async function getPersonaVisualImportCommitStatus(
  personaId: string,
  jobId: string
): Promise<PersonaVisualPortabilityJobResponse> {
  return fetchPersonaVisualJson<PersonaVisualPortabilityJobResponse>(
    personaVisualPath(
      personaId,
      `/visual-packs/imports/${encodeURIComponent(jobId)}`
    )
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
