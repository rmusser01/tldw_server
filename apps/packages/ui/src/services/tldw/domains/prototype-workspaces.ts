import { fetchWithTldwAuth } from "@/services/tldw/auth-fetch"
import { getTldwServerURL } from "@/services/tldw-server"
import type {
  PrototypeCollaboratorSessionCreateInput,
  PrototypeWorkspaceDetail,
  PrototypePromotionCreateInput,
  PrototypePromotionRequest,
  PrototypeSessionJob,
  PrototypeWorkspace,
  PrototypeWorkspaceCreateInput,
  PrototypeWorkspaceSessionCreateInput
} from "@/types/prototype-workspace"
import type {
  PrototypeLinkExchangeRequest,
  PrototypeLinkExchangeResponse
} from "@/types/sharing"

type TldwApiClientCore = object

const apiUrl = async (path: string) => {
  const base = await getTldwServerURL()
  return `${base}/api/v1${path}`
}

const jsonPost = async <T>(path: string, body: unknown): Promise<T> => {
  const url = await apiUrl(path)
  const res = await fetchWithTldwAuth(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || "Request failed")
  }

  return res.json()
}

const jsonGet = async <T>(path: string): Promise<T> => {
  const url = await apiUrl(path)
  const res = await fetchWithTldwAuth(url)

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || "Request failed")
  }

  return res.json()
}

export const createPrototypeWorkspaceRequest = (
  body: PrototypeWorkspaceCreateInput
) => jsonPost<PrototypeWorkspace>("/prototype-workspaces", body)

export const getPrototypeWorkspaceRequest = (prototypeWorkspaceId: string) =>
  jsonGet<PrototypeWorkspaceDetail>(
    `/prototype-workspaces/${encodeURIComponent(prototypeWorkspaceId)}`
  )

export const createPrototypeOwnerBranchSessionRequest = (
  prototypeWorkspaceId: string,
  body: PrototypeWorkspaceSessionCreateInput
) =>
  jsonPost<PrototypeSessionJob>(
    `/prototype-workspaces/${encodeURIComponent(prototypeWorkspaceId)}/sessions`,
    body
  )

export const createPrototypeCollaboratorBranchSessionRequest = (
  body: PrototypeCollaboratorSessionCreateInput
) => jsonPost<PrototypeSessionJob>("/prototype-sessions", body)

export const createPrototypePromotionRequestRequest = (
  body: PrototypePromotionCreateInput
) => jsonPost<PrototypePromotionRequest>("/prototype-promotions", body)

export const exchangePrototypePrivateLinkRequest = (
  token: string,
  body: PrototypeLinkExchangeRequest
) =>
  jsonPost<PrototypeLinkExchangeResponse>(
    `/sharing/public/${encodeURIComponent(token)}/prototype-session`,
    body
  )

export const prototypeWorkspaceMethods = {
  async createPrototypeWorkspace(
    this: TldwApiClientCore,
    body: PrototypeWorkspaceCreateInput
  ): Promise<PrototypeWorkspace> {
    return createPrototypeWorkspaceRequest(body)
  },

  async getPrototypeWorkspace(
    this: TldwApiClientCore,
    prototypeWorkspaceId: string
  ): Promise<PrototypeWorkspaceDetail> {
    return getPrototypeWorkspaceRequest(prototypeWorkspaceId)
  },

  async createPrototypeOwnerBranchSession(
    this: TldwApiClientCore,
    prototypeWorkspaceId: string,
    body: PrototypeWorkspaceSessionCreateInput = {}
  ): Promise<PrototypeSessionJob> {
    return createPrototypeOwnerBranchSessionRequest(prototypeWorkspaceId, body)
  },

  async createPrototypeCollaboratorBranchSession(
    this: TldwApiClientCore,
    body: PrototypeCollaboratorSessionCreateInput
  ): Promise<PrototypeSessionJob> {
    return createPrototypeCollaboratorBranchSessionRequest(body)
  },

  async createPrototypePromotionRequest(
    this: TldwApiClientCore,
    body: PrototypePromotionCreateInput
  ): Promise<PrototypePromotionRequest> {
    return createPrototypePromotionRequestRequest(body)
  },

  async exchangePrototypePrivateLink(
    this: TldwApiClientCore,
    token: string,
    body: PrototypeLinkExchangeRequest
  ): Promise<PrototypeLinkExchangeResponse> {
    return exchangePrototypePrivateLinkRequest(token, body)
  }
}

export type PrototypeWorkspaceMethods = typeof prototypeWorkspaceMethods
