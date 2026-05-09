/**
 * React Query hooks for the sharing API.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { getTldwServerURL } from "@/services/tldw-server"
import { fetchWithTldwAuth } from "@/services/tldw/auth-fetch"
import type {
  ShareWorkspaceRequest,
  ShareResponse,
  ShareListResponse,
  SharedWithMeResponse,
  CreateTokenRequest,
  PrototypeLinkExchangeResponse,
  TokenResponse,
  TokenListResponse,
  PublicSharePreview,
} from "@/types/sharing"

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

const sharingUrl = async (path: string) => {
  const base = await getTldwServerURL()
  return `${base}/api/v1/sharing${path}`
}

const jsonPost = async <T>(path: string, body: unknown): Promise<T> => {
  const url = await sharingUrl(path)
  const res = await fetchWithTldwAuth(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || "Request failed")
  }
  return res.json()
}

const jsonPatch = async <T>(path: string, body: unknown): Promise<T> => {
  const url = await sharingUrl(path)
  const res = await fetchWithTldwAuth(url, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || "Request failed")
  }
  return res.json()
}

const jsonGet = async <T>(path: string): Promise<T> => {
  const url = await sharingUrl(path)
  const res = await fetchWithTldwAuth(url)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || "Request failed")
  }
  return res.json()
}

const jsonDelete = async (path: string): Promise<void> => {
  const url = await sharingUrl(path)
  const res = await fetchWithTldwAuth(url, { method: "DELETE" })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || "Request failed")
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Query Keys
// ─────────────────────────────────────────────────────────────────────────────

export const sharingKeys = {
  all: ["sharing"] as const,
  workspaceShares: (wsId: string) => ["sharing", "workspace", wsId] as const,
  sharedWithMe: () => ["sharing", "shared-with-me"] as const,
  tokens: () => ["sharing", "tokens"] as const,
  publicPreview: (token: string) => ["sharing", "public", token] as const,
}

// ─────────────────────────────────────────────────────────────────────────────
// Workspace Sharing Hooks
// ─────────────────────────────────────────────────────────────────────────────

export function useWorkspaceShares(workspaceId: string, enabled = true) {
  return useQuery<ShareListResponse>({
    queryKey: sharingKeys.workspaceShares(workspaceId),
    queryFn: () => jsonGet(`/workspaces/${workspaceId}/shares`),
    enabled: !!workspaceId && enabled,
  })
}

export function useShareWorkspace(workspaceId: string) {
  const qc = useQueryClient()
  return useMutation<ShareResponse, Error, ShareWorkspaceRequest>({
    mutationFn: (body) => jsonPost(`/workspaces/${workspaceId}/share`, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: sharingKeys.workspaceShares(workspaceId) })
    },
  })
}

export function useUpdateShare() {
  const qc = useQueryClient()
  return useMutation<
    ShareResponse,
    Error,
    { shareId: number; access_level?: string; allow_clone?: boolean }
  >({
    mutationFn: ({ shareId, ...body }) => jsonPatch(`/shares/${shareId}`, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: sharingKeys.all })
    },
  })
}

export function useRevokeShare() {
  const qc = useQueryClient()
  return useMutation<void, Error, number>({
    mutationFn: (shareId) => jsonDelete(`/shares/${shareId}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: sharingKeys.all })
    },
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared With Me Hooks
// ─────────────────────────────────────────────────────────────────────────────

export function useSharedWithMe() {
  return useQuery<SharedWithMeResponse>({
    queryKey: sharingKeys.sharedWithMe(),
    queryFn: () => jsonGet("/shared-with-me"),
  })
}

export function useCloneWorkspace() {
  const qc = useQueryClient()
  return useMutation<
    { job_id: string; status: string; message: string },
    Error,
    { shareId: number; new_name?: string }
  >({
    mutationFn: ({ shareId, ...body }) =>
      jsonPost(`/shared-with-me/${shareId}/clone`, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: sharingKeys.sharedWithMe() })
    },
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Token Hooks
// ─────────────────────────────────────────────────────────────────────────────

export function useShareTokens() {
  return useQuery<TokenListResponse>({
    queryKey: sharingKeys.tokens(),
    queryFn: () => jsonGet("/tokens"),
  })
}

export function useCreateToken() {
  const qc = useQueryClient()
  return useMutation<TokenResponse, Error, CreateTokenRequest>({
    mutationFn: (body) => jsonPost("/tokens", body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: sharingKeys.tokens() })
    },
  })
}

export function useRevokeToken() {
  const qc = useQueryClient()
  return useMutation<void, Error, number>({
    mutationFn: (tokenId) => jsonDelete(`/tokens/${tokenId}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: sharingKeys.tokens() })
    },
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Public Access Hooks
// ─────────────────────────────────────────────────────────────────────────────

export function usePublicPreview(token: string, enabled = true) {
  return useQuery<PublicSharePreview>({
    queryKey: sharingKeys.publicPreview(token),
    queryFn: () => jsonGet(`/public/${token}`),
    enabled: !!token && enabled,
    retry: false,
  })
}

export function useVerifySharePassword() {
  return useMutation<
    { verified: boolean },
    Error,
    { token: string; password: string }
  >({
    mutationFn: ({ token, password }) =>
      jsonPost(`/public/${token}/verify`, { password }),
  })
}

export function useImportFromToken() {
  return useMutation<
    { resource_type: string; resource_id: string; message: string },
    Error,
    string
  >({
    mutationFn: (token) => jsonPost(`/public/${token}/import`, {}),
  })
}

export function usePrototypePrivateLinkExchange() {
  return useMutation<
    PrototypeLinkExchangeResponse,
    Error,
    { token: string; display_name?: string; password?: string }
  >({
    mutationFn: ({ token, ...body }) =>
      jsonPost(`/public/${token}/prototype-session`, body)
  })
}
