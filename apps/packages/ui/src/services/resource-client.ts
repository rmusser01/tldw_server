import type { ServicePromptTargetConfig } from "@/services/tldw/TldwApiClient"
import { bgRequest } from "@/services/background-proxy"
import type {
  AllowedMethodFor,
  AllowedPath,
  PathOrUrl,
  UpperLower
} from "@/services/tldw/openapi-guard"
import { appendPathQuery, toAllowedPath } from "@/services/tldw/path-utils"

export type QueryValue = string | number | boolean
export type QueryParams = Record<
  string,
  QueryValue | QueryValue[] | null | undefined
>

export type QueryOptions = {
  arrayFormat?: "repeat" | "comma"
}

export type RequestFn = <
  T = any,
  P extends PathOrUrl = AllowedPath,
  M extends AllowedMethodFor<P> = AllowedMethodFor<P>
>(init: {
  path: P
  method?: UpperLower<M>
  headers?: Record<string, string>
  body?: any
  timeoutMs?: number
  abortSignal?: AbortSignal
  servicePromptConfig?: ServicePromptTargetConfig
}) => Promise<T>

export type RequestOptions = {
  headers?: Record<string, string>
  timeoutMs?: number
  abortSignal?: AbortSignal
  servicePromptConfig?: ServicePromptTargetConfig
  queryOptions?: QueryOptions
}

const defaultJsonHeaders = {
  "Content-Type": "application/json"
}

const shouldIncludeValue = (value: QueryValue): boolean => {
  if (typeof value === "string") {
    return value.trim().length > 0
  }
  return true
}

export const buildQuery = (
  params?: QueryParams,
  options?: QueryOptions
): string => {
  if (!params) return ""
  const search = new URLSearchParams()
  const arrayFormat = options?.arrayFormat ?? "repeat"

  for (const [key, value] of Object.entries(params)) {
    if (value == null) continue
    if (Array.isArray(value)) {
      const filtered = value.filter(
        (item): item is QueryValue => item != null && shouldIncludeValue(item)
      )
      if (filtered.length === 0) continue
      if (arrayFormat === "comma") {
        search.set(key, filtered.map((item) => String(item)).join(","))
      } else {
        filtered.forEach((item) => search.append(key, String(item)))
      }
      continue
    }
    if (!shouldIncludeValue(value)) continue
    search.set(key, String(value))
  }

  const query = search.toString()
  return query ? `?${query}` : ""
}

const withQuery = (
  path: AllowedPath,
  params?: QueryParams,
  options?: QueryOptions
): AllowedPath => appendPathQuery(path, buildQuery(params, options))

const joinDetailPath = (basePath: AllowedPath, id: string | number): AllowedPath => {
  const trimmed = String(basePath).replace(/\/$/, "")
  return toAllowedPath(`${trimmed}/${encodeURIComponent(String(id))}`)
}

type ResourceClientConfig = {
  basePath: AllowedPath
  request?: RequestFn
  detailPath?: (id: string | number) => AllowedPath
  jsonHeaders?: Record<string, string> | null
  updateMethod?: "PATCH" | "PUT"
}

export const createResourceClient = ({
  basePath,
  request = bgRequest,
  detailPath,
  jsonHeaders = defaultJsonHeaders,
  updateMethod = "PATCH"
}: ResourceClientConfig) => {
  const resolveDetailPath =
    detailPath ?? ((id: string | number) => joinDetailPath(basePath, id))

  const mergeHeaders = (headers?: Record<string, string>) => {
    if (!jsonHeaders) return headers
    return {
      ...jsonHeaders,
      ...(headers || {})
    }
  }

  return {
    list: async <T = any>(params?: QueryParams, options?: RequestOptions) => {
      return await request<T>({
        path: withQuery(basePath, params, options?.queryOptions),
        method: "GET",
        headers: options?.headers,
        timeoutMs: options?.timeoutMs,
        abortSignal: options?.abortSignal,
        ...(options?.servicePromptConfig ? { servicePromptConfig: options.servicePromptConfig } : {})
      })
    },
    create: async <T = any>(body?: any, options?: RequestOptions) => {
      return await request<T>({
        path: basePath,
        method: "POST",
        headers: mergeHeaders(options?.headers),
        body,
        timeoutMs: options?.timeoutMs,
        abortSignal: options?.abortSignal,
        ...(options?.servicePromptConfig ? { servicePromptConfig: options.servicePromptConfig } : {})
      })
    },
    get: async <T = any>(
      id: string | number,
      params?: QueryParams,
      options?: RequestOptions
    ) => {
      return await request<T>({
        path: withQuery(resolveDetailPath(id), params, options?.queryOptions),
        method: "GET",
        headers: options?.headers,
        timeoutMs: options?.timeoutMs,
        abortSignal: options?.abortSignal,
        ...(options?.servicePromptConfig ? { servicePromptConfig: options.servicePromptConfig } : {})
      })
    },
    update: async <T = any>(
      id: string | number,
      body?: any,
      options?: RequestOptions
    ) => {
      return await request<T>({
        path: resolveDetailPath(id),
        method: updateMethod,
        headers: mergeHeaders(options?.headers),
        body,
        timeoutMs: options?.timeoutMs,
        abortSignal: options?.abortSignal,
        ...(options?.servicePromptConfig ? { servicePromptConfig: options.servicePromptConfig } : {})
      })
    },
    remove: async <T = any>(
      id: string | number,
      params?: QueryParams,
      options?: RequestOptions
    ) => {
      return await request<T>({
        path: withQuery(resolveDetailPath(id), params, options?.queryOptions),
        method: "DELETE",
        headers: options?.headers,
        timeoutMs: options?.timeoutMs,
        abortSignal: options?.abortSignal,
        ...(options?.servicePromptConfig ? { servicePromptConfig: options.servicePromptConfig } : {})
      })
    }
  }
}
