#!/usr/bin/env node

import crypto from "node:crypto"
import http from "node:http"
import { pathToFileURL } from "node:url"

const DEFAULT_MAX_REQUESTS = 16
const DEFAULT_MAX_BODY_BYTES = 10 * 1024 * 1024
const CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

const sha256Bytes = (value) => `sha256:${crypto.createHash("sha256").update(value).digest("hex")}`

const containsToolPayload = (value) => {
  if (!value || typeof value !== "object") return false
  if (Array.isArray(value)) return value.some(containsToolPayload)
  for (const [key, item] of Object.entries(value)) {
    if (
      [
        "function_call",
        "functions",
        "parallel_tool_calls",
        "tool_call_id",
        "tool_calls",
        "tool_choice",
        "tools",
      ].includes(key)
    ) {
      return true
    }
    if (key === "role" && ["function", "tool"].includes(String(item))) return true
    if (containsToolPayload(item)) return true
  }
  return false
}

const containsMutationPayload = (value) =>
  /\b(?:DELETE|PATCH|POST|PUT)\s+\/api\//i.test(String(value))

class ProviderProbeValidationError extends Error {
  constructor(message, statusCode = 400) {
    super(`Provider probe rejected: ${message}`)
    this.name = "ProviderProbeValidationError"
    this.statusCode = statusCode
  }
}

const normalizeLoopbackHost = (host) => String(host || "").toLowerCase().replace(/^\[|\]$/g, "")

const requireExactLoopbackHost = (host, label) => {
  const normalized = normalizeLoopbackHost(host)
  if (!["127.0.0.1", "::1"].includes(normalized)) {
    throw new ProviderProbeValidationError(`${label} must use an exact loopback host`)
  }
  return normalized
}

const parseHttpUrl = (value, label) => {
  let parsed
  try {
    parsed = new URL(String(value || ""))
  } catch {
    throw new ProviderProbeValidationError(`${label} must be a valid URL`)
  }
  if (parsed.protocol !== "http:") {
    throw new ProviderProbeValidationError(`${label} must use local HTTP`)
  }
  requireExactLoopbackHost(parsed.hostname, label)
  if (parsed.username || parsed.password) {
    throw new ProviderProbeValidationError(`${label} must not contain credentials`)
  }
  if (parsed.search) throw new ProviderProbeValidationError(`${label} must not contain a query`)
  if (parsed.hash) throw new ProviderProbeValidationError(`${label} must not contain a fragment`)
  return parsed
}

export const validateProviderTargetUrl = (value) => {
  const parsed = parseHttpUrl(value, "upstream target")
  if (parsed.pathname !== CHAT_COMPLETIONS_PATH) {
    throw new ProviderProbeValidationError(
      `upstream target path must be ${CHAT_COMPLETIONS_PATH}`
    )
  }
  return parsed.toString()
}

export const validateProbeListenUrl = (value) => {
  const parsed = parseHttpUrl(value, "listen URL")
  if (parsed.pathname !== "/") {
    throw new ProviderProbeValidationError("listen URL path must be root")
  }
  const port = Number(parsed.port)
  if (!Number.isInteger(port) || port < 1 || port > 65_535) {
    throw new ProviderProbeValidationError("listen URL must include a valid port")
  }
  return { host: requireExactLoopbackHost(parsed.hostname, "listen URL"), port }
}

export const createProviderProbeState = ({
  maxRequests = DEFAULT_MAX_REQUESTS,
  ownerSentinel,
  recipientSentinel,
}) => ({
  bodyUnchanged: true,
  forwardedRequestCount: 0,
  inputBodyHashes: [],
  maxRequests,
  mutationPayloadsAbsent: true,
  outputBodyHashes: [],
  ownerSentinel,
  ownerSentinelAbsent: true,
  payloadJsonValid: true,
  recipientSentinel,
  recipientSentinelAbsent: true,
  toolPayloadsAbsent: true,
  withinRequestBound: true,
})

export const providerContextProof = (state) => ({
  bodyUnchanged: state.bodyUnchanged,
  forwardedRequestCount: state.forwardedRequestCount,
  inputBodyHashes: [...state.inputBodyHashes],
  maximumRequestCount: state.maxRequests,
  mutationPayloadsAbsent: state.mutationPayloadsAbsent,
  outputBodyHashes: [...state.outputBodyHashes],
  ownerSentinelAbsent: state.ownerSentinelAbsent,
  payloadJsonValid: state.payloadJsonValid,
  recipientSentinelAbsent: state.recipientSentinelAbsent,
  toolPayloadsAbsent: state.toolPayloadsAbsent,
  withinRequestBound: state.withinRequestBound,
})

export const forwardProviderRequest = async ({
  body,
  fetchImpl = fetch,
  headers,
  maxBodyBytes = DEFAULT_MAX_BODY_BYTES,
  prepareForwardBody = (value) => Buffer.from(value),
  state,
  targetUrl,
}) => {
  const validatedTargetUrl = validateProviderTargetUrl(targetUrl)
  const inputBody = Buffer.from(body)
  if (!Number.isInteger(maxBodyBytes) || maxBodyBytes < 1 || inputBody.length > maxBodyBytes) {
    throw new ProviderProbeValidationError("request exceeded the body limit", 413)
  }
  if (state.forwardedRequestCount >= state.maxRequests) {
    state.withinRequestBound = false
    throw new ProviderProbeValidationError("request limit exceeded", 429)
  }
  const inputHash = sha256Bytes(inputBody)
  const text = inputBody.toString("utf8")
  state.ownerSentinelAbsent &&= !text.includes(state.ownerSentinel)
  state.recipientSentinelAbsent &&= !text.includes(state.recipientSentinel)
  state.mutationPayloadsAbsent &&= !containsMutationPayload(text)
  let payload
  try {
    payload = JSON.parse(text)
  } catch {
    state.payloadJsonValid = false
  }
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    state.payloadJsonValid = false
  }
  state.toolPayloadsAbsent &&= !containsToolPayload(payload)

  const forwardedBody = Buffer.from(prepareForwardBody(inputBody))
  const outputHash = sha256Bytes(forwardedBody)
  state.bodyUnchanged &&= inputBody.equals(forwardedBody) && inputHash === outputHash
  const violations = []
  if (!state.payloadJsonValid) violations.push("request body is not valid JSON")
  if (!state.ownerSentinelAbsent) violations.push("owner sentinel is present")
  if (!state.recipientSentinelAbsent) violations.push("recipient sentinel is present")
  if (!state.toolPayloadsAbsent) violations.push("tool or function-call payload is present")
  if (!state.mutationPayloadsAbsent) violations.push("mutation payload is present")
  if (!state.bodyUnchanged) violations.push("forwarding byte identity changed")
  if (violations.length) {
    throw new ProviderProbeValidationError(violations.join("; "))
  }

  state.forwardedRequestCount += 1
  state.inputBodyHashes.push(inputHash)
  state.outputBodyHashes.push(outputHash)

  const forwardedHeaders = new Headers(headers)
  forwardedHeaders.delete("content-length")
  forwardedHeaders.delete("host")
  return fetchImpl(validatedTargetUrl, {
    body: forwardedBody,
    headers: forwardedHeaders,
    method: "POST",
  })
}

const readBody = (request, maxBodyBytes) =>
  new Promise((resolve, reject) => {
    const chunks = []
    let size = 0
    let exceeded = false
    request.on("data", (chunk) => {
      if (exceeded) return
      size += chunk.length
      if (size > maxBodyBytes) {
        exceeded = true
        chunks.length = 0
        reject(new ProviderProbeValidationError("request exceeded the body limit", 413))
        return
      }
      chunks.push(chunk)
    })
    request.on("end", () => {
      if (!exceeded) resolve(Buffer.concat(chunks))
    })
    request.on("error", reject)
  })

export const createLocalLlmForwardingProbe = ({
  fetchImpl = fetch,
  maxBodyBytes = DEFAULT_MAX_BODY_BYTES,
  maxRequests = DEFAULT_MAX_REQUESTS,
  ownerSentinel,
  recipientSentinel,
  targetUrl,
}) => {
  const validatedTargetUrl = validateProviderTargetUrl(targetUrl)
  let state = createProviderProbeState({
    maxRequests,
    ownerSentinel,
    recipientSentinel,
  })
  const server = http.createServer(async (request, response) => {
    try {
      const requestUrl = new URL(request.url || "/", "http://127.0.0.1")
      if (request.method === "GET" && requestUrl.pathname === "/health") {
        response.writeHead(200, { "content-type": "application/json" })
        response.end('{"status":"ok"}')
        return
      }
      if (request.method === "GET" && requestUrl.pathname === "/__tldw_provider_probe/proof") {
        response.writeHead(200, { "content-type": "application/json" })
        response.end(JSON.stringify(providerContextProof(state)))
        return
      }
      if (request.method === "POST" && requestUrl.pathname === "/__tldw_provider_probe/reset") {
        state = createProviderProbeState({
          maxRequests,
          ownerSentinel,
          recipientSentinel,
        })
        response.writeHead(204)
        response.end()
        return
      }
      if (request.method !== "POST" || requestUrl.pathname !== "/v1/chat/completions") {
        response.writeHead(404, { "content-type": "application/json" })
        response.end('{"detail":"Not found"}')
        return
      }

      const body = await readBody(request, maxBodyBytes)
      const upstream = await forwardProviderRequest({
        body,
        fetchImpl,
        headers: request.headers,
        maxBodyBytes,
        state,
        targetUrl: validatedTargetUrl,
      })
      const responseBody = Buffer.from(await upstream.arrayBuffer())
      const responseHeaders = {}
      for (const [key, value] of upstream.headers.entries()) {
        if (!["content-encoding", "content-length", "transfer-encoding"].includes(key)) {
          responseHeaders[key] = value
        }
      }
      response.writeHead(upstream.status, responseHeaders)
      response.end(responseBody)
    } catch (error) {
      const statusCode =
        error instanceof ProviderProbeValidationError ? error.statusCode : 502
      response.writeHead(statusCode, { "content-type": "application/json" })
      response.end(
        statusCode === 502
          ? '{"detail":"Provider forwarding failed"}'
          : '{"detail":"Provider request rejected"}'
      )
    }
  })
  return {
    close: () =>
      new Promise((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve()))
      ),
    listen: ({ host, port }) =>
      new Promise((resolve, reject) => {
        let validatedHost
        try {
          validatedHost = requireExactLoopbackHost(host, "listen host")
        } catch (error) {
          reject(error)
          return
        }
        server.once("error", reject)
        server.listen(port, validatedHost, () => {
          server.off("error", reject)
          resolve(server.address())
        })
      }),
    proof: () => providerContextProof(state),
  }
}

const main = async ({ env = process.env } = {}) => {
  const listenUrl = String(
    env.TLDW_PROVIDER_PROBE_LISTEN_URL || "http://127.0.0.1:19099"
  )
  const listen = validateProbeListenUrl(listenUrl)
  const targetUrl = String(env.TLDW_PROVIDER_PROBE_TARGET_URL || "").trim()
  if (!targetUrl) throw new Error("TLDW_PROVIDER_PROBE_TARGET_URL is required")
  const probe = createLocalLlmForwardingProbe({
    ownerSentinel: env.TLDW_PROVIDER_PROBE_OWNER_SENTINEL || "OWNER-UNRELATED-SENTINEL-7F3C9D",
    recipientSentinel:
      env.TLDW_PROVIDER_PROBE_RECIPIENT_SENTINEL || "RECIPIENT-LOCAL-SENTINEL-4A8E2B",
    targetUrl,
  })
  await probe.listen(listen)
  console.log(`[provider-probe] listening=http://${listen.host}:${listen.port}`)
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) {
  await main()
}
