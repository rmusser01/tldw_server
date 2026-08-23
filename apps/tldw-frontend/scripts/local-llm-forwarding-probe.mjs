#!/usr/bin/env node

import crypto from "node:crypto"
import http from "node:http"
import { pathToFileURL } from "node:url"

const DEFAULT_MAX_REQUESTS = 16
const DEFAULT_MAX_BODY_BYTES = 10 * 1024 * 1024

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
  state,
  targetUrl,
}) => {
  const inputBody = Buffer.from(body)
  const inputHash = sha256Bytes(inputBody)
  state.forwardedRequestCount += 1
  if (state.forwardedRequestCount > state.maxRequests) {
    state.withinRequestBound = false
  } else {
    state.inputBodyHashes.push(inputHash)
  }

  const text = inputBody.toString("utf8")
  state.ownerSentinelAbsent &&= !text.includes(state.ownerSentinel)
  state.recipientSentinelAbsent &&= !text.includes(state.recipientSentinel)
  state.mutationPayloadsAbsent &&= !containsMutationPayload(text)
  try {
    const payload = JSON.parse(text)
    state.toolPayloadsAbsent &&= !containsToolPayload(payload)
  } catch {
    state.payloadJsonValid = false
  }

  const forwardedBody = inputBody
  const outputHash = sha256Bytes(forwardedBody)
  state.bodyUnchanged &&= inputHash === outputHash
  if (state.forwardedRequestCount <= state.maxRequests) {
    state.outputBodyHashes.push(outputHash)
  }

  const forwardedHeaders = new Headers(headers)
  forwardedHeaders.delete("content-length")
  forwardedHeaders.delete("host")
  return fetchImpl(targetUrl, {
    body: forwardedBody,
    headers: forwardedHeaders,
    method: "POST",
  })
}

const readBody = (request, maxBodyBytes) =>
  new Promise((resolve, reject) => {
    const chunks = []
    let size = 0
    request.on("data", (chunk) => {
      size += chunk.length
      if (size > maxBodyBytes) {
        reject(new Error("Provider probe request exceeded the body limit"))
        request.destroy()
        return
      }
      chunks.push(chunk)
    })
    request.on("end", () => resolve(Buffer.concat(chunks)))
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
        state,
        targetUrl,
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
    } catch {
      response.writeHead(502, { "content-type": "application/json" })
      response.end('{"detail":"Provider forwarding failed"}')
    }
  })
  return {
    close: () =>
      new Promise((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve()))
      ),
    listen: ({ host, port }) =>
      new Promise((resolve, reject) => {
        server.once("error", reject)
        server.listen(port, host, () => {
          server.off("error", reject)
          resolve(server.address())
        })
      }),
    proof: () => providerContextProof(state),
  }
}

const main = async ({ env = process.env } = {}) => {
  const listenUrl = new URL(env.TLDW_PROVIDER_PROBE_LISTEN_URL || "http://127.0.0.1:19099")
  const targetUrl = String(env.TLDW_PROVIDER_PROBE_TARGET_URL || "").trim()
  if (!targetUrl) throw new Error("TLDW_PROVIDER_PROBE_TARGET_URL is required")
  const probe = createLocalLlmForwardingProbe({
    ownerSentinel: env.TLDW_PROVIDER_PROBE_OWNER_SENTINEL || "OWNER-UNRELATED-SENTINEL-7F3C9D",
    recipientSentinel:
      env.TLDW_PROVIDER_PROBE_RECIPIENT_SENTINEL || "RECIPIENT-LOCAL-SENTINEL-4A8E2B",
    targetUrl,
  })
  await probe.listen({ host: listenUrl.hostname, port: Number(listenUrl.port) })
  console.log(`[provider-probe] listening=${listenUrl.origin}`)
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) {
  await main()
}
