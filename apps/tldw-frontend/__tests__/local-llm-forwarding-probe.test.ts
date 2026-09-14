import http from "node:http"

import { describe, expect, it, vi } from "vitest"

import {
  createLocalLlmForwardingProbe,
  createProviderProbeState,
  forwardProviderRequest,
  providerContextProof,
  validateProbeListenUrl,
} from "../scripts/local-llm-forwarding-probe.mjs"

const OWNER_SENTINEL = "OWNER-UNRELATED-SENTINEL-7F3C9D"
const RECIPIENT_SENTINEL = "RECIPIENT-LOCAL-SENTINEL-4A8E2B"

describe("local LLM forwarding probe", () => {
  it("forwards the exact request bytes and persists only bounded hashes and booleans", async () => {
    const body = Buffer.from(
      JSON.stringify({
        messages: [{ content: "Use only shared evidence.", role: "user" }],
        model: "Qwen2.5-0.5B-Instruct",
      })
    )
    const state = createProviderProbeState({
      ownerSentinel: OWNER_SENTINEL,
      recipientSentinel: RECIPIENT_SENTINEL,
    })
    const fetchImpl = vi.fn(async (_url: string, init: RequestInit) => {
      expect(Buffer.from(init.body as Uint8Array)).toEqual(body)
      expect(init.redirect).toBe("error")
      return new Response('{"choices":[]}', {
        headers: { "content-type": "application/json" },
        status: 200,
      })
    })

    const response = await forwardProviderRequest({
      body,
      fetchImpl,
      headers: { "content-type": "application/json" },
      state,
      targetUrl: "http://127.0.0.1:9099/v1/chat/completions",
    })
    const proof = providerContextProof(state)

    expect(response.status).toBe(200)
    expect(fetchImpl).toHaveBeenCalledTimes(1)
    expect(proof).toEqual({
      bodyUnchanged: true,
      forwardedRequestCount: 1,
      inputBodyHashes: [expect.stringMatching(/^sha256:[a-f0-9]{64}$/)],
      maximumRequestCount: 16,
      mutationPayloadsAbsent: true,
      outputBodyHashes: [expect.stringMatching(/^sha256:[a-f0-9]{64}$/)],
      ownerSentinelAbsent: true,
      payloadJsonValid: true,
      recipientSentinelAbsent: true,
      toolPayloadsAbsent: true,
      withinRequestBound: true,
    })
    expect(JSON.stringify(proof)).not.toContain("Use only shared evidence")
    expect(JSON.stringify(proof)).not.toContain("messages")
  })

  it.each([301, 302, 303, 307, 308])(
    "fails closed on upstream redirect %i without contacting its non-loopback target",
    async (status) => {
      const requestBodyMarker = `redirect-body-${status}`
      const redirectTargetMarker = `external-target-${status}`
      let initialUpstreamRequests = 0
      let externalTargetRequests = 0
      const externalTarget = http.createServer((request, response) => {
        externalTargetRequests += 1
        request.resume()
        response.writeHead(200, { "content-type": "application/json" })
        response.end('{"choices":[]}')
      })
      const externalTargetPort = await new Promise<number>((resolve, reject) => {
        externalTarget.once("error", reject)
        externalTarget.listen(0, "127.0.0.1", () => {
          externalTarget.off("error", reject)
          const address = externalTarget.address()
          if (!address || typeof address === "string") {
            reject(new Error("external target did not expose a TCP address"))
            return
          }
          resolve(address.port)
        })
      })
      const redirector = http.createServer((_request, response) => {
        initialUpstreamRequests += 1
        response.writeHead(status, {
          location: `http://0.0.0.0:${externalTargetPort}/${redirectTargetMarker}`,
        })
        response.end()
      })
      const redirectorPort = await new Promise<number>((resolve, reject) => {
        redirector.once("error", reject)
        redirector.listen(0, "127.0.0.1", () => {
          redirector.off("error", reject)
          const address = redirector.address()
          if (!address || typeof address === "string") {
            reject(new Error("redirector did not expose a TCP address"))
            return
          }
          resolve(address.port)
        })
      })
      const probe = createLocalLlmForwardingProbe({
        ownerSentinel: OWNER_SENTINEL,
        recipientSentinel: RECIPIENT_SENTINEL,
        targetUrl: `http://127.0.0.1:${redirectorPort}/v1/chat/completions`,
      })
      let responseStatus = 0
      let responseText = ""
      try {
        const probeAddress = (await probe.listen({ host: "127.0.0.1", port: 0 })) as {
          port: number
        }
        const response = await fetch(
          `http://127.0.0.1:${probeAddress.port}/v1/chat/completions`,
          {
            body: JSON.stringify({ messages: [{ content: requestBodyMarker, role: "user" }] }),
            headers: { "content-type": "application/json" },
            method: "POST",
          }
        )
        responseStatus = response.status
        responseText = await response.text()
      } finally {
        await probe.close()
        await new Promise<void>((resolve, reject) =>
          redirector.close((error) => (error ? reject(error) : resolve()))
        )
        await new Promise<void>((resolve, reject) =>
          externalTarget.close((error) => (error ? reject(error) : resolve()))
        )
      }

      expect(initialUpstreamRequests).toBe(1)
      expect(externalTargetRequests).toBe(0)
      expect(responseStatus).toBe(502)
      expect(responseText).toBe('{"detail":"Provider forwarding failed"}')
      expect(responseText.length).toBeLessThan(64)
      expect(responseText).not.toContain(requestBodyMarker)
      expect(responseText).not.toContain(redirectTargetMarker)
    }
  )

  it.each([
    ["owner sentinel", { messages: [{ content: OWNER_SENTINEL, role: "user" }] }],
    ["recipient sentinel", { messages: [{ content: RECIPIENT_SENTINEL, role: "user" }] }],
    [
      "tool declaration",
      { messages: [], tools: [{ function: { name: "lookup" }, type: "function" }] },
    ],
    ["tool message", { messages: [{ content: "result", role: "tool" }] }],
    ["mutation payload", { messages: [{ content: "DELETE /api/v1/workspaces/42", role: "user" }] }],
  ])(
    "rejects %s before forwarding and retains no raw request data",
    async (_label, payload) => {
      const state = createProviderProbeState({
        ownerSentinel: OWNER_SENTINEL,
        recipientSentinel: RECIPIENT_SENTINEL,
      })
      const fetchImpl = vi.fn(async () => new Response("ok"))
      await expect(
        forwardProviderRequest({
          body: Buffer.from(JSON.stringify(payload)),
          fetchImpl,
          headers: { "content-type": "application/json" },
          state,
          targetUrl: "http://127.0.0.1:9099/v1/chat/completions",
        })
      ).rejects.toThrow(/provider probe/i)

      const proof = providerContextProof(state)
      expect(fetchImpl).not.toHaveBeenCalled()
      expect(
        proof.ownerSentinelAbsent &&
          proof.recipientSentinelAbsent &&
          proof.toolPayloadsAbsent &&
          proof.mutationPayloadsAbsent
      ).toBe(false)
      expect(JSON.stringify(proof)).not.toContain(OWNER_SENTINEL)
      expect(JSON.stringify(proof)).not.toContain(RECIPIENT_SENTINEL)
      expect(JSON.stringify(proof)).not.toContain("DELETE")
    }
  )

  it("rejects invalid JSON before forwarding", async () => {
    const state = createProviderProbeState({
      ownerSentinel: OWNER_SENTINEL,
      recipientSentinel: RECIPIENT_SENTINEL,
    })
    const fetchImpl = vi.fn(async () => new Response("ok"))

    await expect(
      forwardProviderRequest({
        body: Buffer.from("{not-json"),
        fetchImpl,
        headers: { "content-type": "application/json" },
        state,
        targetUrl: "http://127.0.0.1:9099/v1/chat/completions",
      })
    ).rejects.toThrow(/provider probe/i)
    expect(fetchImpl).not.toHaveBeenCalled()
    expect(providerContextProof(state).payloadJsonValid).toBe(false)
  })

  it("rejects an oversized body before forwarding", async () => {
    const state = createProviderProbeState({
      ownerSentinel: OWNER_SENTINEL,
      recipientSentinel: RECIPIENT_SENTINEL,
    })
    const fetchImpl = vi.fn(async () => new Response("ok"))

    await expect(
      forwardProviderRequest({
        body: Buffer.from(JSON.stringify({ messages: [{ content: "x".repeat(128) }] })),
        fetchImpl,
        headers: { "content-type": "application/json" },
        maxBodyBytes: 32,
        state,
        targetUrl: "http://127.0.0.1:9099/v1/chat/completions",
      })
    ).rejects.toThrow(/body limit/i)
    expect(fetchImpl).not.toHaveBeenCalled()
  })

  it("rejects requests over the configured count before forwarding", async () => {
    const state = createProviderProbeState({
      maxRequests: 1,
      ownerSentinel: OWNER_SENTINEL,
      recipientSentinel: RECIPIENT_SENTINEL,
    })
    const fetchImpl = vi.fn(async () => new Response("ok"))
    const request = {
      body: Buffer.from('{"messages":[]}'),
      fetchImpl,
      headers: { "content-type": "application/json" },
      state,
      targetUrl: "http://127.0.0.1:9099/v1/chat/completions",
    }

    await forwardProviderRequest(request)
    await expect(forwardProviderRequest(request)).rejects.toThrow(/request limit/i)
    expect(fetchImpl).toHaveBeenCalledTimes(1)
    expect(providerContextProof(state).withinRequestBound).toBe(false)
  })

  it("rejects a forwarding-byte mismatch before invoking upstream fetch", async () => {
    const state = createProviderProbeState({
      ownerSentinel: OWNER_SENTINEL,
      recipientSentinel: RECIPIENT_SENTINEL,
    })
    const fetchImpl = vi.fn(async () => new Response("ok"))

    await expect(
      forwardProviderRequest({
        body: Buffer.from('{"messages":[]}'),
        fetchImpl,
        headers: { "content-type": "application/json" },
        prepareForwardBody: () => Buffer.from('{"messages":["changed"]}'),
        state,
        targetUrl: "http://127.0.0.1:9099/v1/chat/completions",
      })
    ).rejects.toThrow(/byte identity/i)
    expect(fetchImpl).not.toHaveBeenCalled()
    expect(providerContextProof(state).bodyUnchanged).toBe(false)
  })

  it.each([
    "not-a-url",
    "https://127.0.0.1:9099/v1/chat/completions",
    "http://192.0.2.10:9099/v1/chat/completions",
    "http://user:password@127.0.0.1:9099/v1/chat/completions",
    "http://127.0.0.1:9099/other",
    "http://127.0.0.1:9099/v1/chat/completions?secret=value",
    "http://127.0.0.1:9099/v1/chat/completions#fragment",
  ])("rejects unsafe upstream target %s before fetch", async (targetUrl) => {
    const state = createProviderProbeState({
      ownerSentinel: OWNER_SENTINEL,
      recipientSentinel: RECIPIENT_SENTINEL,
    })
    const fetchImpl = vi.fn(async () => new Response("ok"))

    await expect(
      forwardProviderRequest({
        body: Buffer.from('{"messages":[]}'),
        fetchImpl,
        headers: { "content-type": "application/json" },
        state,
        targetUrl,
      })
    ).rejects.toThrow(/provider probe rejected/i)
    expect(fetchImpl).not.toHaveBeenCalled()
  })

  it.each(["0.0.0.0", "localhost", "192.0.2.10"])(
    "rejects non-exact loopback bind host %s",
    async (host) => {
      const probe = createLocalLlmForwardingProbe({
        ownerSentinel: OWNER_SENTINEL,
        recipientSentinel: RECIPIENT_SENTINEL,
        targetUrl: "http://127.0.0.1:9099/v1/chat/completions",
      })
      let unexpectedlyListening = false
      try {
        await probe.listen({ host, port: 0 })
        unexpectedlyListening = true
      } catch {
        // Expected: validation rejects before server.listen.
      } finally {
        if (unexpectedlyListening) await probe.close()
      }
      expect(unexpectedlyListening).toBe(false)
    }
  )

  it.each([
    "https://127.0.0.1:19099",
    "http://0.0.0.0:19099",
    "http://localhost:19099",
    "http://user:password@127.0.0.1:19099",
    "http://127.0.0.1:19099/other",
    "http://127.0.0.1:19099?query=value",
    "http://127.0.0.1:19099#fragment",
  ])("rejects unsafe main listen URL %s", (listenUrl) => {
    expect(() => validateProbeListenUrl(listenUrl)).toThrow(/listen|loopback|http/i)
  })

  it.each(["http://127.0.0.1:19099", "http://[::1]:19099"])(
    "accepts exact loopback main listen URL %s",
    (listenUrl) => {
      expect(validateProbeListenUrl(listenUrl)).toMatchObject({ port: 19099 })
    }
  )
})
