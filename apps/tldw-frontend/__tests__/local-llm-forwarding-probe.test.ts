import { describe, expect, it, vi } from "vitest"

import {
  createProviderProbeState,
  forwardProviderRequest,
  providerContextProof,
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
    "records a failed proof for %s without retaining raw request data",
    async (_label, payload) => {
      const state = createProviderProbeState({
        ownerSentinel: OWNER_SENTINEL,
        recipientSentinel: RECIPIENT_SENTINEL,
      })
      await forwardProviderRequest({
        body: Buffer.from(JSON.stringify(payload)),
        fetchImpl: vi.fn(async () => new Response("ok")),
        headers: { "content-type": "application/json" },
        state,
        targetUrl: "http://127.0.0.1:9099/v1/chat/completions",
      })

      const proof = providerContextProof(state)
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
})
