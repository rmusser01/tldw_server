import { describe, expect, it } from "vitest"
import {
  buildDynamicUIEnvelope,
  formatDynamicUIActionUserMessage,
  normalizeDynamicUIActionPayload,
  normalizeDynamicUIEnvelope,
  preflightOpenUISource,
  shouldBlockDynamicUIActionValues
} from "../dynamic-ui"

describe("dynamic UI utilities", () => {
  it("normalizes valid OpenUI envelopes", () => {
    const envelope = normalizeDynamicUIEnvelope({
      renderer: "openui",
      version: "v1",
      source: "root = <Card />",
      state: { count: 1 },
      capabilities: ["forms"]
    })

    expect(envelope).toEqual({
      renderer: "openui",
      version: "v1",
      source: "root = <Card />",
      state: { count: 1 },
      capabilities: ["forms"]
    })
  })

  it("rejects envelopes with non-strict JSON state", () => {
    const circular: Record<string, unknown> = {}
    circular.self = circular

    expect(
      normalizeDynamicUIEnvelope({
        renderer: "openui",
        version: "v1",
        source: "root = <Card />",
        state: { callback: () => undefined }
      })
    ).toBeNull()
    expect(
      normalizeDynamicUIEnvelope({
        renderer: "openui",
        version: "v1",
        source: "root = <Card />",
        state: { selected: new Map([["a", "b"]]) }
      })
    ).toBeNull()
    expect(
      normalizeDynamicUIEnvelope({
        renderer: "openui",
        version: "v1",
        source: "root = <Card />",
        state: circular
      })
    ).toBeNull()
  })

  it("rejects unknown renderers and empty source", () => {
    expect(normalizeDynamicUIEnvelope({ renderer: "html", source: "<script />" })).toBeNull()
    expect(normalizeDynamicUIEnvelope({ renderer: "openui", source: "" })).toBeNull()
  })

  it("rejects unsupported dynamic UI contract versions", () => {
    expect(
      normalizeDynamicUIEnvelope({
        renderer: "openui",
        version: "v2",
        source: "root = <Card />"
      })
    ).toBeNull()
  })

  it("preflights only plausible completed OpenUI source", () => {
    expect(preflightOpenUISource("root = <Card><Text>Hello</Text></Card>").ok).toBe(true)
    expect(preflightOpenUISource("I cannot produce that UI.").ok).toBe(false)
  })

  it("builds envelopes only after source preflight", () => {
    expect(buildDynamicUIEnvelope("openui", "root = <Card />")).toMatchObject({
      renderer: "openui",
      source: "root = <Card />"
    })
    expect(buildDynamicUIEnvelope("openui", "plain refusal")).toBeNull()
  })

  it("normalizes action payloads and blocks sensitive-looking values", () => {
    const payload = normalizeDynamicUIActionPayload({
      renderer: "openui",
      sourceMessageId: "assistant-1",
      actionId: "profile-submit",
      actionType: "submit",
      values: { name: "Ada", password: "secret" }
    }, { currentMessageIds: new Set(["assistant-1"]) })

    expect(payload?.actionId).toBe("profile-submit")
    expect(shouldBlockDynamicUIActionValues(payload?.values)).toBe(true)
  })

  it("blocks nested sensitive-looking action values", () => {
    expect(shouldBlockDynamicUIActionValues({ profile: { password: "secret" } })).toBe(true)
    expect(shouldBlockDynamicUIActionValues([{ settings: { authToken: "abc123" } }])).toBe(true)
  })

  it("blocks top-level sensitive key variants", () => {
    expect(shouldBlockDynamicUIActionValues({ key: "abc123" })).toBe(true)
    expect(shouldBlockDynamicUIActionValues({ privateKey: "abc123" })).toBe(true)
    expect(shouldBlockDynamicUIActionValues({ access_key: "abc123" })).toBe(true)
    expect(shouldBlockDynamicUIActionValues({ "public-key": "abc123" })).toBe(true)
  })

  it("blocks nested sensitive key variants", () => {
    expect(shouldBlockDynamicUIActionValues({ settings: { privateKey: "abc123" } })).toBe(true)
    expect(shouldBlockDynamicUIActionValues([{ config: { access_key: "abc123" } }])).toBe(true)
    expect(shouldBlockDynamicUIActionValues({ auth: { keys: [{ key: "abc123" }] } })).toBe(true)
  })

  it("blocks plural and authorization sensitive key variants", () => {
    expect(shouldBlockDynamicUIActionValues({ apiKeys: ["abc123"] })).toBe(true)
    expect(shouldBlockDynamicUIActionValues({ settings: { accessKeys: ["abc123"] } })).toBe(true)
    expect(shouldBlockDynamicUIActionValues({ credentials: { username: "Ada" } })).toBe(true)
    expect(shouldBlockDynamicUIActionValues([{ authorization: "Bearer abc123" }])).toBe(true)
  })

  it("blocks when sensitive action value inspection exceeds the depth limit", () => {
    expect(shouldBlockDynamicUIActionValues({ harmless: "value" }, 9)).toBe(true)

    const deeplyNestedSensitiveValue = {
      a: { b: { c: { d: { e: { f: { g: { h: { i: { token: "abc123" } } } } } } } } }
    }
    expect(shouldBlockDynamicUIActionValues(deeplyNestedSensitiveValue)).toBe(true)
  })

  it("rejects non-serializable action values without throwing", () => {
    const circular: Record<string, unknown> = {}
    circular.self = circular
    expect(
      normalizeDynamicUIActionPayload(
        {
          renderer: "openui",
          sourceMessageId: "assistant-1",
          actionId: "bad",
          actionType: "submit",
          values: circular
        },
        { currentMessageIds: new Set(["assistant-1"]) }
      )
    ).toBeNull()
  })

  it("rejects action values that are not strict JSON values", () => {
    expect(
      normalizeDynamicUIActionPayload(
        {
          renderer: "openui",
          sourceMessageId: "assistant-1",
          actionId: "bad",
          actionType: "submit",
          values: { callback: () => undefined }
        },
        { currentMessageIds: new Set(["assistant-1"]) }
      )
    ).toBeNull()

    expect(
      normalizeDynamicUIActionPayload(
        {
          renderer: "openui",
          sourceMessageId: "assistant-1",
          actionId: "bad",
          actionType: "submit",
          values: { nested: { missing: undefined, token: Symbol("secret") } }
        },
        { currentMessageIds: new Set(["assistant-1"]) }
      )
    ).toBeNull()
  })

  it("rejects Blob-like action values", () => {
    const blob = typeof Blob === "function" ? new Blob(["file contents"]) : { [Symbol.toStringTag]: "Blob", size: 13 }
    expect(
      normalizeDynamicUIActionPayload(
        {
          renderer: "openui",
          sourceMessageId: "assistant-1",
          actionId: "bad",
          actionType: "submit",
          values: { upload: blob }
        },
        { currentMessageIds: new Set(["assistant-1"]) }
      )
    ).toBeNull()
  })

  it("rejects non-plain action value objects", () => {
    expect(
      normalizeDynamicUIActionPayload(
        {
          renderer: "openui",
          sourceMessageId: "assistant-1",
          actionId: "bad",
          actionType: "submit",
          values: { selected: new Map([["a", "b"]]) }
        },
        { currentMessageIds: new Set(["assistant-1"]) }
      )
    ).toBeNull()

    class CustomValue { answer = "yes" }

    expect(
      normalizeDynamicUIActionPayload(
        {
          renderer: "openui",
          sourceMessageId: "assistant-1",
          actionId: "bad",
          actionType: "submit",
          values: { custom: new CustomValue() }
        },
        { currentMessageIds: new Set(["assistant-1"]) }
      )
    ).toBeNull()
  })

  it("formats action payloads as visible user messages", () => {
    const text = formatDynamicUIActionUserMessage({
      renderer: "openui",
      sourceMessageId: "assistant-1",
      actionId: "survey",
      actionType: "submit",
      values: { answer: "yes" },
      submittedAt: "2026-06-01T00:00:00.000Z"
    })

    expect(text).toContain("OpenUI action: submit survey")
    expect(text).toContain("- answer: yes")
  })
})
