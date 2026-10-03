import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const boundary = vi.hoisted(() => ({
  get: vi.fn(),
  fetch: vi.fn(),
  sendMessage: vi.fn(),
  runtimeId: null as string | null
}))
vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      get id() {
        return boundary.runtimeId
      },
      sendMessage: (...args: unknown[]) => boundary.sendMessage(...args)
    }
  }
}))
vi.mock("@/utils/safe-storage", () => ({
  safeStorageSerde: {
    serializer: JSON.stringify,
    deserializer: (value: unknown) => value
  },
  createSafeStorage: () => ({
    get: boundary.get,
    set: vi.fn(async () => {}),
    remove: vi.fn(async () => {})
  })
}))
vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () => null,
  isCookieSessionConfigInvalidated: () => false
}))

import { TldwApiClient } from "@/services/tldw/TldwApiClient"
import { bgRequest } from "../background-proxy"
import { requestScopeFields } from "../tldw/domains/service-prompts"
import { isServicePromptRequestPath } from "../tldw/service-prompt-scope-error"
import type { WebClipperSaveRequest } from "../web-clipper/types"

const config = (user = 7, serverUrl = "https://clips.example") => ({
  serverUrl,
  authMode: "multi-user" as const,
  accessToken: `fixture.${btoa(JSON.stringify({ sub: String(user) }))}.signature`
})
const requestScope = {
  config: {
    serverUrl: "https://clips.example",
    authMode: "multi-user" as const
  },
  userId: 7
}
const clipId = "925871f6-30fa-470b-a2c4-2272051f2373"
const payload: WebClipperSaveRequest = {
  clip_id: clipId,
  clip_type: "article",
  source_url: "https://example.com/story",
  source_title: "Article",
  destination_mode: "note",
  note: { title: "Article", comment: null, folder_id: null, keywords: [] },
  content: {
    visible_body: "Article excerpt",
    full_extract: "Article excerpt",
    selected_text: null
  },
  attachments: [],
  enhancements: { run_ocr: false, run_vlm: false },
  capture_metadata: { captured_via: "browser" }
}
const response = {
  clip_id: clipId,
  note_id: clipId,
  status: "saved",
  workspace_placement_saved: false,
  workspace_placement_count: 0,
  warnings: []
}
const save = (path = "/api/v1/web-clipper/save") => {
  const client = new TldwApiClient()
  vi.spyOn(client, "resolveApiPath").mockResolvedValue(path as never)
  return client.saveWebClip(payload, {
    requestScope,
    signal: new AbortController().signal
  })
}

beforeEach(() => {
  vi.clearAllMocks()
  boundary.runtimeId = null
  boundary.get.mockImplementation(async (key: string) =>
    key === "tldwConfig" ? config() : null
  )
  boundary.fetch.mockImplementation(
    async () =>
      new Response(JSON.stringify(response), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      })
  )
  boundary.sendMessage.mockResolvedValue({
    ok: true,
    status: 200,
    data: response
  })
  vi.stubGlobal("fetch", boundary.fetch)
})
afterEach(() => vi.unstubAllGlobals())

describe("captured web clip save through its scoped client and transport", () => {
  it.each(["/api/v1/web-clipper/save", "/api/v1/web-clipper/save/"])(
    "dispatches canonical POST %s under the captured principal",
    async (path) => {
      await expect(save(path)).resolves.toMatchObject({
        note_id: clipId,
        status: "saved"
      })
      expect(boundary.fetch).toHaveBeenCalledTimes(1)
      const [url, init] = boundary.fetch.mock.calls[0]
      expect(String(url)).toBe(`https://clips.example${path}`)
      expect(init.method).toBe("POST")
      expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("7")
      expect(new Headers(init.headers).get("Authorization")).toBe(
        `Bearer ${config().accessToken}`
      )
      expect(JSON.parse(init.body)).toEqual(payload)
      expect(init.signal).toBeInstanceOf(AbortSignal)
    }
  )

  it("retains captured config and expected principal on extension messaging", async () => {
    boundary.runtimeId = "extension-fixture"
    await expect(save()).resolves.toMatchObject({ note_id: clipId })
    expect(boundary.sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "tldw:request",
        payload: expect.objectContaining({
          path: "/api/v1/web-clipper/save",
          method: "POST",
          body: payload,
          servicePromptConfig: { ...requestScope.config, expectedUserId: 7 },
          headers: {
            "Content-Type": "application/json",
            "X-TLDW-Expected-User-ID": "7"
          }
        })
      })
    )
    expect(boundary.fetch).not.toHaveBeenCalled()
  })

  it.each(["principal", "server"])(
    "rejects a changed %s during credential loading before HTTP dispatch",
    async (change) => {
      let release!: (value: unknown) => void
      const pendingConfig = new Promise((resolve) => {
        release = resolve
      })
      boundary.get.mockImplementation(async (key: string) =>
        key === "tldwConfig" ? pendingConfig : null
      )
      const pending = save()
      const rejected = expect(pending).rejects.toMatchObject({ status: 412 })
      release(
        change === "principal" ? config(8) : config(7, "https://other.example")
      )
      await rejected
      expect(boundary.fetch).not.toHaveBeenCalled()
    }
  )

  it.each([
    "/api/v1/web-clipper/save/extra",
    "/api/v1/web-clipper/save//",
    "/api/v1/web-clipper//save",
    "/api/v1/web-clipper/../web-clipper/save",
    "/api/v1/web-clipper/%2e%2e/save",
    "/api/v1/web-clipper%2fsave",
    "/api/v1/web-clipper\\save",
    "/api/v1/web-clipper/save%zz",
    "/api/v1/web-clipper/clip-id/enrichments",
    "https://other.example/api/v1/web-clipper/save"
  ])("rejects malformed or expanded scoped save path %s", async (path) => {
    await expect(save(path)).rejects.toThrow(/Service Prompt config/)
    expect(boundary.fetch).not.toHaveBeenCalled()
    expect(boundary.sendMessage).not.toHaveBeenCalled()
  })

  it.each(["GET", "PUT", "PATCH", "DELETE"])(
    "rejects scoped save method %s before dispatch",
    async (method) => {
      expect(
        isServicePromptRequestPath("/api/v1/web-clipper/save", method)
      ).toBe(false)
      expect(
        isServicePromptRequestPath("/api/v1/web-clipper/save/", method)
      ).toBe(false)
      await expect(
        bgRequest({
          ...requestScopeFields(requestScope),
          path: "/api/v1/web-clipper/save",
          method: method as never
        })
      ).rejects.toThrow(/Service Prompt config/)
      expect(boundary.fetch).not.toHaveBeenCalled()
    }
  )
})
