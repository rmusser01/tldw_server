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

import { tldwMedia } from "@/services/tldw/TldwMedia"
import { deriveSingleUserApiKeyCredentialScope } from "@/services/chat-surface-scope"
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

describe("public article capture through actual scoped direct transport", () => {
  const options = () => ({
    requestScope: Object.freeze({
      config: Object.freeze({
        ...requestScope.config,
        expectedRefreshToken: "capture-refresh"
      }),
      userId: 7
    }),
    signal: new AbortController().signal
  })
  beforeEach(() => {
    boundary.get.mockImplementation(async (key: string) =>
      key === "tldwConfig"
        ? { ...config(), refreshToken: "capture-refresh" }
        : null
    )
  })
  it.each([
    ["extract", "/api/v1/media/ingest-web-content", "POST"],
    ["status", `/api/v1/web-clipper/${clipId}`, "GET"],
    ["status slash", `/api/v1/web-clipper/${clipId}/`, "GET"],
    ["versions", "/api/v1/media/71/versions?include_content=true", "GET"],
    ["version", "/api/v1/media/71/versions/7?include_content=true", "GET"],
    ["sources", "/api/v1/workspaces/workspace%20with%20spaces/sources", "GET"],
    [
      "preview",
      `/api/v1/workspaces/workspace%20with%20spaces/sources/web-clipper%3A${clipId}/preview?version_number=7`,
      "GET"
    ]
  ])(
    "dispatches the real %s client with frozen owner credentials",
    async (operation, path, method) => {
      const client = new TldwApiClient()
      boundary.fetch.mockImplementation(
        async (url: unknown) =>
          new Response(
            JSON.stringify(
              String(url).endsWith("/openapi.json")
                ? {
                    paths: {
                      [operation === "status slash"
                        ? "/api/v1/web-clipper/{clip_id}/"
                        : "/api/v1/web-clipper/{clip_id}"]: { get: {} }
                    }
                  }
                : response
            ),
            { status: 200, headers: { "Content-Type": "application/json" } }
          )
      )
      const captured = options()
      if (operation === "extract")
        await tldwMedia.extractPublicArticle(
          "https://example.com/story",
          captured
        )
      else if (operation.startsWith("status"))
        await client.getWebClipStatus(clipId, captured)
      else if (operation === "versions")
        await client.listMediaDocumentVersions(71, captured)
      else if (operation === "version")
        await client.getMediaDocumentVersion(71, 7, captured)
      else if (operation === "sources")
        await client.getWorkspaceSources("workspace with spaces", captured)
      else
        await client.getWorkspaceSourcePreview(
          "workspace with spaces",
          `web-clipper:${clipId}`,
          { version_number: 7 },
          captured
        )
      const requests = boundary.fetch.mock.calls.filter(
        ([url]) => !String(url).endsWith("/openapi.json")
      )
      expect(requests).toHaveLength(1)
      const [url, init] = requests[0]
      expect(String(url)).toBe(`https://clips.example${path}`)
      expect(init.method).toBe(method)
      expect(new Headers(init.headers).get("Authorization")).toBe(
        `Bearer ${config().accessToken}`
      )
      expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("7")
      expect(init.signal).toBeInstanceOf(AbortSignal)
      expect(captured.signal.aborted).toBe(false)
      if (operation === "extract")
        expect(JSON.parse(init.body)).toEqual({
          urls: ["https://example.com/story"],
          scrape_method: "individual",
          credential_free: true,
          perform_analysis: false,
          perform_translation: false,
          perform_chunking: false,
          auto_chunking_use_llm: false,
          use_cookies: false,
          overwrite_existing: false,
          perform_rolling_summarization: false,
          perform_confabulation_check_of_analysis: false
        })
    }
  )

  it.each(["principal", "origin", "refresh"])(
    "rejects changed capture %s before HTTP",
    async (change) => {
      boundary.get.mockImplementation(async (key: string) =>
        key === "tldwConfig"
          ? {
              ...config(
                change === "principal" ? 8 : 7,
                change === "origin"
                  ? "https://other.example"
                  : "https://clips.example"
              ),
              refreshToken:
                change === "refresh" ? "other-refresh" : "capture-refresh"
            }
          : null
      )
      await expect(
        tldwMedia.extractPublicArticle("https://example.com/story", options())
      ).rejects.toMatchObject({ status: 412 })
      expect(boundary.fetch).not.toHaveBeenCalled()
    }
  )
  it("rejects a changed capture API key before HTTP", async () => {
    boundary.get.mockImplementation(async (key: string) =>
      key === "tldwConfig"
        ? {
            serverUrl: "https://clips.example",
            authMode: "single-user",
            apiKey: "changed-key"
          }
        : null
    )
    await expect(
      tldwMedia.extractPublicArticle("https://example.com/story", {
        requestScope: {
          userId: null,
          config: {
            serverUrl: "https://clips.example",
            authMode: "single-user",
            expectedSingleUserApiKeyScope:
              deriveSingleUserApiKeyCredentialScope(
                "single-user",
                "captured-key"
              )!
          }
        }
      })
    ).rejects.toMatchObject({ status: 412 })
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
  it("aborts capture while HTTP is pending", async () => {
    const controller = new AbortController()
    const captured = { ...options(), signal: controller.signal }
    let started!: () => void
    const dispatched = new Promise<void>((resolve) => {
      started = resolve
    })
    boundary.fetch.mockImplementation(
      async (_url: unknown, init: RequestInit) =>
        new Promise((_resolve, reject) => {
          init.signal?.addEventListener("abort", () =>
            reject(new DOMException("Aborted", "AbortError"))
          )
          started()
        })
    )
    const pending = tldwMedia.extractPublicArticle(
      "https://example.com/story",
      captured
    )
    await Promise.race([dispatched, pending])
    controller.abort()
    await expect(pending).rejects.toMatchObject({ name: "AbortError" })
  })

  it.each([
    ["/api/v1/media/ingest-web-content", "GET"],
    ["/api/v1/media/ingest-web-content/extra", "POST"],
    ["/api/v1/media/71/versions", "POST"],
    ["/api/v1/media/71/versions/0", "GET"],
    ["/api/v1/media/71/versions/-1", "GET"],
    ["/api/v1/media/71/versions/advanced", "GET"],
    ["/api/v1/media/71/versions/7/metadata", "GET"],
    ["/api/v1/workspaces/ws/sources", "POST"],
    ["/api/v1/workspaces/ws/sources/src/preview", "PUT"],
    ["/api/v1/workspaces/ws/sources/src", "GET"],
    ["/api/v1/workspaces/ws/sources/status", "GET"],
    ["/api/v1/workspaces/ws/sources/src/preview/extra", "GET"],
    ["/api/v1/workspaces/../ws/sources", "GET"],
    ["/api/v1/workspaces/%2e%2e/sources", "GET"],
    ["/api/v1/workspaces/ws%2fforeign/sources", "GET"],
    ["/api/v1/workspaces/ws/sources/a%5cb/preview", "GET"],
    ["/api/v1/workspaces/ws%zz/sources", "GET"],
    ["/api/v1/web-clipper/clip/enrichments", "GET"],
    ["/api/v1/web-clipper/clip", "DELETE"],
    ["https://clips.example/api/v1/media/ingest-web-content", "POST"]
  ])(
    "rejects unsupported capture route %s %s before dispatch",
    async (path, method) => {
      await expect(
        bgRequest({
          ...requestScopeFields(options().requestScope),
          path: path as never,
          method: method as never
        })
      ).rejects.toThrow(/Service Prompt config/)
      expect(boundary.fetch).not.toHaveBeenCalled()
      expect(boundary.sendMessage).not.toHaveBeenCalled()
    }
  )

  it.each(["\t", "\r", "\n"])(
    "rejects raw URL-normalizing control %j before direct capture dispatch",
    async (control) => {
      for (const path of [
        `/api/v1/workspaces/${control}../sources`,
        `/api/v1/web-clipper/${control}..`,
        `/api/v1/workspaces/ws/sources/${control}../preview`
      ]) {
        await expect(
          bgRequest({
            ...requestScopeFields(options().requestScope),
            path: path as never,
            method: "GET"
          })
        ).rejects.toThrow(/Service Prompt config/)
        expect(boundary.fetch).not.toHaveBeenCalled()
        expect(boundary.sendMessage).not.toHaveBeenCalled()
      }
    }
  )
})
