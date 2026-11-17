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


import { bgRequest } from "../background-proxy"
import { requestScopeFields } from "../tldw/domains/service-prompts"
import { getAudioStudioProject, createAudioStudioProject, upsertAudioStudioSection, createAudioStudioGeneration, createAudioStudioRender, createAudioStudioExport, listAudioStudioJobs, cancelAudioStudioJob, getAudioStudioSpeechProviders, getAudioStudioVoices } from "../audio-studio"
const serverUrl = "https://studio.example"
const config = (user = 7) => ({ serverUrl, authMode: "multi-user" as const, authSource: "manual" as const, accessToken: `fixture.${btoa(JSON.stringify({sub: String(user)}))}.signature` })
const options = { requestScope: { config: { serverUrl, authMode: "multi-user" as const, authSource: "manual" as const }, userId: 7 }, signal: new AbortController().signal }
const target = { idempotency_key: "studio-unit-operation", target_resource_kind: "section", target_resource_id: "s", target_revision_id: "r" }
const operations = [
  { name: "detail", invoke: () => getAudioStudioProject("p", options) },
  { name: "create", invoke: () => createAudioStudioProject({ title: "Private", workflow: "narration" }, options) },
  { name: "section", invoke: () => upsertAudioStudioSection("p", "s", { base_revision_id: "r", body_text: "Text" }, options) },
  { name: "generate", invoke: () => createAudioStudioGeneration("p", { ...target, kind: "speech", provider: "tts" }, options) },
  { name: "render", invoke: () => createAudioStudioRender("p", { ...target, render_type: "preview_mix" }, options) },
  { name: "export", invoke: () => createAudioStudioExport("p", { ...target, export_type: "single_audio" }, options) },
  { name: "jobs", invoke: () => listAudioStudioJobs("p", options) },
  { name: "cancel", invoke: () => cancelAudioStudioJob("p", "j", options) },
  { name: "speech providers", invoke: () => getAudioStudioSpeechProviders(options) },
  { name: "voices", invoke: () => getAudioStudioVoices("pocket_tts", "pocket_tts", options) },
]
beforeEach(() => {
  vi.clearAllMocks(); boundary.runtimeId = null
  boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? config() : null)
  boundary.fetch.mockImplementation(async () => new Response(JSON.stringify({ jobs: [], providers: {}, voices: [] }), { status: 200, headers: { "Content-Type": "application/json" } }))
  boundary.sendMessage.mockResolvedValue({ ok: true, status: 200, data: { jobs: [], providers: {}, voices: [] } })
  vi.stubGlobal("fetch", boundary.fetch)
})
afterEach(() => vi.unstubAllGlobals())
describe("B06 actual captured Studio transport", () => {
  it.each(operations)("dispatches direct $name with captured owner", async ({ invoke, name }) => {
    await invoke()
    expect(boundary.fetch).toHaveBeenCalledTimes(name === "voices" ? 2 : 1)
    for (const [url, init] of boundary.fetch.mock.calls) {
    expect(String(url)).toMatch(/^https:\/\/studio\.example\/api\/v1\/audio/)
    expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("7")
    expect(new Headers(init.headers).get("Authorization")).toBe(`Bearer ${config().accessToken}`)
    expect(init.signal).toBeInstanceOf(AbortSignal)
    }
  })
  it.each(operations)("retains extension scope for $name", async ({ invoke, name }) => {
    boundary.runtimeId = "extension-fixture"
    await invoke()
    expect(boundary.sendMessage).toHaveBeenCalledTimes(name === "voices" ? 2 : 1)
    for (const [message] of boundary.sendMessage.mock.calls) expect(message.payload.servicePromptConfig).toMatchObject({ serverUrl, expectedUserId: 7 })
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
  it.each(operations)("refuses changed principal before $name dispatch", async ({ invoke }) => {
    boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? config(8) : null)
    await expect(invoke()).rejects.toMatchObject({ status: 412 })
    expect(boundary.fetch).not.toHaveBeenCalled()
    expect(boundary.sendMessage).not.toHaveBeenCalled()
  })
  it("B06 refuses a principal change between owned voice and builtin catalog reads", async () => {
    boundary.fetch.mockImplementationOnce(async () => {
      boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? config(8) : null)
      return new Response(JSON.stringify({ voices: [{ voice_id: "owned", provider: "pocket_tts", name: "Private" }] }), { status: 200, headers: { "Content-Type": "application/json" } })
    })
    await expect(getAudioStudioVoices("pocket_tts", "pocket_tts", options)).rejects.toMatchObject({ status: 412 })
    expect(boundary.fetch).toHaveBeenCalledTimes(1)
  })
  it.each([
    ["/api/v1/audio-studio/projects/p", "DELETE"],
    ["/api/v1/audio-studio/projects/p/jobs/j/cancel", "GET"],
    ["/api/v1/audio-studio/projects/p/admin", "GET"],
    ["/api/v1/audio-studio/projects/p/sections/s/extra", "PUT"],
    ["/api/v1/audio-studio/projects/a%2fb", "GET"],
    ["/api/v1/audio/providers", "POST"],
    ["/api/v1/audio/voices", "POST"],
    ["/api/v1/audio/voices/owned", "GET"],
    ["/api/v1/audio/voices/catalog/extra", "GET"],
  ])("refuses unrepresented method/path %s %s", async (path, method) => {
    await expect(bgRequest({ path: path as never, method: method as never, ...requestScopeFields(options.requestScope) })).rejects.toThrow(/Service Prompt config/)
    expect(boundary.fetch).not.toHaveBeenCalled()
    expect(boundary.sendMessage).not.toHaveBeenCalled()
  })
})
