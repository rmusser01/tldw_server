import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgUpload: vi.fn(),
  bgStream: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: (...args: unknown[]) => mocks.bgUpload(...args),
  bgStream: (...args: unknown[]) => mocks.bgStream(...args)
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: vi.fn(async () => null),
    set: vi.fn(async () => undefined),
    remove: vi.fn(async () => undefined)
  }),
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  }
}))

import { TldwApiClient } from "@/services/tldw/TldwApiClient"

describe("TldwApiClient getModels normalization", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("prefers model-like name over conflicting id when model field is absent", async () => {
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      if (request.path === "/api/v1/llm/models/metadata") {
        return {
          models: [
            {
              id: "z-ai/glm-4.6",
              name: "deepseek/deepseek-r1",
              provider: "openrouter",
              type: "chat"
            }
          ]
        }
      }
      return {}
    })

    const client = new TldwApiClient()
    const models = await client.getModels()

    expect(models).toHaveLength(1)
    expect(models[0]?.id).toBe("deepseek/deepseek-r1")
    expect(models[0]?.name).toBe("deepseek/deepseek-r1")
  })

  it("keeps canonical id and appends friendly label when name is non-model text", async () => {
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      if (request.path === "/api/v1/llm/models/metadata") {
        return {
          models: [
            {
              id: "openai/gpt-4o-mini",
              name: "GPT-4o Mini",
              provider: "openai",
              type: "chat"
            }
          ]
        }
      }
      return {}
    })

    const client = new TldwApiClient()
    const models = await client.getModels()

    expect(models).toHaveLength(1)
    expect(models[0]?.id).toBe("openai/gpt-4o-mini")
    expect(models[0]?.name).toBe("GPT-4o Mini (openai/gpt-4o-mini)")
  })

  it("routes llama.cpp profile runtime methods to managed runtime endpoints", async () => {
    const requests: Array<{ path?: string; method?: string; body?: unknown }> = []
    mocks.bgRequest.mockImplementation(async (request: { path?: string; method?: string; body?: unknown }) => {
      requests.push(request)
      if (request.path === "/api/v1/llamacpp/profiles") return { profiles: [] }
      if (request.path === "/api/v1/llamacpp/instances") return { runtimes: [] }
      if (request.path === "/api/v1/llamacpp/instances/default/logs/tail?lines=10") {
        return { lines: [], truncated: false, warnings: [] }
      }
      return { profile_id: "default", action: "ok", state: "running", accepted: true }
    })

    const client = new TldwApiClient()

    await client.listLlamacppProfiles()
    await client.createLlamacppProfile({ name: "Default", model_id: "gguf:default" })
    await client.updateLlamacppProfile("default", { name: "Updated" })
    await client.deleteLlamacppProfile("default")
    await client.startLlamacppProfile("default")
    await client.stopLlamacppProfile("default")
    await client.pauseLlamacppProfile("default")
    await client.resumeLlamacppProfile("default")
    await client.useLlamacppProfileInChat("default")
    await client.listLlamacppInstances()
    await client.tailLlamacppInstanceLogs("default", 10)

    expect(requests.map((request) => [request.method, request.path])).toEqual([
      ["GET", "/api/v1/llamacpp/profiles"],
      ["POST", "/api/v1/llamacpp/profiles"],
      ["PUT", "/api/v1/llamacpp/profiles/default"],
      ["DELETE", "/api/v1/llamacpp/profiles/default"],
      ["POST", "/api/v1/llamacpp/profiles/default/start"],
      ["POST", "/api/v1/llamacpp/profiles/default/stop"],
      ["POST", "/api/v1/llamacpp/profiles/default/pause"],
      ["POST", "/api/v1/llamacpp/profiles/default/resume"],
      ["POST", "/api/v1/llamacpp/profiles/default/use-in-chat"],
      ["GET", "/api/v1/llamacpp/instances"],
      ["GET", "/api/v1/llamacpp/instances/default/logs/tail?lines=10"]
    ])
    expect(requests[1]?.body).toEqual({ name: "Default", model_id: "gguf:default" })
    expect(requests[2]?.body).toEqual({ name: "Updated" })
  })

  it("routes llama.cpp acquisition workflow methods", async () => {
    const requests: Array<{ path?: string; method?: string; body?: unknown }> = []
    mocks.bgRequest.mockImplementation(
      async (request: { path?: string; method?: string; body?: unknown }) => {
        requests.push(request)
        if (request.path === "/api/v1/llamacpp/assets/import-folder/preview") {
          return {
            folder: {},
            assets: [],
            asset_counts: {},
            warnings: [],
            scan_limited: false,
            will_persist: false
          }
        }
        if (request.path === "/api/v1/llamacpp/assets/downloads") {
          return request.method === "GET"
            ? { jobs: [] }
            : {
                job_id: "42",
                status: "queued",
                operation: "download",
                queue: "acquisition",
                progress: {},
                warnings: []
              }
        }
        return {
          job_id: "42",
          status: request.method === "DELETE" ? "canceled" : "running",
          operation: "download",
          queue: "acquisition",
          progress: {},
          warnings: []
        }
      }
    )

    const client = new TldwApiClient()

    await client.previewLlamacppAssetFolder("/models")
    await client.startLlamacppAssetDownload({
      url: "https://example.com/model.gguf",
      destination_dir: "/models",
      filename: "model.gguf"
    })
    await client.listLlamacppAssetDownloads()
    await client.getLlamacppAssetDownload("42")
    await client.cancelLlamacppAssetDownload("42")

    expect(requests.map((request) => [request.method, request.path])).toEqual([
      ["POST", "/api/v1/llamacpp/assets/import-folder/preview"],
      ["POST", "/api/v1/llamacpp/assets/downloads"],
      ["GET", "/api/v1/llamacpp/assets/downloads"],
      ["GET", "/api/v1/llamacpp/assets/downloads/42"],
      ["DELETE", "/api/v1/llamacpp/assets/downloads/42"]
    ])
    expect(requests[0]?.body).toEqual({ path: "/models" })
    expect(requests[1]?.body).toEqual({
      url: "https://example.com/model.gguf",
      destination_dir: "/models",
      filename: "model.gguf"
    })
  })
})
