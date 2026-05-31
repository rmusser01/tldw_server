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

  it("preserves backend model configuration flags for readiness gating", async () => {
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      if (request.path === "/api/v1/llm/models/metadata") {
        return {
          models: [
            {
              name: "gpt-4o",
              provider: "openai",
              type: "chat",
              is_configured: false,
              provider_is_configured: false,
              provider_enabled: false,
              availability: "unavailable",
              catalog_only: true
            },
            {
              name: "legacy-model",
              provider: "custom",
              type: "chat",
              is_configured: "false",
              provider_is_configured: 1,
              catalog_only: null
            }
          ]
        }
      }
      return {}
    })

    const client = new TldwApiClient()
    const models = await client.getModels()

    expect(models[0]).toEqual(
      expect.objectContaining({
        id: "gpt-4o",
        provider: "openai",
        is_configured: false,
        provider_is_configured: false,
        provider_enabled: false,
        availability: "unavailable",
        catalog_only: true
      })
    )
    expect(models[1]).toEqual(
      expect.objectContaining({
        id: "legacy-model",
        provider: "custom",
        is_configured: undefined,
        provider_is_configured: undefined,
        provider_enabled: undefined,
        availability: undefined,
        catalog_only: undefined
      })
    )
  })

  it("enriches model availability from the provider listing when metadata omits it", async () => {
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      if (request.path === "/api/v1/llm/models/metadata") {
        return {
          models: [
            {
              id: "openai/gpt-4o-mini",
              provider: "openai",
              type: "chat"
            },
            {
              id: "anthropic/claude-sonnet-4",
              provider: "anthropic",
              type: "chat"
            }
          ]
        }
      }
      if (request.path === "/api/v1/llm/providers") {
        return {
          providers: [
            {
              name: "openai",
              is_configured: true,
              enabled: true,
              availability: "available"
            },
            {
              name: "anthropic",
              is_configured: false,
              enabled: false,
              availability: "unavailable"
            }
          ]
        }
      }
      return {}
    })

    const client = new TldwApiClient()
    const models = await client.getModels()

    expect(models[0]).toEqual(
      expect.objectContaining({
        id: "openai/gpt-4o-mini",
        is_configured: true,
        provider_is_configured: true,
        provider_enabled: true,
        availability: "available"
      })
    )
    expect(models[1]).toEqual(
      expect.objectContaining({
        id: "anthropic/claude-sonnet-4",
        is_configured: false,
        provider_is_configured: false,
        provider_enabled: false,
        availability: "unavailable"
      })
    )
  })

  it("matches provider availability across provider key aliases", async () => {
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      if (request.path === "/api/v1/llm/models/metadata") {
        return {
          models: [
            {
              id: "local/llama",
              provider: "llamacpp",
              type: "chat"
            },
            {
              id: "custom/model",
              provider: "custom-openai-api",
              type: "chat"
            }
          ]
        }
      }
      if (request.path === "/api/v1/llm/providers") {
        return {
          providers: [
            {
              name: "llama.cpp",
              is_configured: false,
              enabled: false
            },
            {
              name: "custom_openai_api",
              is_configured: true,
              enabled: true
            }
          ]
        }
      }
      return {}
    })

    const client = new TldwApiClient()
    const models = await client.getModels()

    expect(models.find((model) => model.id === "local/llama")).toEqual(
      expect.objectContaining({
        is_configured: false,
        provider_is_configured: false,
        provider_enabled: false
      })
    )
    expect(models.find((model) => model.id === "custom/model")).toEqual(
      expect.objectContaining({
        is_configured: true,
        provider_is_configured: true,
        provider_enabled: true
      })
    )
  })

  it("skips provider listing when model metadata already has availability status", async () => {
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      if (request.path === "/api/v1/llm/models/metadata") {
        return {
          models: [
            {
              id: "openai/gpt-4o-mini",
              provider: "openai",
              type: "chat",
              is_configured: true,
              provider_enabled: true,
              availability: "available"
            }
          ]
        }
      }
      if (request.path === "/api/v1/llm/providers") {
        throw new Error("provider listing should not be fetched")
      }
      return {}
    })

    const client = new TldwApiClient()
    const models = await client.getModels()

    expect(models[0]).toEqual(
      expect.objectContaining({
        id: "openai/gpt-4o-mini",
        is_configured: true,
        provider_enabled: true,
        availability: "available"
      })
    )
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
  })

  it("enriches chat providers even when non-chat catalog entries already have availability metadata", async () => {
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      if (request.path === "/api/v1/llm/models/metadata") {
        return {
          models: [
            {
              id: "openai/gpt-4o-mini",
              provider: "openai",
              type: "chat"
            },
            {
              id: "anthropic/claude-sonnet-4",
              provider: "anthropic",
              type: "chat"
            },
            {
              id: "image/stable_diffusion_cpp",
              provider: "image",
              type: "image",
              is_configured: false
            }
          ]
        }
      }
      if (request.path === "/api/v1/llm/providers") {
        return {
          providers: [
            {
              name: "openai",
              is_configured: true
            },
            {
              name: "anthropic",
              is_configured: false
            }
          ]
        }
      }
      return {}
    })

    const client = new TldwApiClient()
    const models = await client.getModels()

    expect(models.find((model) => model.id === "openai/gpt-4o-mini")).toEqual(
      expect.objectContaining({
        is_configured: true
      })
    )
    expect(models.find((model) => model.id === "anthropic/claude-sonnet-4")).toEqual(
      expect.objectContaining({
        is_configured: false
      })
    )
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
