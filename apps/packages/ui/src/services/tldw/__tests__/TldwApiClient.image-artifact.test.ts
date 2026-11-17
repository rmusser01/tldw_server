import { beforeEach, describe, expect, it, vi } from "vitest"

const transport = vi.hoisted(() => ({ request: vi.fn() }))
vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => transport.request(...args),
  bgStream: vi.fn(),
  bgUpload: vi.fn()
}))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: vi.fn(async () => null),
    set: vi.fn(),
    remove: vi.fn()
  }),
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  }
}))

import { TldwApiClient, TldwApiClientBase } from "../TldwApiClient"
import type { ScopedRequestOptions } from "../TldwApiClient"

const scope: NonNullable<ScopedRequestOptions["requestScope"]> = {
  config: { serverUrl: "https://original-image.test", authMode: "multi-user" },
  userId: 42
}
const image = {
  backend: "fluxklein",
  prompt: "original image",
  width: 512,
  referenceFileId: 7
}
const response = { artifact: { file_id: 77 } }

beforeEach(() => {
  vi.restoreAllMocks()
  transport.request.mockReset()
})

describe.each([TldwApiClientBase, TldwApiClient])(
  "runtime image artifact: %s",
  (Client) => {
    it("uses the captured request owner and signal through the actual public method", async () => {
      const client = new Client()
      const replacementConfig = vi
        .spyOn(client, "getConfig")
        .mockResolvedValue({
          serverUrl: "https://replacement-image.test",
          authMode: "single-user",
          apiKey: "image-fixture-key"
        })
      const controller = new AbortController()
      let release!: (value: typeof response) => void
      transport.request.mockImplementationOnce(
        () =>
          new Promise<typeof response>((resolve) => {
            release = resolve
          })
      )
      const pending = client.createImageArtifact(image, {
        requestScope: scope,
        signal: controller.signal
      })
      const settled = pending.catch((error) => error)
      await vi.waitFor(() => expect(transport.request).toHaveBeenCalledOnce())
      const request = transport.request.mock.calls[0][0]
      expect(request).toMatchObject({
        path: "/api/v1/files/create",
        method: "POST",
        abortSignal: controller.signal,
        servicePromptConfig: { ...scope.config, expectedUserId: 42 },
        headers: { "X-TLDW-Expected-User-ID": "42" },
        body: {
          file_type: "image",
          payload: {
            backend: "fluxklein",
            prompt: "original image",
            width: 512,
            reference_file_id: 7
          }
        }
      })
      expect(replacementConfig).not.toHaveBeenCalled()
      release(response)
      expect(await settled).toBe(response)
    })

    it("refuses a stopped image request before entering transport", async () => {
      const client = new Client()
      transport.request.mockResolvedValue(response)
      await expect(
        client.createImageArtifact(image, {
          requestScope: scope,
          signal: AbortSignal.abort()
        })
      ).rejects.toMatchObject({ name: "AbortError" })
      expect(transport.request).not.toHaveBeenCalled()
    })

    it("keeps the existing payload-only image caller working", async () => {
      const client = new Client()
      vi.spyOn(client, "getConfig").mockResolvedValue({
        serverUrl: "https://images.test",
        authMode: "single-user",
        apiKey: "image-fixture-key"
      })
      transport.request.mockResolvedValue(response)
      expect(await client.createImageArtifact(image)).toBe(response)
      expect(transport.request.mock.calls[0][0]).toMatchObject({
        path: "/api/v1/files/create",
        body: { payload: { prompt: "original image" } }
      })
    })
  }
)
