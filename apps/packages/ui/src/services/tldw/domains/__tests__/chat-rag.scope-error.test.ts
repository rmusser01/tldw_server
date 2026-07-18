import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgStream: vi.fn(),
  bgUpload: vi.fn()
}))

import { chatRagMethods } from "../chat-rag"

describe("chat RAG scoped errors", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("preserves a structured request-scope rejection", async () => {
    const scopeError = Object.assign(new Error("scope changed"), {
      status: 412,
      details: {
        detail: { code: "request_config_scope_changed" }
      }
    })
    mocks.bgRequest.mockRejectedValueOnce(scopeError)

    const request = chatRagMethods.ragSearch.call(
      {
        normalizeRagQuery: (query: string) => query
      } as any,
      "question",
      {
        requestScope: {
          config: {
            serverUrl: "https://server.test",
            authMode: "multi-user"
          },
          userId: 42
        }
      }
    )

    await expect(request).rejects.toBe(scopeError)
  })
})
