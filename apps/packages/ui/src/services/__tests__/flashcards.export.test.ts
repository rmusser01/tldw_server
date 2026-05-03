import { beforeEach, describe, expect, it, vi } from "vitest"

vi.mock("@/services/background-proxy", () => ({
  bgRequest: vi.fn(),
  bgUpload: vi.fn()
}))

vi.mock("@/services/resource-client", () => ({
  buildQuery: (params: Record<string, unknown>) => {
    const searchParams = new URLSearchParams()
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        searchParams.set(key, String(value))
      }
    })
    const query = searchParams.toString()
    return query ? `?${query}` : ""
  },
  createResourceClient: vi.fn(() => ({
    list: vi.fn(),
    get: vi.fn(),
    create: vi.fn(),
    update: vi.fn(),
    remove: vi.fn()
  }))
}))

import { bgRequest } from "@/services/background-proxy"
import { exportFlashcards } from "@/services/flashcards"

describe("flashcards export service", () => {
  beforeEach(() => {
    vi.mocked(bgRequest).mockReset()
    vi.mocked(bgRequest).mockResolvedValue("[]" as never)
  })

  it("requests text responses so JSON exports are not auto-parsed", async () => {
    await exportFlashcards({
      format: "json",
      tag: "chapter-1"
    })

    expect(bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        method: "GET",
        path: "/api/v1/flashcards/export?tag=chapter-1&format=json",
        responseType: "text"
      })
    )
  })
})
