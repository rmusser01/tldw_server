import { describe, expect, it, vi } from "vitest"

import { chatRagMethods } from "@/services/tldw/domains/chat-rag"

describe("ragSourceHealth client", () => {
  it("requests the focused source health endpoint", async () => {
    const request = vi.fn().mockResolvedValue({ sources: [] })

    await chatRagMethods.ragSourceHealth.call({ request } as any)

    expect(request).toHaveBeenCalledWith({
      path: "/api/v1/rag/source-health",
      method: "GET",
    })
  })
})
