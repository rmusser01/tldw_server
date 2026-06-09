import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: vi.fn(),
  bgStream: vi.fn()
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

describe("TldwApiClient Explainer methods", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.bgRequest.mockResolvedValue({ id: "session-1" })
  })

  it("creates an Explainer session through the typed client", async () => {
    const client = new TldwApiClient()

    await client.createExplainerSession({
      mode: "goal",
      title: "Learn attention",
      outputIntent: "explain",
      grounding: "open",
      depthPreset: "standard",
      rootPrompt: "Explain transformer attention",
      selectedSources: []
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/explainer/sessions",
        method: "POST",
        body: expect.objectContaining({
          mode: "goal",
          outputIntent: "explain",
          depthPreset: "standard",
          rootPrompt: "Explain transformer attention",
          selectedSources: []
        })
      })
    )
  })

  it("lists and fetches Explainer sessions", async () => {
    const client = new TldwApiClient()

    await client.listExplainerSessions({ limit: 25, offset: 50 })
    await client.getExplainerSession("session-1")

    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        path: "/api/v1/explainer/sessions?limit=25&offset=50",
        method: "GET"
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        path: "/api/v1/explainer/sessions/session-1",
        method: "GET"
      })
    )
  })

  it("updates and deletes Explainer sessions", async () => {
    const client = new TldwApiClient()

    await client.updateExplainerSession("session-1", {
      outputIntent: "both",
      grounding: "source_led"
    })
    await client.deleteExplainerSession("session-1")

    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        path: "/api/v1/explainer/sessions/session-1",
        method: "PATCH",
        body: { outputIntent: "both", grounding: "source_led" }
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        path: "/api/v1/explainer/sessions/session-1",
        method: "DELETE"
      })
    )
  })

  it("manages Explainer nodes and generation jobs", async () => {
    const client = new TldwApiClient()

    await client.createExplainerNode("session-1", {
      parentId: "root",
      title: "Scaled attention",
      intent: "explain"
    })
    await client.updateExplainerNode("session-1", "node-2", {
      body: "Updated",
      evidenceState: "supported"
    })
    await client.expandExplainerNode("session-1", "node-2", { intent: "plan" })
    await client.answerExplainerQuestion("session-1", "node-2", {
      selectedOptionId: "math"
    })
    await client.deleteExplainerNode("session-1", "node-2")
    await client.getExplainerJob("job-1")

    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        path: "/api/v1/explainer/sessions/session-1/nodes",
        method: "POST"
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        path: "/api/v1/explainer/sessions/session-1/nodes/node-2",
        method: "PATCH",
        body: { body: "Updated", evidenceState: "supported" }
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      3,
      expect.objectContaining({
        path: "/api/v1/explainer/sessions/session-1/nodes/node-2/expand",
        method: "POST",
        body: { intent: "plan" }
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      4,
      expect.objectContaining({
        path: "/api/v1/explainer/sessions/session-1/nodes/node-2/answer-question",
        method: "POST",
        body: { selectedOptionId: "math" }
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      5,
      expect.objectContaining({
        path: "/api/v1/explainer/sessions/session-1/nodes/node-2",
        method: "DELETE"
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      6,
      expect.objectContaining({
        path: "/api/v1/explainer/jobs/job-1",
        method: "GET"
      })
    )
  })

  it("exports an Explainer session as a Chatbook item", async () => {
    const client = new TldwApiClient()

    await client.exportExplainerChatbook("session-1", {
      name: "Attention explainer",
      asyncMode: false
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/explainer/sessions/session-1/export-chatbook",
        method: "POST",
        body: {
          name: "Attention explainer",
          asyncMode: false
        }
      })
    )
  })
})
