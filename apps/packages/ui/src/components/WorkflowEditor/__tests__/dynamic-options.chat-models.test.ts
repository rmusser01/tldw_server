// @vitest-environment jsdom

import { beforeEach, describe, expect, it, vi } from "vitest"
import { renderHook, waitFor } from "@testing-library/react"
import { useWorkflowDynamicOptions } from "../dynamic-options"
import type { ConfigFieldSchema } from "@/types/workflow-editor"

const { getChatModels } = vi.hoisted(() => ({ getChatModels: vi.fn() }))

vi.mock("@/services/evaluations", () => ({
  listDatasets: vi.fn(),
  listEvaluations: vi.fn(),
  listRuns: vi.fn(),
  listRunsGlobal: vi.fn()
}))

vi.mock("@/services/tldw", () => ({
  tldwClient: {},
  tldwModels: {
    getChatModels,
    getEmbeddingModels: vi.fn(),
    getImageModels: vi.fn()
  }
}))

vi.mock("@/services/tldw/audio-models", () => ({ fetchTldwTtsModels: vi.fn() }))
vi.mock("@/services/tldw/audio-providers", () => ({ fetchTtsProviders: vi.fn() }))
vi.mock("@/services/tldw/audio-voices", () => ({
  fetchTldwVoices: vi.fn(),
  fetchTldwVoiceCatalog: vi.fn()
}))
vi.mock("@/services/tldw/embedding-collections", () => ({
  fetchEmbeddingCollections: vi.fn()
}))
vi.mock("@/services/folder-api", () => ({ fetchFolders: vi.fn() }))

describe("Workflow editor chat model options (TASK-15583)", () => {
  const fields: ConfigFieldSchema[] = [{ key: "model", type: "select", label: "Model" }]
  const load = async () => {
    const { result, unmount } = renderHook(() =>
      useWorkflowDynamicOptions({ fields, stepType: "llm", config: {} })
    )
    await waitFor(() => expect(result.current.optionsByKey.model?.length).toBeGreaterThan(0))
    const options = result.current.optionsByKey.model
    unmount()
    return options
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("does not reuse one user's own-key models for the next load", async () => {
    getChatModels.mockResolvedValueOnce([{ id: "gpt-user-a", name: "gpt-user-a", provider: "openai" }])
    getChatModels.mockResolvedValueOnce([{ id: "gpt-user-b", name: "gpt-user-b", provider: "openai" }])

    const first = await load()
    const second = await load()

    expect([first.map((o) => o.value), second.map((o) => o.value)]).toEqual([
      ["gpt-user-a"],
      ["gpt-user-b"]
    ])
  })
})
