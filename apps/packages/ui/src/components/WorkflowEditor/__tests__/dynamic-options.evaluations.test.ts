// @vitest-environment jsdom

import { beforeEach, describe, expect, it, vi } from "vitest"
import { renderHook, waitFor } from "@testing-library/react"
import {
  __resetWorkflowDynamicOptionsCacheForTests,
  useWorkflowDynamicOptions
} from "../dynamic-options"
import type { ConfigFieldSchema } from "@/types/workflow-editor"

const { listRunsSpy, listRunsGlobalSpy } = vi.hoisted(() => ({
  listRunsSpy: vi.fn(),
  listRunsGlobalSpy: vi.fn()
}))

vi.mock("@/services/evaluations", () => ({
  listDatasets: vi.fn(),
  listEvaluations: vi.fn(),
  listRuns: listRunsSpy,
  listRunsGlobal: listRunsGlobalSpy
}))

vi.mock("@/services/tldw", () => ({
  tldwClient: {
    getTranscriptionModels: vi.fn(),
    getProviders: vi.fn(),
    getPrompts: vi.fn(),
    getReadingList: vi.fn(),
    listOutputs: vi.fn()
  },
  tldwModels: {
    getChatModels: vi.fn(),
    getEmbeddingModels: vi.fn(),
    getImageModels: vi.fn()
  }
}))

vi.mock("@/services/tldw/audio-models", () => ({
  fetchTldwTtsModels: vi.fn()
}))

vi.mock("@/services/tldw/audio-providers", () => ({
  fetchTtsProviders: vi.fn()
}))

vi.mock("@/services/tldw/audio-voices", () => ({
  fetchTldwVoices: vi.fn(),
  fetchTldwVoiceCatalog: vi.fn()
}))

vi.mock("@/services/tldw/embedding-collections", () => ({
  fetchEmbeddingCollections: vi.fn()
}))

vi.mock("@/services/folder-api", () => ({
  fetchFolders: vi.fn()
}))

describe("useWorkflowDynamicOptions evaluation runs", () => {
  const fields: ConfigFieldSchema[] = [
    {
      key: "run_id",
      type: "select",
      label: "Run"
    }
  ]

  beforeEach(() => {
    vi.clearAllMocks()
    __resetWorkflowDynamicOptionsCacheForTests()
    listRunsSpy.mockResolvedValue({
      ok: true,
      data: {
        data: [{ id: "run-1" }]
      }
    })
    listRunsGlobalSpy.mockResolvedValue({
      ok: true,
      data: {
        data: [{ id: "run-1" }]
      }
    })
  })

  it("loads evaluation runs through the scoped eval endpoint", async () => {
    const { result } = renderHook(() =>
      useWorkflowDynamicOptions({
        fields,
        stepType: "evaluations",
        config: { evaluation_id: "eval-1" }
      })
    )

    await waitFor(() => {
      expect(result.current.optionsByKey.run_id).toEqual([
        { value: "run-1", label: "run-1" }
      ])
    })

    expect(listRunsSpy).toHaveBeenCalledWith("eval-1", { limit: 100 })
    expect(listRunsGlobalSpy).not.toHaveBeenCalled()
  })

  it("does not call a dead global runs endpoint when no evaluation id is present", async () => {
    const { result } = renderHook(() =>
      useWorkflowDynamicOptions({
        fields,
        stepType: "evaluations",
        config: {}
      })
    )

    await waitFor(() => {
      expect(result.current.loadingByKey.run_id).toBeFalsy()
    })

    expect(listRunsSpy).not.toHaveBeenCalled()
    expect(listRunsGlobalSpy).not.toHaveBeenCalled()
    expect(result.current.optionsByKey.run_id ?? []).toEqual([])
  })

  it("does not refetch already loaded sources on rerender with equivalent inputs", async () => {
    const { result, rerender } = renderHook(
      ({ config }) =>
        useWorkflowDynamicOptions({
          fields,
          stepType: "evaluations",
          config
        }),
      {
        initialProps: {
          config: { evaluation_id: "eval-1" }
        }
      }
    )

    await waitFor(() => {
      expect(result.current.optionsByKey.run_id).toEqual([
        { value: "run-1", label: "run-1" }
      ])
    })

    expect(listRunsSpy).toHaveBeenCalledTimes(1)

    rerender({
      config: { evaluation_id: "eval-1" }
    })

    await waitFor(() => {
      expect(result.current.optionsByKey.run_id).toEqual([
        { value: "run-1", label: "run-1" }
      ])
    })

    expect(listRunsSpy).toHaveBeenCalledTimes(1)
  })
})
