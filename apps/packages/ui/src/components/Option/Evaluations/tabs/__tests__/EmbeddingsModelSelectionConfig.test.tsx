// @vitest-environment jsdom

import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import React from "react"
import type { DatasetSample, RecipeManifest } from "@/services/evaluations"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { EmbeddingsModelSelectionConfig } from "../recipe-configs/EmbeddingsModelSelectionConfig"

const useEmbeddingRecipeCandidatesSpy = vi.fn()

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
            [key: string]: unknown
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) {
        return defaultValueOrOptions.defaultValue.replace(
          /\{\{(\w+)\}\}/g,
          (_match, key) => String(defaultValueOrOptions[key] ?? "")
        )
      }
      return _key
    }
  })
}))

vi.mock("../../hooks/useRecipes", () => ({
  useEmbeddingRecipeCandidates: (enabled: boolean) =>
    useEmbeddingRecipeCandidatesSpy(enabled)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    searchMedia: vi.fn()
  }
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

const manifest = {
  recipe_id: "embeddings_model_selection",
  recipe_version: "1",
  name: "Embeddings Model Selection",
  description: "Pick an embedding model for RAG",
  launchable: true,
  supported_modes: ["labeled", "unlabeled"],
  tags: ["rag", "embeddings"],
  capabilities: {
    source_labeling: {
      source_id_contract: { kind: "media_id", type: "integer" }
    }
  },
  default_run_config: {
    comparison_mode: "embedding_only",
    top_k: 10,
    hybrid_alpha: 0.7
  }
} as RecipeManifest

describe("EmbeddingsModelSelectionConfig", () => {
  beforeEach(() => {
    vi.mocked(tldwClient.searchMedia).mockReset()
    useEmbeddingRecipeCandidatesSpy.mockReset()
    useEmbeddingRecipeCandidatesSpy.mockReturnValue({
      data: {
        ok: true,
        data: {
          candidates: []
        }
      },
      isLoading: false
    })
  })

  it("serializes query rows and selected media ids into recipe payload shape", () => {
    const onDatasetChange = vi.fn()
    const onRunConfigChange = vi.fn()

    const Harness = () => {
      const [dataset, setDataset] = React.useState<DatasetSample[]>([
        { query_id: "q-1", input: "", expected_ids: [] } as DatasetSample
      ])
      const [runConfig, setRunConfig] = React.useState<Record<string, any>>({
        comparison_mode: "embedding_only",
        candidates: []
      })

      const handleDatasetChange = (next: DatasetSample[]) => {
        onDatasetChange(next)
        setDataset(next)
      }
      const handleRunConfigChange = (next: Record<string, any>) => {
        onRunConfigChange(next)
        setRunConfig(next)
      }

      return (
        <EmbeddingsModelSelectionConfig
          datasetSource="inline"
          dataset={dataset}
          runConfig={runConfig}
          manifest={manifest}
          onDatasetChange={handleDatasetChange}
          onRunConfigChange={handleRunConfigChange}
        />
      )
    }

    render(<Harness />)

    fireEvent.change(screen.getByLabelText("Query text 1"), {
      target: { value: "find the beta launch notes" }
    })
    fireEvent.change(screen.getByLabelText("Expected media IDs 1"), {
      target: { value: "7, 9" }
    })

    expect(onDatasetChange).toHaveBeenLastCalledWith([
      {
        query_id: "q-1",
        input: "find the beta launch notes",
        expected_ids: ["7", "9"]
      }
    ])
  })

  it("serializes the source id contract as the backend string when run config changes", () => {
    const onRunConfigChange = vi.fn()

    const Harness = () => {
      const [dataset, setDataset] = React.useState<DatasetSample[]>([])
      const [runConfig, setRunConfig] = React.useState<Record<string, any>>({
        comparison_mode: "embedding_only",
        candidates: [],
        top_k: 10
      })

      const handleRunConfigChange = (next: Record<string, any>) => {
        onRunConfigChange(next)
        setRunConfig(next)
      }

      return (
        <EmbeddingsModelSelectionConfig
          datasetSource="inline"
          dataset={dataset}
          runConfig={runConfig}
          manifest={manifest}
          onDatasetChange={setDataset}
          onRunConfigChange={handleRunConfigChange}
        />
      )
    }

    render(<Harness />)

    fireEvent.change(screen.getByLabelText("Top K"), {
      target: { value: "12" }
    })

    expect(onRunConfigChange).toHaveBeenLastCalledWith(
      expect.objectContaining({
        top_k: 12,
        source_id_contract: "media_id"
      })
    )
  })

  it("lets ready candidates be selected while disallowed candidates remain status-only", async () => {
    let runConfigState: Record<string, any> = {}

    useEmbeddingRecipeCandidatesSpy.mockReturnValue({
      data: {
        ok: true,
        data: {
          candidates: [
            {
              provider: "openai",
              model: "text-embedding-3-small",
              status: "ready",
              is_local: false,
              default: true
            },
            {
              provider: "anthropic",
              model: "not-embedding",
              status: "disallowed_provider",
              status_reason: "Provider does not expose embeddings for this recipe",
              is_local: false,
              default: false
            }
          ]
        }
      },
      isLoading: false
    })

    const Harness = () => {
      const [dataset, setDataset] = React.useState<DatasetSample[]>([])
      const [runConfig, setRunConfig] = React.useState<Record<string, any>>({
        comparison_mode: "embedding_only",
        candidates: [{ provider: "local", model: "existing-embedding" }]
      })

      React.useEffect(() => {
        runConfigState = runConfig
      }, [runConfig])

      return (
        <EmbeddingsModelSelectionConfig
          datasetSource="inline"
          dataset={dataset}
          runConfig={runConfig}
          manifest={manifest}
          onDatasetChange={setDataset}
          onRunConfigChange={setRunConfig}
        />
      )
    }

    render(<Harness />)

    expect(screen.getByText("disallowed_provider")).toBeInTheDocument()
    expect(
      screen.getByText("Provider does not expose embeddings for this recipe")
    ).toBeInTheDocument()

    fireEvent.click(
      screen.getByRole("checkbox", { name: /openai text-embedding-3-small/i })
    )

    await waitFor(() =>
      expect(runConfigState.candidates).toEqual([
        { provider: "local", model: "existing-embedding" },
        { provider: "openai", model: "text-embedding-3-small" }
      ])
    )
    expect(runConfigState.candidates).not.toContainEqual(
      expect.objectContaining({ provider: "anthropic" })
    )
  })

  it("stores only integer media id strings from media search source selections", async () => {
    let datasetState: DatasetSample[] = []

    vi.mocked(tldwClient.searchMedia).mockResolvedValue({
      media: [
        { id: 42, title: "Launch notes", url: "file.md" },
        { id: "chunk-10", title: "Chunk result", url: "chunk.md" },
        { note_id: "note-9", title: "Note result", url: "note.md" }
      ]
    })

    const Harness = () => {
      const [dataset, setDataset] = React.useState<DatasetSample[]>([
        { query_id: "q-1", input: "launch", expected_ids: [] } as DatasetSample
      ])
      const [runConfig, setRunConfig] = React.useState<Record<string, any>>({
        comparison_mode: "embedding_only",
        candidates: []
      })

      React.useEffect(() => {
        datasetState = dataset
      }, [dataset])

      return (
        <EmbeddingsModelSelectionConfig
          datasetSource="inline"
          dataset={dataset}
          runConfig={runConfig}
          manifest={manifest}
          onDatasetChange={setDataset}
          onRunConfigChange={setRunConfig}
        />
      )
    }

    render(<Harness />)

    fireEvent.change(screen.getByLabelText("Find expected sources for query 1"), {
      target: { value: "launch" }
    })

    await waitFor(() =>
      expect(tldwClient.searchMedia).toHaveBeenCalledWith(
        { query: "launch" },
        { page: 1, results_per_page: 8 }
      )
    )

    fireEvent.click(await screen.findByRole("checkbox", { name: /Launch notes/i }))

    expect(datasetState).toEqual([
      expect.objectContaining({ expected_ids: ["42"] })
    ])
  })

  it("renders media search failures with the design-system Alert", async () => {
    vi.mocked(tldwClient.searchMedia).mockRejectedValueOnce(
      new Error("Media search failed.")
    )

    render(
      <EmbeddingsModelSelectionConfig
        datasetSource="inline"
        dataset={[{ query_id: "q-1", input: "query", expected_ids: [] } as DatasetSample]}
        runConfig={{ comparison_mode: "embedding_only", candidates: [] }}
        manifest={manifest}
        onDatasetChange={vi.fn()}
        onRunConfigChange={vi.fn()}
      />
    )

    fireEvent.change(screen.getByLabelText("Find expected sources for query 1"), {
      target: { value: "launch notes" }
    })

    await waitFor(() =>
      expect(screen.getByText("Media search failed.")).toBeInTheDocument()
    )
    expect(
      screen
        .getByText("Media search failed.")
        .closest('[data-ds-component="Alert"]')
    ).toBeInTheDocument()
  })
})
