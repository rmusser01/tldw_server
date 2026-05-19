// @vitest-environment jsdom

import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import React from "react"
import type { DatasetSample } from "@/services/evaluations"
import { RagRetrievalTuningConfig } from "../recipe-configs/RagRetrievalTuningConfig"

const generateSyntheticDraftsSpy = vi.fn()

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

vi.mock("../../hooks/useSyntheticEval", () => ({
  useGenerateSyntheticEvalDrafts: () => ({
    mutateAsync: generateSyntheticDraftsSpy,
    isPending: false
  })
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

const DEFAULT_WEAK_SUPERVISION_BUDGET = {
  review_sample_fraction: 0.2,
  max_review_samples: 25,
  min_review_samples: 3,
  synthetic_query_limit: 20
}

describe("RagRetrievalTuningConfig", () => {
  beforeEach(() => {
    generateSyntheticDraftsSpy.mockReset()
    generateSyntheticDraftsSpy.mockResolvedValue({
      ok: true,
      data: {
        generation_batch_id: "batch-123",
        samples: [{ sample_id: "draft-1" }, { sample_id: "draft-2" }]
      }
    })
  })

  it("serializes labeled targets from guided dataset fields", () => {
    let datasetState: DatasetSample[] = []

    const Harness = () => {
      const [dataset, setDataset] = React.useState<DatasetSample[]>([
        {
          sample_id: "sample-1",
          query: ""
        } as DatasetSample
      ])
      const [runConfig, setRunConfig] = React.useState<Record<string, any>>({
        candidate_creation_mode: "auto_sweep",
        corpus_scope: { sources: ["media_db"] },
        weak_supervision_budget: DEFAULT_WEAK_SUPERVISION_BUDGET
      })

      React.useEffect(() => {
        datasetState = dataset
      }, [dataset])

      return (
        <RagRetrievalTuningConfig
          datasetSource="inline"
          dataset={dataset}
          runConfig={runConfig}
          onDatasetChange={setDataset}
          onRunConfigChange={setRunConfig}
        />
      )
    }

    render(<Harness />)

    fireEvent.change(screen.getByLabelText("Query text 1"), {
      target: { value: "find the architecture summary" }
    })
    fireEvent.change(screen.getByLabelText("Relevant media targets 1 (optional)"), {
      target: { value: "10:3\n12:1" }
    })
    fireEvent.change(screen.getByLabelText("Relevant note targets 1 (optional)"), {
      target: { value: "note-7:2" }
    })
    fireEvent.change(screen.getByLabelText("Relevant spans 1 (optional)"), {
      target: { value: "media_db,10,0,42,3" }
    })

    expect(datasetState).toEqual([
      {
        sample_id: "sample-1",
        query: "find the architecture summary",
        targets: {
          relevant_media_ids: [
            { id: "10", grade: 3 },
            { id: "12", grade: 1 }
          ],
          relevant_note_ids: [{ id: "note-7", grade: 2 }],
          relevant_spans: [
            {
              source: "media_db",
              record_id: "10",
              start: 0,
              end: 42,
              grade: 3
            }
          ]
        }
      }
    ])
  })

  it("disables synthetic generation until a retrieval corpus scope is selected", () => {
    const Harness = () => {
      const [dataset, setDataset] = React.useState<DatasetSample[]>([])
      const [runConfig, setRunConfig] = React.useState<Record<string, any>>({
        candidate_creation_mode: "auto_sweep",
        corpus_scope: { sources: ["media_db"], media_ids: [], note_ids: [] },
        weak_supervision_budget: DEFAULT_WEAK_SUPERVISION_BUDGET
      })

      return (
        <RagRetrievalTuningConfig
          datasetSource="inline"
          dataset={dataset}
          runConfig={runConfig}
          onDatasetChange={setDataset}
          onRunConfigChange={setRunConfig}
        />
      )
    }

    render(<Harness />)

    expect(
      screen.getByRole("button", { name: "Generate synthetic drafts" })
    ).toBeDisabled()

    fireEvent.change(screen.getByLabelText("Media IDs"), {
      target: { value: "10" }
    })

    expect(
      screen.getByRole("button", { name: "Generate synthetic drafts" })
    ).toBeEnabled()
  })

  it("submits structured retrieval examples and corpus scope to synthetic generation", async () => {
    const Harness = () => {
      const [dataset, setDataset] = React.useState<DatasetSample[]>([])
      const [runConfig, setRunConfig] = React.useState<Record<string, any>>({
        candidate_creation_mode: "auto_sweep",
        corpus_scope: {
          sources: ["media_db", "notes"],
          media_ids: [10],
          note_ids: ["note-7"],
          indexing_fixed: true
        },
        weak_supervision_budget: DEFAULT_WEAK_SUPERVISION_BUDGET
      })

      return (
        <RagRetrievalTuningConfig
          datasetSource="inline"
          dataset={dataset}
          runConfig={runConfig}
          onDatasetChange={setDataset}
          onRunConfigChange={setRunConfig}
        />
      )
    }

    render(<Harness />)

    fireEvent.change(screen.getByLabelText("Total draft set size"), {
      target: { value: "8" }
    })
    fireEvent.change(screen.getByLabelText("Real example 1 query"), {
      target: { value: "Which source explains the indexing plan?" }
    })
    fireEvent.change(screen.getByLabelText("Real example 1 source"), {
      target: { value: "media_db" }
    })
    fireEvent.change(screen.getByLabelText("Real example 1 query intent"), {
      target: { value: "lookup" }
    })
    fireEvent.change(screen.getByLabelText("Real example 1 difficulty"), {
      target: { value: "straightforward" }
    })
    fireEvent.change(screen.getByLabelText("Real example 1 relevant media ID"), {
      target: { value: "10" }
    })
    fireEvent.change(screen.getByLabelText("Real example 1 media grade"), {
      target: { value: "3" }
    })
    fireEvent.change(screen.getByLabelText("Real example 1 relevant note ID"), {
      target: { value: "note-7" }
    })
    fireEvent.change(screen.getByLabelText("Real example 1 note grade"), {
      target: { value: "1" }
    })
    fireEvent.change(screen.getByLabelText("Seed example 1 query"), {
      target: { value: "Compare the note guidance with the media transcript." }
    })
    fireEvent.change(screen.getByLabelText("Seed example 1 source"), {
      target: { value: "notes" }
    })
    fireEvent.change(screen.getByLabelText("Seed example 1 query intent"), {
      target: { value: "comparison" }
    })
    fireEvent.change(screen.getByLabelText("Seed example 1 difficulty"), {
      target: { value: "multi-source" }
    })

    fireEvent.click(
      screen.getByRole("button", { name: "Generate synthetic drafts" })
    )

    await waitFor(() => {
      expect(generateSyntheticDraftsSpy).toHaveBeenCalledWith({
        recipe_kind: "rag_retrieval_tuning",
        corpus_scope: {
          sources: ["media_db", "notes"],
          media_ids: [10],
          note_ids: ["note-7"],
          indexing_fixed: true
        },
        target_sample_count: 8,
        real_examples: [
          {
            source_kind: "media_db",
            query_intent: "lookup",
            difficulty: "straightforward",
            sample_payload: {
              query: "Which source explains the indexing plan?",
              relevant_media_ids: [{ id: "10", grade: 3 }],
              relevant_note_ids: [{ id: "note-7", grade: 1 }]
            }
          }
        ],
        seed_examples: [
          {
            source_kind: "notes",
            query_intent: "comparison",
            difficulty: "multi-source",
            sample_payload: {
              query: "Compare the note guidance with the media transcript."
            }
          }
        ]
      })
    })
  })

  it("shows an inline success summary without mutating the dataset", async () => {
    let datasetState: DatasetSample[] = []

    const Harness = () => {
      const [dataset, setDataset] = React.useState<DatasetSample[]>([
        {
          sample_id: "sample-1",
          query: "find the architecture summary"
        } as DatasetSample
      ])
      const [runConfig, setRunConfig] = React.useState<Record<string, any>>({
        candidate_creation_mode: "auto_sweep",
        corpus_scope: { sources: ["media_db"], media_ids: [10], note_ids: [] },
        weak_supervision_budget: DEFAULT_WEAK_SUPERVISION_BUDGET
      })

      React.useEffect(() => {
        datasetState = dataset
      }, [dataset])

      return (
        <RagRetrievalTuningConfig
          datasetSource="saved"
          dataset={dataset}
          runConfig={runConfig}
          onDatasetChange={setDataset}
          onRunConfigChange={setRunConfig}
        />
      )
    }

    render(<Harness />)

    fireEvent.change(screen.getByLabelText("Total draft set size"), {
      target: { value: "12" }
    })
    fireEvent.click(
      screen.getByRole("button", { name: "Generate synthetic drafts" })
    )

    expect(
      await screen.findByText("Batch batch-123 created 2 drafts and is ready for review.")
    ).toBeInTheDocument()
    expect(datasetState).toEqual([
      {
        sample_id: "sample-1",
        query: "find the architecture summary"
      }
    ])
  })

  it("shows an inline error state when synthetic generation fails", async () => {
    generateSyntheticDraftsSpy.mockRejectedValueOnce(new Error("generation failed"))

    const Harness = () => {
      const [dataset, setDataset] = React.useState<DatasetSample[]>([])
      const [runConfig, setRunConfig] = React.useState<Record<string, any>>({
        candidate_creation_mode: "auto_sweep",
        corpus_scope: { sources: ["media_db"], media_ids: [10], note_ids: [] },
        weak_supervision_budget: DEFAULT_WEAK_SUPERVISION_BUDGET
      })

      return (
        <RagRetrievalTuningConfig
          datasetSource="inline"
          dataset={dataset}
          runConfig={runConfig}
          onDatasetChange={setDataset}
          onRunConfigChange={setRunConfig}
        />
      )
    }

    render(<Harness />)

    fireEvent.click(
      screen.getByRole("button", { name: "Generate synthetic drafts" })
    )

    expect(
      await screen.findByText("generation failed")
    ).toBeInTheDocument()
  })

  it("uses saved dataset samples as preferred real examples for synthetic generation", async () => {
    const Harness = () => {
      const [dataset, setDataset] = React.useState<DatasetSample[]>([
        {
          sample_id: "saved-1",
          query: "How is the release checklist described?",
          targets: {
            relevant_media_ids: [{ id: "10", grade: 2 }]
          }
        } as DatasetSample
      ])
      const [runConfig, setRunConfig] = React.useState<Record<string, any>>({
        candidate_creation_mode: "auto_sweep",
        corpus_scope: {
          sources: ["media_db"],
          media_ids: [10],
          note_ids: [],
          indexing_fixed: false
        },
        weak_supervision_budget: DEFAULT_WEAK_SUPERVISION_BUDGET
      })

      return (
        <RagRetrievalTuningConfig
          datasetSource="saved"
          dataset={dataset}
          runConfig={runConfig}
          onDatasetChange={setDataset}
          onRunConfigChange={setRunConfig}
        />
      )
    }

    render(<Harness />)

    fireEvent.change(screen.getByLabelText("Total draft set size"), {
      target: { value: "4" }
    })
    fireEvent.click(
      screen.getByRole("button", { name: "Generate synthetic drafts" })
    )

    await waitFor(() => {
      expect(generateSyntheticDraftsSpy).toHaveBeenCalledWith({
        recipe_kind: "rag_retrieval_tuning",
        corpus_scope: {
          sources: ["media_db"],
          media_ids: [10],
          note_ids: [],
          indexing_fixed: false
        },
        target_sample_count: 4,
        real_examples: [
          {
            sample_id: "saved-1",
            source_kind: "media_db",
            query_intent: "lookup",
            difficulty: "straightforward",
            sample_payload: {
              query: "How is the release checklist described?",
              relevant_media_ids: [{ id: "10", grade: 2 }]
            }
          }
        ],
        seed_examples: []
      })
    })
  })

  it("switches to manual candidates and builds candidate rows", () => {
    let runConfigState: Record<string, any> = {}

    const Harness = () => {
      const [dataset, setDataset] = React.useState<DatasetSample[]>([])
      const [runConfig, setRunConfig] = React.useState<Record<string, any>>({
        candidate_creation_mode: "auto_sweep",
        corpus_scope: { sources: ["media_db", "notes"] },
        weak_supervision_budget: DEFAULT_WEAK_SUPERVISION_BUDGET
      })

      React.useEffect(() => {
        runConfigState = runConfig
      }, [runConfig])

      return (
        <RagRetrievalTuningConfig
          datasetSource="saved"
          dataset={dataset}
          runConfig={runConfig}
          onDatasetChange={setDataset}
          onRunConfigChange={setRunConfig}
        />
      )
    }

    render(<Harness />)

    fireEvent.change(screen.getByLabelText("Candidate creation mode"), {
      target: { value: "manual" }
    })
    fireEvent.change(screen.getByLabelText("Candidate ID 1"), {
      target: { value: "manual-a" }
    })
    fireEvent.change(screen.getByLabelText("Search mode 1"), {
      target: { value: "vector" }
    })
    fireEvent.change(screen.getByLabelText("Top K 1"), {
      target: { value: "8" }
    })
    fireEvent.change(screen.getByLabelText("Chunking preset 1"), {
      target: { value: "fixed_index" }
    })

    expect(runConfigState).toMatchObject({
      candidate_creation_mode: "manual",
      candidates: [
        {
          candidate_id: "manual-a",
          retrieval_config: expect.objectContaining({
            search_mode: "vector",
            top_k: 8
          }),
          indexing_config: expect.objectContaining({
            chunking_preset: "fixed_index"
          })
        }
      ]
    })
  })

  it("allows clearing numeric retrieval inputs without snapping back to defaults", () => {
    let runConfigState: Record<string, any> = {}

    const Harness = () => {
      const [dataset, setDataset] = React.useState<DatasetSample[]>([])
      const [runConfig, setRunConfig] = React.useState<Record<string, any>>({
        candidate_creation_mode: "auto_sweep",
        corpus_scope: { sources: ["media_db"] },
        retrieval_config: {
          search_mode: "hybrid",
          top_k: 10,
          hybrid_alpha: 0.7,
          enable_reranking: true,
          reranking_strategy: "flashrank",
          rerank_top_k: 10
        },
        indexing_config: {
          chunking_preset: "baseline"
        },
        weak_supervision_budget: DEFAULT_WEAK_SUPERVISION_BUDGET
      })

      React.useEffect(() => {
        runConfigState = runConfig
      }, [runConfig])

      return (
        <RagRetrievalTuningConfig
          datasetSource="saved"
          dataset={dataset}
          runConfig={runConfig}
          onDatasetChange={setDataset}
          onRunConfigChange={setRunConfig}
        />
      )
    }

    render(<Harness />)

    const topKInput = screen.getByLabelText("Top K") as HTMLInputElement

    fireEvent.change(topKInput, {
      target: { value: "" }
    })

    expect(topKInput).toHaveValue("")

    fireEvent.change(topKInput, {
      target: { value: "12" }
    })

    expect(runConfigState.retrieval_config.top_k).toBe(12)
  })
})
