// @vitest-environment jsdom

import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import React from "react"
import type { DatasetSample } from "@/services/evaluations"
import { RagAnswerQualityConfig } from "../recipe-configs/RagAnswerQualityConfig"

const generateSyntheticDraftsSpy = vi.fn()
const DEFAULT_CANDIDATE = {
  candidate_id: "candidate-1",
  generation_model: "openai:gpt-4.1-mini",
  prompt_variant: "default",
  formatting_citation_mode: "citations_required"
}

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

const expectDesignSystemAlert = (text: string | RegExp) => {
  const node = screen.getByText(text)
  expect(node.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
}

describe("RagAnswerQualityConfig", () => {
  beforeEach(() => {
    generateSyntheticDraftsSpy.mockReset()
    generateSyntheticDraftsSpy.mockResolvedValue({
      ok: true,
      data: {
        generation_batch_id: "batch-answer-123",
        samples: [{ sample_id: "draft-answer-1" }, { sample_id: "draft-answer-2" }]
      }
    })
  })

  it("serializes fixed-context dataset samples and anchor selection from guided fields", () => {
    let datasetState: DatasetSample[] = []
    let runConfigState: Record<string, any> = {}

    const Harness = () => {
      const [dataset, setDataset] = React.useState<DatasetSample[]>([
        {
          sample_id: "sample-1",
          query: "",
          expected_behavior: "answer"
        } as DatasetSample
      ])
      const [runConfig, setRunConfig] = React.useState<Record<string, any>>({
        evaluation_mode: "fixed_context",
        supervision_mode: "rubric",
        context_snapshot_ref: "ctx-initial",
        candidates: [
          {
            candidate_id: "candidate-1",
            generation_model: "openai:gpt-4.1-mini",
            prompt_variant: "default",
            formatting_citation_mode: "citations_required"
          }
        ]
      })

      React.useEffect(() => {
        datasetState = dataset
      }, [dataset])

      React.useEffect(() => {
        runConfigState = runConfig
      }, [runConfig])

      return (
        <RagAnswerQualityConfig
          datasetSource="inline"
          dataset={dataset}
          runConfig={runConfig}
          onDatasetChange={setDataset}
          onRunConfigChange={setRunConfig}
        />
      )
    }

    render(<Harness />)

    fireEvent.change(screen.getByLabelText("Context snapshot reference"), {
      target: { value: "ctx-release-2026-03-29" }
    })
    fireEvent.change(screen.getByLabelText("Query text 1"), {
      target: { value: "What changed in the rollout?" }
    })
    fireEvent.change(screen.getByLabelText("Expected behavior 1"), {
      target: { value: "hedge" }
    })
    fireEvent.change(screen.getByLabelText("Retrieved contexts 1"), {
      target: {
        value: "Doc A :: Rollout completed on Friday.\nDoc B :: Beta remained invite-only."
      }
    })
    fireEvent.change(screen.getByLabelText("Reference answer 1 (optional)"), {
      target: {
        value: "The rollout finished on Friday, but beta access remained limited."
      }
    })

    expect(runConfigState).toMatchObject({
      evaluation_mode: "fixed_context",
      supervision_mode: "rubric",
      context_snapshot_ref: "ctx-release-2026-03-29",
      candidates: [
        expect.objectContaining({
          candidate_id: "candidate-1",
          generation_model: "openai:gpt-4.1-mini",
          prompt_variant: "default",
          formatting_citation_mode: "citations_required"
        })
      ]
    })

    expect(datasetState).toEqual([
      {
        sample_id: "sample-1",
        query: "What changed in the rollout?",
        expected_behavior: "hedge",
        retrieved_contexts: [
          { content: "Doc A :: Rollout completed on Friday." },
          { content: "Doc B :: Beta remained invite-only." }
        ],
        reference_answer:
          "The rollout finished on Friday, but beta access remained limited."
      }
    ])
  })

  it("switches to live mode and preserves bounded candidate controls", () => {
    let runConfigState: Record<string, any> = {}

    const Harness = () => {
      const [dataset, setDataset] = React.useState<DatasetSample[]>([])
      const [runConfig, setRunConfig] = React.useState<Record<string, any>>({
        evaluation_mode: "fixed_context",
        supervision_mode: "rubric",
        context_snapshot_ref: "ctx-1",
        candidates: [
          {
            candidate_id: "candidate-1",
            generation_model: "openai:gpt-4.1-mini",
            prompt_variant: "default",
            formatting_citation_mode: "citations_required"
          }
        ]
      })

      React.useEffect(() => {
        runConfigState = runConfig
      }, [runConfig])

      return (
        <RagAnswerQualityConfig
          datasetSource="saved"
          dataset={dataset}
          runConfig={runConfig}
          onDatasetChange={setDataset}
          onRunConfigChange={setRunConfig}
        />
      )
    }

    render(<Harness />)

    fireEvent.change(screen.getByLabelText("Evaluation mode"), {
      target: { value: "live_end_to_end" }
    })
    fireEvent.change(screen.getByLabelText("Retrieval baseline reference"), {
      target: { value: "baseline-rag-v2" }
    })
    fireEvent.change(screen.getByLabelText("Supervision mode"), {
      target: { value: "mixed" }
    })
    fireEvent.change(screen.getByLabelText("Candidate ID 1"), {
      target: { value: "live-best" }
    })
    fireEvent.change(screen.getByLabelText("Generation model 1"), {
      target: { value: "local:qwen3-14b" }
    })
    fireEvent.change(screen.getByLabelText("Prompt variant 1"), {
      target: { value: "citation-heavy" }
    })
    fireEvent.change(screen.getByLabelText("Formatting/citation mode 1"), {
      target: { value: "markdown_citations" }
    })

    expect(runConfigState).toMatchObject({
      evaluation_mode: "live_end_to_end",
      supervision_mode: "mixed",
      retrieval_baseline_ref: "baseline-rag-v2",
      candidates: [
        {
          candidate_id: "live-best",
          generation_model: "local:qwen3-14b",
          prompt_variant: "citation-heavy",
          formatting_citation_mode: "markdown_citations"
        }
      ]
    })
    expect(runConfigState.context_snapshot_ref).toBeUndefined()
  })

  it("requires a context snapshot or retrieval baseline before generation", () => {
    const Harness = () => {
      const [dataset, setDataset] = React.useState<DatasetSample[]>([])
      const [runConfig, setRunConfig] = React.useState<Record<string, any>>({
        evaluation_mode: "fixed_context",
        supervision_mode: "rubric",
        context_snapshot_ref: "",
        candidates: [{ ...DEFAULT_CANDIDATE }]
      })

      return (
        <RagAnswerQualityConfig
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

    fireEvent.change(screen.getByLabelText("Context snapshot reference"), {
      target: { value: "ctx-release-2026-03-29" }
    })

    expect(
      screen.getByRole("button", { name: "Generate synthetic drafts" })
    ).toBeEnabled()

    fireEvent.change(screen.getByLabelText("Evaluation mode"), {
      target: { value: "live_end_to_end" }
    })

    expect(
      screen.getByRole("button", { name: "Generate synthetic drafts" })
    ).toBeDisabled()

    fireEvent.change(screen.getByLabelText("Retrieval baseline reference"), {
      target: { value: "baseline-rag-v2" }
    })

    expect(
      screen.getByRole("button", { name: "Generate synthetic drafts" })
    ).toBeEnabled()
  })

  it("submits answer-quality generation with expected behavior hints and anchor metadata", async () => {
    const Harness = () => {
      const [dataset, setDataset] = React.useState<DatasetSample[]>([])
      const [runConfig, setRunConfig] = React.useState<Record<string, any>>({
        evaluation_mode: "fixed_context",
        supervision_mode: "rubric",
        context_snapshot_ref: "ctx-release-2026-03-29",
        candidates: [{ ...DEFAULT_CANDIDATE }]
      })

      return (
        <RagAnswerQualityConfig
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
      target: { value: "12" }
    })
    fireEvent.change(screen.getByLabelText("Seed example 1 query"), {
      target: { value: "What changed in the rollout?" }
    })
    fireEvent.change(screen.getByLabelText("Seed example 1 expected behavior"), {
      target: { value: "hedge" }
    })
    fireEvent.change(screen.getByLabelText("Seed example 1 reference answer"), {
      target: {
        value: "The rollout completed on Friday, but beta access remained limited."
      }
    })
    fireEvent.change(screen.getByLabelText("Seed example 1 retrieved contexts"), {
      target: {
        value: "Doc A :: Rollout completed on Friday.\nDoc B :: Beta remained invite-only."
      }
    })

    fireEvent.click(
      screen.getByRole("button", { name: "Generate synthetic drafts" })
    )

    await waitFor(() => {
      expect(generateSyntheticDraftsSpy).toHaveBeenCalledWith({
        recipe_kind: "rag_answer_quality",
        context_snapshot_ref: "ctx-release-2026-03-29",
        target_sample_count: 12,
        real_examples: [],
        seed_examples: [
          {
            expected_behavior: "hedge",
            reference_answer:
              "The rollout completed on Friday, but beta access remained limited.",
            sample_payload: {
              query: "What changed in the rollout?",
              expected_behavior: "hedge",
              reference_answer:
                "The rollout completed on Friday, but beta access remained limited.",
              retrieved_contexts: [
                { content: "Doc A :: Rollout completed on Friday." },
                { content: "Doc B :: Beta remained invite-only." }
              ]
            }
          }
        ]
      })
    })
  })

  it("uses saved dataset samples as preferred real examples for answer-quality generation", async () => {
    const Harness = () => {
      const [dataset, setDataset] = React.useState<DatasetSample[]>([
        {
          sample_id: "saved-answer-1",
          query: "Should we say beta is open to everyone?",
          expected_behavior: "abstain",
          reference_answer: "No. The retrieved context does not support that claim."
        } as DatasetSample
      ])
      const [runConfig, setRunConfig] = React.useState<Record<string, any>>({
        evaluation_mode: "live_end_to_end",
        supervision_mode: "mixed",
        retrieval_baseline_ref: "baseline-rag-v2",
        candidates: [{ ...DEFAULT_CANDIDATE }]
      })

      return (
        <RagAnswerQualityConfig
          datasetSource="saved"
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

    await waitFor(() => {
      expect(generateSyntheticDraftsSpy).toHaveBeenCalledWith({
        recipe_kind: "rag_answer_quality",
        retrieval_baseline_ref: "baseline-rag-v2",
        target_sample_count: 12,
        real_examples: [
          {
            sample_id: "saved-answer-1",
            expected_behavior: "abstain",
            reference_answer: "No. The retrieved context does not support that claim.",
            sample_payload: {
              query: "Should we say beta is open to everyone?",
              expected_behavior: "abstain",
              reference_answer: "No. The retrieved context does not support that claim."
            }
          }
        ],
        seed_examples: []
      })
    })
  })

  it("shows inline success and error states for synthetic generation", async () => {
    const Harness = () => {
      const [dataset, setDataset] = React.useState<DatasetSample[]>([])
      const [runConfig, setRunConfig] = React.useState<Record<string, any>>({
        evaluation_mode: "fixed_context",
        supervision_mode: "rubric",
        context_snapshot_ref: "ctx-release-2026-03-29",
        candidates: [{ ...DEFAULT_CANDIDATE }]
      })

      return (
        <RagAnswerQualityConfig
          datasetSource="inline"
          dataset={dataset}
          runConfig={runConfig}
          onDatasetChange={setDataset}
          onRunConfigChange={setRunConfig}
        />
      )
    }

    render(<Harness />)

    expectDesignSystemAlert("Current answer-quality anchor")

    fireEvent.change(screen.getByLabelText("Seed example 1 query"), {
      target: { value: "What changed in the rollout?" }
    })
    fireEvent.change(screen.getByLabelText("Seed example 1 expected behavior"), {
      target: { value: "answer" }
    })
    fireEvent.change(screen.getByLabelText("Seed example 1 reference answer"), {
      target: { value: "The rollout completed on Friday." }
    })
    fireEvent.click(
      screen.getByRole("button", { name: "Generate synthetic drafts" })
    )

    expect(
      await screen.findByText(
        "Batch batch-answer-123 created 2 drafts and is ready for review."
      )
    ).toBeInTheDocument()
    expectDesignSystemAlert("Synthetic drafts created")

    generateSyntheticDraftsSpy.mockRejectedValueOnce(new Error("generation failed"))
    fireEvent.click(
      screen.getByRole("button", { name: "Generate synthetic drafts" })
    )

    expect(await screen.findByText("generation failed")).toBeInTheDocument()
    expectDesignSystemAlert("Failed to generate synthetic drafts")
  })
})
