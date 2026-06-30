// @vitest-environment jsdom

import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen } from "@testing-library/react"
import { SyntheticReviewTab } from "../SyntheticReviewTab"

const queueState = {
  data: {
    data: {
      data: [
        {
          sample_id: "draft-1",
          recipe_kind: "rag_retrieval_tuning",
          provenance: "synthetic_from_corpus",
          review_state: "draft",
          sample_payload: {
            query: "What changed in the rollout?",
            relevant_media_ids: [1]
          },
          sample_metadata: {}
        }
      ],
      total: 1
    }
  },
  isLoading: false,
  isError: false,
  error: null
}

const reviewMutateAsync = vi.fn()
const promoteMutateAsync = vi.fn()
const useSyntheticEvalQueueSpy = vi.fn((_params?: unknown) => queueState)
const storeState = {
  syntheticReviewRecipeKind: "rag_retrieval_tuning",
  syntheticReviewBatchId: "batch-123",
  syntheticReviewSampleIds: ["draft-1"],
  setSyntheticReviewRecipeKind: vi.fn()
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock("@/store/evaluations", () => ({
  useEvaluationsStore: (selector: (state: typeof storeState) => unknown) =>
    selector(storeState)
}))

vi.mock("../../hooks/useSyntheticEval", () => ({
  useSyntheticEvalQueue: (params: any) => useSyntheticEvalQueueSpy(params),
  useReviewSyntheticEvalSample: () => ({
    mutateAsync: reviewMutateAsync,
    isPending: false,
    data: null
  }),
  usePromoteSyntheticEvalSamples: () => ({
    mutateAsync: promoteMutateAsync,
    isPending: false,
    data: {
      data: {
        dataset_id: "dataset_123",
        sample_count: 1
      }
    }
  })
}))

vi.mock("antd", () => {
  const Button = ({ children, loading, danger, ...props }: any) => (
    <button
      data-loading={loading ? "true" : "false"}
      data-danger={danger ? "true" : "false"}
      {...props}
    >
      {children}
    </button>
  )
  const Select = ({ value, onChange, options = [], "aria-label": ariaLabel }: any) => (
    <select
      aria-label={ariaLabel}
      value={value}
      onChange={(event) => onChange?.(event.target.value)}
    >
      {options.map((option: any) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  )
  const Input = ({ value, onChange, "aria-label": ariaLabel, className }: any) => (
    <input
      aria-label={ariaLabel}
      className={className}
      value={value}
      onChange={onChange}
    />
  )
  Input.TextArea = ({ value, onChange, "aria-label": ariaLabel }: any) => (
    <textarea aria-label={ariaLabel} value={value} onChange={onChange} />
  )
  const Checkbox = ({ children, checked, onChange, "aria-label": ariaLabel }: any) => (
    <label>
      <input
        aria-label={ariaLabel}
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange?.({ target: { checked: event.target.checked } })}
      />
      {children}
    </label>
  )
  return {
    Alert: ({ title, description }: any) => (
      <div>
        <div>{title}</div>
        <div>{description}</div>
      </div>
    ),
    Button,
    Card: ({ title, children }: any) => (
      <section>
        <div>{title}</div>
        <div>{children}</div>
      </section>
    ),
    Checkbox,
    Empty: ({ description }: any) => <div>{description}</div>,
    Input,
    Select,
    Space: ({ children }: any) => <div>{children}</div>,
    Spin: () => <div>Loading</div>,
    Tag: ({ children }: any) => <span>{children}</span>,
    Typography: {
      Paragraph: ({ children, className }: any) => <p className={className}>{children}</p>,
      Text: ({ children }: any) => <span>{children}</span>,
      Title: ({ children }: any) => <h3>{children}</h3>
    }
  }
})

describe("SyntheticReviewTab", () => {
  beforeEach(() => {
    reviewMutateAsync.mockReset()
    promoteMutateAsync.mockReset()
    useSyntheticEvalQueueSpy.mockClear()
    storeState.setSyntheticReviewRecipeKind.mockReset()
    queueState.data = {
      data: {
        data: [
          {
            sample_id: "draft-1",
            recipe_kind: "rag_retrieval_tuning",
            provenance: "synthetic_from_corpus",
            review_state: "draft",
            sample_payload: {
              query: "What changed in the rollout?",
              relevant_media_ids: [1]
            },
            sample_metadata: {}
          }
        ],
        total: 1
      }
    }
    queueState.isLoading = false
    queueState.isError = false
    queueState.error = null
  })

  it("requests the queue filtered to the stored generation batch when present", () => {
    render(<SyntheticReviewTab />)

    expect(useSyntheticEvalQueueSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        recipeKind: "rag_retrieval_tuning",
        reviewState: "draft",
        generationBatchId: "batch-123"
      })
    )
  })

  it("renders the filtered queue and applies review actions", () => {
    render(<SyntheticReviewTab />)

    expect(screen.getByText("draft-1")).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText("Review notes draft-1"), {
      target: { value: "Looks grounded." }
    })
    fireEvent.click(screen.getByRole("button", { name: "Approve" }))

    expect(reviewMutateAsync).toHaveBeenCalledWith({
      sampleId: "draft-1",
      action: "approve",
      notes: "Looks grounded."
    })
  })

  it("updates the recipe filter and promotes selected samples", () => {
    render(<SyntheticReviewTab />)

    fireEvent.change(screen.getByLabelText("Synthetic review recipe filter"), {
      target: { value: "rag_answer_quality" }
    })
    expect(storeState.setSyntheticReviewRecipeKind).toHaveBeenCalledWith(
      "rag_answer_quality"
    )

    fireEvent.click(screen.getByLabelText("Select draft-1"))
    fireEvent.change(screen.getByLabelText("Synthetic promoted dataset name"), {
      target: { value: "reviewed synthetic retrieval" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Promote selected" }))

    expect(promoteMutateAsync).toHaveBeenCalledWith({
      sample_ids: ["draft-1"],
      dataset_name: "reviewed synthetic retrieval"
    })
  })

  it("shows synthetic queue endpoint diagnostics when the queue is unavailable", () => {
    queueState.data = undefined as any
    queueState.isError = true
    queueState.error = new Error("HTTP 503")

    render(<SyntheticReviewTab />)

    expect(screen.getByText("Unavailable")).toBeInTheDocument()
    expect(screen.getByText("Unable to load synthetic review queue")).toBeInTheDocument()
    expect(screen.getByLabelText("Diagnostics")).toHaveTextContent(
      "/api/v1/evaluations/synthetic/queue"
    )
  })
})
