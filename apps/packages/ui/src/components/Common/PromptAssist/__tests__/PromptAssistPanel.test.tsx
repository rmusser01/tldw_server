import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { StrictMode, useRef, useState } from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { PromptAssistPanel } from "../PromptAssistPanel"
import type {
  PromptAssistAnalyzingState,
  PromptAssistAppliedState,
  PromptAssistFailedState,
  PromptAssistReviewingState
} from "../prompt-assist-state"
import { type PromptTargetAdapter, usePromptAssist } from "../usePromptAssist"

const promptMocks = vi.hoisted(() => ({
  improvePrompt: vi.fn()
}))

vi.mock("@/services/prompt-improvement", async (importOriginal) => {
  const original =
    await importOriginal<typeof import("@/services/prompt-improvement")>()
  return {
    ...original,
    improvePrompt: (...args: unknown[]) => promptMocks.improvePrompt(...args)
  }
})

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      defaultValue?: string,
      options?: Record<string, string | number>
    ) =>
      (defaultValue ?? _key).replace(/{{(\w+)}}/g, (_, key) =>
        String(options?.[key] ?? "")
      )
  })
}))

const operation = {
  operationId: "11111111-1111-4111-8111-111111111111",
  target: "system" as const,
  mode: "review_changes" as const,
  originalText: "System draft",
  revision: "r1",
  route: { selected_model: "auto" }
}

const response = {
  schema_version: 1 as const,
  operation_id: operation.operationId,
  status: "improved" as const,
  improved_text: "Improved system draft",
  findings: [],
  review_required: false,
  warnings: [],
  resolved_model: {
    provider: "openai",
    model: "gpt-5-mini",
    display_name: "GPT-5 mini"
  },
  meta_prompt_version: "prompt-improvement-v1"
}

const callbacks = () => ({
  onCancel: vi.fn(),
  onRetry: vi.fn(),
  onSelectModel: vi.fn(),
  onCandidateChange: vi.fn(),
  onApply: vi.fn(),
  onConfirmReplace: vi.fn(),
  onUndo: vi.fn(),
  onRequestReturnFocus: vi.fn()
})

type StatefulOwnerHarnessProps = {
  focusRequests: string[]
  unmountOnApply?: boolean
}

function StatefulOwnerHarness({
  focusRequests,
  unmountOnApply = false
}: StatefulOwnerHarnessProps) {
  const [open, setOpen] = useState(true)
  const [, forceRender] = useState(0)
  const textRef = useRef("System draft")
  const revisionRef = useRef("r1")
  const adapterRef = useRef<PromptTargetAdapter | null>(null)
  if (!adapterRef.current) {
    adapterRef.current = {
      target: "system",
      read: () => textRef.current,
      readRevision: () => revisionRef.current,
      apply: (candidate) => {
        textRef.current = candidate
        revisionRef.current = "r3"
        forceRender((version) => version + 1)
      },
      captureUndo: () => ({
        text: textRef.current,
        revision: revisionRef.current
      }),
      restoreUndo: () => undefined
    }
  }
  const assist = usePromptAssist({
    adapter: adapterRef.current,
    readActiveRoute: () => operation.route,
    contextKey: "conversation-1",
    surfaceOpen: open
  })
  const requestReturnFocus = () => {
    focusRequests.push(assist.state.status)
    setOpen(false)
    setTimeout(() =>
      document
        .querySelector<HTMLButtonElement>("#stateful-remounted-trigger")
        ?.focus()
    )
  }

  if (!open) {
    return <button id="stateful-remounted-trigger">Prompt actions</button>
  }

  return (
    <>
      <button type="button" onClick={() => void assist.reviewChanges()}>
        Start review
      </button>
      <button
        type="button"
        onClick={() => {
          textRef.current = "New live draft"
          revisionRef.current = "r2"
          forceRender((version) => version + 1)
        }}>
        Edit live draft
      </button>
      <PromptAssistPanel
        state={assist.state}
        onCancel={() => {
          assist.dismiss()
          setOpen(false)
        }}
        onRetry={assist.retry}
        onSelectModel={vi.fn()}
        onCandidateChange={assist.editCandidate}
        onApply={() => {
          assist.applyCandidate()
          if (unmountOnApply) setOpen(false)
        }}
        onConfirmReplace={assist.confirmReplaceCurrent}
        onUndo={assist.undo}
        onRequestReturnFocus={requestReturnFocus}
      />
    </>
  )
}

describe("PromptAssistPanel", () => {
  beforeEach(() => {
    promptMocks.improvePrompt.mockReset()
    promptMocks.improvePrompt.mockImplementation(async (request) => ({
      ...response,
      operation_id: request.operation_id
    }))
  })

  it("shows the captured draft while analyzing and keeps Cancel available", async () => {
    const user = userEvent.setup()
    const props = callbacks()
    const state: PromptAssistAnalyzingState = {
      status: "analyzing",
      operation,
      undo: null
    }
    render(<PromptAssistPanel state={state} {...props} />)

    expect(screen.getByRole("status")).toHaveTextContent("Analyzing with Auto")
    expect(screen.getByText("System draft")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Cancel" }))
    expect(props.onCancel).toHaveBeenCalledTimes(1)
  })

  it("handles local Escape without owning a focus trap and requests owner focus return", async () => {
    const user = userEvent.setup()
    const props = callbacks()
    const state: PromptAssistAnalyzingState = {
      status: "analyzing",
      operation,
      undo: null
    }
    render(<PromptAssistPanel state={state} {...props} />)

    screen.getByRole("region", { name: "Prompt improvement" }).focus()
    await user.keyboard("{Escape}")
    expect(props.onCancel).toHaveBeenCalledTimes(1)
    expect(props.onRequestReturnFocus).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("honors retry-after before enabling retry and resets the countdown for a new error", () => {
    vi.useFakeTimers()
    const props = callbacks()
    const state: PromptAssistFailedState = {
      status: "failed",
      operation,
      undo: null,
      error: {
        code: "provider_rate_limited",
        message: "Prompt improvement failed.",
        retryable: true,
        retryAfterSeconds: 2,
        requestId: "request-1"
      }
    }
    const { rerender } = render(<PromptAssistPanel state={state} {...props} />)

    const retry = screen.getByRole("button", { name: "Retry" })
    expect(retry).toBeDisabled()
    expect(screen.getByRole("status")).toHaveTextContent(
      "Retry available in 2 seconds."
    )
    fireEvent.click(retry)
    expect(props.onRetry).not.toHaveBeenCalled()

    act(() => vi.advanceTimersByTime(2_000))
    expect(retry).toBeEnabled()
    fireEvent.click(retry)
    expect(props.onRetry).toHaveBeenCalledTimes(1)

    rerender(
      <PromptAssistPanel
        state={{
          ...state,
          error: {
            ...state.error,
            retryAfterSeconds: 3,
            requestId: "request-2"
          }
        }}
        {...props}
      />
    )
    expect(screen.getByRole("button", { name: "Retry" })).toBeDisabled()
    expect(screen.getByRole("status")).toHaveTextContent(
      "Retry available in 3 seconds."
    )
    vi.useRealTimers()
  })

  it("renders model recovery for stable model failures", async () => {
    const user = userEvent.setup()
    const props = callbacks()
    const state: PromptAssistFailedState = {
      status: "failed",
      operation,
      undo: null,
      error: {
        code: "missing_model",
        message: "Prompt improvement failed.",
        retryable: false
      }
    }
    render(<PromptAssistPanel state={state} {...props} />)

    expect(screen.getByRole("alert")).toHaveTextContent(
      "Select a chat model and try again."
    )
    await user.click(screen.getByRole("button", { name: "Select model" }))
    expect(props.onSelectModel).toHaveBeenCalledTimes(1)
  })

  it("does not render a dead model-recovery control without an owner flow", () => {
    const props = callbacks()
    const state: PromptAssistFailedState = {
      status: "failed",
      operation,
      undo: null,
      error: {
        code: "missing_model",
        message: "Prompt improvement failed.",
        retryable: false
      }
    }

    render(
      <PromptAssistPanel {...props} state={state} onSelectModel={undefined} />
    )

    expect(screen.getByRole("alert")).toHaveTextContent(
      "Select a chat model and try again."
    )
    expect(
      screen.queryByRole("button", { name: "Select model" })
    ).not.toBeInTheDocument()
  })

  it("projects reviewing state into the editable review surface", () => {
    const props = callbacks()
    const state: PromptAssistReviewingState = {
      status: "reviewing",
      operation,
      response,
      candidate: "Edited candidate",
      notice: "review_required",
      replaceConfirmationRequired: false,
      undo: null
    }
    render(<PromptAssistPanel state={state} {...props} />)

    expect(
      screen.getByRole("textbox", { name: "Improved prompt candidate" })
    ).toHaveValue("Edited candidate")
    expect(
      screen.getByText("Review the safety notices before applying.")
    ).toBeInTheDocument()
  })

  it("announces Apply state, opens inspection, and announces Undo", async () => {
    const user = userEvent.setup()
    const props = callbacks()
    const state: PromptAssistAppliedState = {
      status: "applied",
      operation,
      response,
      candidate: "Improved system draft",
      undo: {
        target: "system",
        snapshot: { override: undefined },
        operationId: operation.operationId,
        candidate: "Improved system draft"
      }
    }
    render(<PromptAssistPanel state={state} {...props} />)

    expect(screen.getByRole("status")).toHaveTextContent("Improvement applied.")
    await user.click(screen.getByRole("button", { name: "View changes" }))
    expect(
      screen.queryByRole("button", { name: "Apply to draft" })
    ).not.toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Undo improvement" }))
    expect(props.onUndo).toHaveBeenCalledTimes(1)
    expect(screen.getByRole("status")).toHaveTextContent("Improvement undone.")
  })

  it("removes inspection Undo as soon as the state invalidates its snapshot", async () => {
    const user = userEvent.setup()
    const props = callbacks()
    const state: PromptAssistAppliedState = {
      status: "applied",
      operation,
      response,
      candidate: "Improved system draft",
      undo: {
        target: "system",
        snapshot: { override: undefined },
        operationId: operation.operationId,
        candidate: "Improved system draft"
      }
    }
    const { rerender } = render(<PromptAssistPanel state={state} {...props} />)
    await user.click(screen.getByRole("button", { name: "View changes" }))
    expect(
      screen.getByRole("button", { name: "Undo improvement" })
    ).toBeInTheDocument()

    rerender(<PromptAssistPanel state={{ ...state, undo: null }} {...props} />)
    expect(
      screen.queryByRole("button", { name: "Undo improvement" })
    ).not.toBeInTheDocument()
    expect(screen.getByRole("status")).not.toHaveTextContent(
      "Improvement undone."
    )
  })

  it("routes Escape through confirmation, inspection, and then the owning assist surface", async () => {
    const user = userEvent.setup()
    const props = callbacks()
    const reviewing: PromptAssistReviewingState = {
      status: "reviewing",
      operation,
      response,
      candidate: "Edited candidate",
      notice: "draft_changed",
      replaceConfirmationRequired: true,
      undo: null
    }
    const { rerender } = render(
      <PromptAssistPanel state={reviewing} {...props} />
    )
    await user.click(
      screen.getByRole("button", { name: "Replace current draft" })
    )
    await user.keyboard("{Escape}")
    expect(
      screen.queryByText("Replace the current draft with this candidate?")
    ).not.toBeInTheDocument()
    expect(props.onCancel).not.toHaveBeenCalled()
    await user.keyboard("{Escape}")
    expect(props.onCancel).toHaveBeenCalledTimes(1)

    const applied: PromptAssistAppliedState = {
      status: "applied",
      operation,
      response,
      candidate: "Improved system draft",
      undo: {
        target: "system",
        snapshot: {},
        operationId: operation.operationId,
        candidate: "Improved system draft"
      }
    }
    rerender(<PromptAssistPanel state={applied} {...props} />)
    await user.click(screen.getByRole("button", { name: "View changes" }))
    screen.getByRole("button", { name: "Close" }).focus()
    await user.keyboard("{Escape}")
    expect(
      screen.getByRole("button", { name: "View changes" })
    ).toBeInTheDocument()
    expect(props.onCancel).toHaveBeenCalledTimes(1)
  })

  it("keeps stale Apply in review and returns focus once after confirmed Replace", async () => {
    const user = userEvent.setup()
    const focusRequests: string[] = []
    render(
      <StrictMode>
        <StatefulOwnerHarness focusRequests={focusRequests} />
      </StrictMode>
    )

    await user.click(screen.getByRole("button", { name: "Start review" }))
    await screen.findByRole("button", { name: "Apply to draft" })
    await user.click(screen.getByRole("button", { name: "Edit live draft" }))
    await user.click(screen.getByRole("button", { name: "Apply to draft" }))

    expect(
      screen.getByRole("button", { name: "Replace current draft" })
    ).toBeInTheDocument()
    expect(focusRequests).toEqual([])
    expect(
      screen.queryByRole("button", { name: "Prompt actions" })
    ).not.toBeInTheDocument()

    const candidate = screen.getByRole("textbox", {
      name: "Improved prompt candidate"
    })
    await user.clear(candidate)
    await user.type(candidate, "Confirmed replacement")
    await user.click(
      screen.getByRole("button", { name: "Replace current draft" })
    )
    await user.click(screen.getByRole("button", { name: "Confirm replace" }))

    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Prompt actions" })
      ).toHaveFocus()
    )
    expect(focusRequests).toEqual(["applied"])
  })

  it("returns focus once after a successful Apply under StrictMode", async () => {
    const user = userEvent.setup()
    const focusRequests: string[] = []
    render(
      <StrictMode>
        <StatefulOwnerHarness focusRequests={focusRequests} />
      </StrictMode>
    )

    await user.click(screen.getByRole("button", { name: "Start review" }))
    await user.click(
      await screen.findByRole("button", { name: "Apply to draft" })
    )

    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Prompt actions" })
      ).toHaveFocus()
    )
    expect(focusRequests).toEqual(["applied"])
  })

  it("retains explicit Cancel focus return after owner remount", async () => {
    const user = userEvent.setup()
    const focusRequests: string[] = []
    render(<StatefulOwnerHarness focusRequests={focusRequests} />)

    await user.click(screen.getByRole("button", { name: "Start review" }))
    await user.click(await screen.findByRole("button", { name: "Cancel" }))

    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Prompt actions" })
      ).toHaveFocus()
    )
    expect(focusRequests).toEqual(["reviewing"])
  })

  it("does not request focus after the panel unmounts during Apply", async () => {
    const user = userEvent.setup()
    const focusRequests: string[] = []
    render(
      <StrictMode>
        <StatefulOwnerHarness focusRequests={focusRequests} unmountOnApply />
      </StrictMode>
    )

    await user.click(screen.getByRole("button", { name: "Start review" }))
    await user.click(
      await screen.findByRole("button", { name: "Apply to draft" })
    )

    expect(
      screen.getByRole("button", { name: "Prompt actions" })
    ).not.toHaveFocus()
    expect(focusRequests).toEqual([])
  })

  it("lets an owner focus a trigger that remounts only after assist closes", async () => {
    function OwnerHarness() {
      const [open, setOpen] = useState(true)
      const requestFocus = () => {
        setTimeout(() =>
          document
            .querySelector<HTMLButtonElement>("#remounted-trigger")
            ?.focus()
        )
      }
      return open ? (
        <PromptAssistPanel
          state={{ status: "analyzing", operation, undo: null }}
          {...callbacks()}
          onCancel={() => setOpen(false)}
          onRequestReturnFocus={requestFocus}
        />
      ) : (
        <button id="remounted-trigger">Prompt actions</button>
      )
    }

    const user = userEvent.setup()
    render(<OwnerHarness />)
    await user.click(screen.getByRole("button", { name: "Cancel" }))
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Prompt actions" })
      ).toHaveFocus()
    )
  })

  it("marks the analyzing spinner as static for reduced-motion users", () => {
    render(
      <PromptAssistPanel
        state={{ status: "analyzing", operation, undo: null }}
        {...callbacks()}
      />
    )
    expect(screen.getByTestId("prompt-assist-spinner")).toHaveClass(
      "motion-reduce:animate-none"
    )
  })
})
