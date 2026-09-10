import type { PromptImproveErrorCode } from "@/services/prompt-improvement"
import { render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import english from "../../../../assets/locale/en/common.json"
import { PromptAssistPanel } from "../PromptAssistPanel"
import { PromptReviewSurface } from "../PromptReviewSurface"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => `translated:${key}`
  })
}))

const errorCodes: PromptImproveErrorCode[] = [
  "invalid_input",
  "missing_model",
  "unsupported_model",
  "provider_not_configured",
  "draft_too_large",
  "provider_rate_limited",
  "provider_timeout",
  "provider_unavailable",
  "model_refusal",
  "invalid_model_output",
  "preservation_failed",
  "internal_error"
]

const warningCodes = [
  "unstructured_output",
  "target_mismatch",
  "placeholder_mismatch",
  "url_mismatch",
  "protected_token_mismatch",
  "code_fence_mismatch",
  "wrapper_mismatch",
  "large_rewrite",
  "unknown"
] as const

describe("PromptAssist locale contract", () => {
  it("contains English copy for every stable error and warning code", () => {
    expect(Object.keys(english.promptAssist.errors).sort()).toEqual(
      errorCodes.sort()
    )
    expect(Object.keys(english.promptAssist.warnings).sort()).toEqual(
      [...warningCodes].sort()
    )
  })

  it("localizes stable and unknown warnings without exposing raw server codes", () => {
    render(
      <PromptReviewSurface
        original="Draft"
        candidate="Improved draft"
        findings={[]}
        warnings={["protected_token_mismatch", "future_warning_code"]}
        notice={null}
        resolvedModel={{
          provider: "openai",
          model: "gpt",
          display_name: "GPT"
        }}
        onCandidateChange={vi.fn()}
        onApply={vi.fn()}
        onConfirmReplace={vi.fn()}
        onCancel={vi.fn()}
      />
    )

    const notices = screen.getByRole("list", {
      name: "translated:common:promptAssist.safetyNotices"
    })
    const items = within(notices).getAllByRole("listitem")
    expect(items[0]).toHaveTextContent(
      "translated:common:promptAssist.warnings.protected_token_mismatch"
    )
    expect(items[1]).toHaveTextContent(
      "translated:common:promptAssist.warnings.unknown"
    )
    expect(screen.queryByText(/future_warning_code/)).not.toBeInTheDocument()
  })

  it("localizes error recovery and the Auto analyzing route", () => {
    const operation = {
      operationId: "11111111-1111-4111-8111-111111111111",
      target: "system" as const,
      mode: "review_changes" as const,
      originalText: "Draft",
      revision: "r1",
      route: { selected_model: "auto" }
    }
    const callbacks = {
      onCancel: vi.fn(),
      onRetry: vi.fn(),
      onSelectModel: vi.fn(),
      onCandidateChange: vi.fn(),
      onApply: vi.fn(),
      onConfirmReplace: vi.fn(),
      onUndo: vi.fn()
    }
    const { rerender } = render(
      <PromptAssistPanel
        state={{
          status: "failed",
          operation,
          undo: null,
          error: {
            code: "draft_too_large",
            message: "raw english",
            retryable: false
          }
        }}
        {...callbacks}
      />
    )
    expect(screen.getByRole("alert")).toHaveTextContent(
      "translated:common:promptAssist.errors.draft_too_large"
    )
    expect(screen.queryByText("raw english")).not.toBeInTheDocument()

    rerender(
      <PromptAssistPanel
        state={{ status: "analyzing", operation, undo: null }}
        {...callbacks}
      />
    )
    expect(screen.getByRole("status")).toHaveTextContent(
      "translated:common:promptAssist.analyzing"
    )
  })
})
