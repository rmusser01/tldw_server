// @vitest-environment jsdom
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { BulkDecisionBar } from "../BulkDecisionBar"
import type { ModerationReviewBulkDecisionResponse } from "@/services/moderation"

const partialResult: ModerationReviewBulkDecisionResponse = {
  ok_count: 1,
  error_count: 1,
  results: [
    {
      item_id: "review-1",
      ok: true
    },
    {
      item_id: "review-2",
      ok: false,
      error: "not_found"
    }
  ]
}

describe("BulkDecisionBar", () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it("requires a reason and confirmation for high-risk bulk decisions", async () => {
    const user = userEvent.setup()
    const onBulkDecision = vi.fn()
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true)

    render(
      <BulkDecisionBar
        selectedCount={2}
        onBulkDecision={onBulkDecision}
        onClearSelection={vi.fn()}
      />
    )

    await user.click(screen.getByRole("button", { name: /redact selected/i }))
    expect(screen.getByText(/reason required/i)).toBeInTheDocument()
    expect(onBulkDecision).not.toHaveBeenCalled()

    await user.type(screen.getByLabelText(/bulk decision reason/i), "Privacy request")
    await user.click(screen.getByRole("button", { name: /redact selected/i }))

    await waitFor(() => {
      expect(onBulkDecision).toHaveBeenCalledWith("redact", "Privacy request")
    })
    expect(confirmSpy).toHaveBeenCalled()
  })

  it("renders partial failure feedback and clear selection affordance", () => {
    const onClearSelection = vi.fn()

    render(
      <BulkDecisionBar
        selectedCount={2}
        result={partialResult}
        onBulkDecision={vi.fn()}
        onClearSelection={onClearSelection}
      />
    )

    expect(screen.getByText(/1 updated/i)).toBeInTheDocument()
    expect(screen.getByText(/1 failed/i)).toBeInTheDocument()
    expect(screen.getByText(/review-2: not_found/i)).toBeInTheDocument()
    screen.getByRole("button", { name: /clear selection/i }).click()
    expect(onClearSelection).toHaveBeenCalled()
  })
})
