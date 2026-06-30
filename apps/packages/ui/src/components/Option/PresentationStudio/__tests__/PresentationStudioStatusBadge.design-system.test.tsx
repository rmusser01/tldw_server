import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { getDesignSystemState } from "@/design-system"
import { PresentationStudioStatusBadge } from "../PresentationStudioStatusBadge"
import type { PresentationStudioAssetStatus } from "@/store/presentation-studio"

vi.mock("@/design-system", async (importActual) => {
  const actual = await importActual<typeof import("@/design-system")>()

  return {
    ...actual,
    getDesignSystemState: vi.fn(actual.getDesignSystemState),
  }
})

const STATUS_CASES = [
  ["missing", "empty", "secondary"],
  ["ready", "ready", "success"],
  ["stale", "degraded", "warning"],
  ["generating", "retrying", "info"],
  ["failed", "error", "danger"],
] satisfies Array<[PresentationStudioAssetStatus, string, string]>

describe("PresentationStudioStatusBadge design-system adapter", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it.each(STATUS_CASES)(
    "renders %s through the canonical %s state",
    (status, stateKey, variant) => {
      const { container } = render(
        <PresentationStudioStatusBadge status={status} />
      )

      const badge = container.querySelector('[data-ds-component="Badge"]')

      expect(screen.getByText(status)).toBeInTheDocument()
      expect(getDesignSystemState).toHaveBeenCalledWith(stateKey)
      expect(badge).toHaveAttribute("data-ds-size", "sm")
      expect(badge).toHaveAttribute("data-ds-variant", variant)
    }
  )

  it("falls back to the canonical empty state for nullish statuses", () => {
    const { container } = render(
      <PresentationStudioStatusBadge status={undefined} className="asset-pill" />
    )

    const badge = container.querySelector('[data-ds-component="Badge"]')

    expect(screen.getByText("missing")).toBeInTheDocument()
    expect(getDesignSystemState).toHaveBeenCalledWith("empty")
    expect(badge).toHaveClass("asset-pill")
  })
})
