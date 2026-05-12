// @vitest-environment jsdom
import { render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { describe, expect, it } from "vitest"

import { MODERATION_RULES_PATH } from "@/routes/route-paths"
import { ModerationReviewShell } from "../ModerationReviewShell"

const renderShell = () =>
  render(
    <MemoryRouter>
      <ModerationReviewShell />
    </MemoryRouter>
  )

describe("ModerationReviewShell", () => {
  it("renders an honest first-slice review queue state", () => {
    renderShell()

    expect(
      screen.getByRole("heading", { name: "Moderation Review" })
    ).toBeInTheDocument()
    expect(screen.getByText(/review queue is not connected yet/i)).toBeInTheDocument()
    expect(screen.getByText(/backend contract pending/i)).toBeInTheDocument()
    expect(screen.getByText(/reviewer permission pending/i)).toBeInTheDocument()
    expect(screen.getByText(/needs review/i)).toBeInTheDocument()
    expect(screen.getByText(/blocked/i)).toBeInTheDocument()
    expect(screen.getByText(/redacted/i)).toBeInTheDocument()
  })

  it("links reviewers to content rules without making it the primary destination", () => {
    renderShell()

    const rulesLink = screen.getByRole("link", { name: /open content rules/i })
    expect(rulesLink).toHaveAttribute("href", MODERATION_RULES_PATH)
  })
})
