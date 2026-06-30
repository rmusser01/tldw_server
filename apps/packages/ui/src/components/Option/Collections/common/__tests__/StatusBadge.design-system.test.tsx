import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { StatusBadge } from "../StatusBadge"
import type { ReadingStatus } from "@/types/collections"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback: string) => fallback,
  }),
}))

describe("Collections StatusBadge design-system adapter", () => {
  it.each([
    ["saved", "saved"],
    ["reading", "reading"],
    ["read", "read"],
    ["archived", "archived"],
  ] satisfies Array<[ReadingStatus, string]>)(
    "renders the %s status through the shared Badge primitive",
    (status, label) => {
      const { container } = render(<StatusBadge status={status} />)

      expect(screen.getByText(label)).toBeInTheDocument()
      expect(screen.getByTestId(`collections-status-icon-${status}`)).toBeInTheDocument()
      expect(
        container.querySelector('[data-ds-component="Badge"]')
      ).toBeInTheDocument()
    }
  )

  it("uses the shared Badge compact sizing for small badges", () => {
    const { container } = render(<StatusBadge status="saved" size="small" />)
    const badge = container.querySelector('[data-ds-component="Badge"]')

    expect(badge).toHaveClass("py-0.5")
    expect(badge).toHaveClass("text-[10px]")
    expect(badge).not.toHaveClass("py-0")
    expect(badge).not.toHaveClass("text-xs")
  })
})
