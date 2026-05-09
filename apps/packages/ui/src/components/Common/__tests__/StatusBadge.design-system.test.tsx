import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { StatusBadge, type StatusBadgeProps } from "../StatusBadge"

describe("StatusBadge design-system adapter", () => {
  it.each([
    ["demo", "Demo mode"],
    ["warning", "Setup required"],
    ["error", "Feature unavailable"],
  ] satisfies Array<[StatusBadgeProps["variant"], string]>)(
    "renders the %s variant through the shared Badge primitive",
    (variant, label) => {
      const { container } = render(
        <StatusBadge variant={variant}>{label}</StatusBadge>
      )

      expect(screen.getByText(label)).toBeInTheDocument()
      expect(
        container.querySelector('[data-ds-component="Badge"]')
      ).toBeInTheDocument()
    }
  )
})
