import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it } from "vitest"
import { ComposerStyleSettings } from "../settings/ComposerStyleSettings"
import { COMPOSER_VARIANT_PREFERENCE_KEY } from "../hooks/useComposerVariantPreference"

describe("ComposerStyleSettings", () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it("renders a radiogroup with all three variants", () => {
    render(<ComposerStyleSettings />)
    expect(
      screen.getByRole("radiogroup", { name: /composer variant/i })
    ).toBeTruthy()
    expect(
      screen.getByRole("radio", { name: /terminal stack/i })
    ).toBeTruthy()
    expect(screen.getByRole("radio", { name: /split brief/i })).toBeTruthy()
    expect(
      screen.getByRole("radio", { name: /radial command/i })
    ).toBeTruthy()
  })

  it("marks V1 as checked by default", () => {
    render(<ComposerStyleSettings />)
    const v1 = screen.getByRole("radio", { name: /terminal stack/i })
    expect(v1.getAttribute("aria-checked")).toBe("true")
  })

  it("reads the stored preference on mount", () => {
    window.localStorage.setItem(COMPOSER_VARIANT_PREFERENCE_KEY, "v5")
    render(<ComposerStyleSettings />)
    const v5 = screen.getByRole("radio", { name: /radial command/i })
    expect(v5.getAttribute("aria-checked")).toBe("true")
    const v1 = screen.getByRole("radio", { name: /terminal stack/i })
    expect(v1.getAttribute("aria-checked")).toBe("false")
  })

  it("updates the preference and persists to localStorage when a card is clicked", () => {
    render(<ComposerStyleSettings />)
    const v3 = screen.getByRole("radio", { name: /split brief/i })
    fireEvent.click(v3)

    expect(v3.getAttribute("aria-checked")).toBe("true")
    expect(
      window.localStorage.getItem(COMPOSER_VARIANT_PREFERENCE_KEY)
    ).toBe("v3")
  })

  it("activates focused variant cards with Space and Enter", () => {
    render(<ComposerStyleSettings />)
    const v3 = screen.getByRole("radio", { name: /split brief/i })
    const v5 = screen.getByRole("radio", { name: /radial command/i })

    v3.focus()
    expect(document.activeElement).toBe(v3)
    fireEvent.keyDown(v3, { key: " ", code: "Space" })
    expect(v3.getAttribute("aria-checked")).toBe("true")

    v5.focus()
    expect(document.activeElement).toBe(v5)
    fireEvent.keyDown(v5, { key: "Enter", code: "Enter" })
    expect(v5.getAttribute("aria-checked")).toBe("true")
  })

  it("only one card is checked at a time", () => {
    render(<ComposerStyleSettings />)
    fireEvent.click(screen.getByRole("radio", { name: /split brief/i }))
    fireEvent.click(screen.getByRole("radio", { name: /radial command/i }))

    const radios = screen.getAllByRole("radio")
    const checked = radios.filter(
      (r) => r.getAttribute("aria-checked") === "true"
    )
    expect(checked).toHaveLength(1)
    expect(checked[0].textContent).toContain("Radial Command")
  })

  it("exposes data-variant-option attrs for integration testing", () => {
    const { container } = render(<ComposerStyleSettings />)
    expect(container.querySelector('[data-variant-option="v1"]')).toBeTruthy()
    expect(container.querySelector('[data-variant-option="v3"]')).toBeTruthy()
    expect(container.querySelector('[data-variant-option="v5"]')).toBeTruthy()
  })

  it("renders description copy for each variant", () => {
    render(<ComposerStyleSettings />)
    expect(screen.getByText(/cleaned-up take/i)).toBeTruthy()
    expect(screen.getByText(/labelled field chips/i)).toBeTruthy()
    expect(screen.getByText(/single line with a/i)).toBeTruthy()
  })
})
