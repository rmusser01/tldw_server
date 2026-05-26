import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { KeyboardShortcutsModal } from "../KeyboardShortcutsModal"

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
      if (defaultValueOrOptions?.defaultValue) {
        return defaultValueOrOptions.defaultValue.replace(
          /\{\{(\w+)\}\}/g,
          (_match, token: string) =>
            String((defaultValueOrOptions as Record<string, unknown>)[token] ?? `{{${token}}}`)
        )
      }
      return key
    }
  })
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

if (typeof window !== "undefined" && typeof window.matchMedia !== "function") {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn()
    }))
  })
}

describe("KeyboardShortcutsModal rating scale guidance", () => {
  it("shows explicit rating-to-SM2 mapping on the review tab", () => {
    render(
      <KeyboardShortcutsModal
        open
        onClose={vi.fn()}
        activeTab="review"
      />
    )

    expect(screen.getByText("Rating Scale (SM-2 values)")).toBeInTheDocument()
    expect(screen.getByText("Again = 0 (forgot it, repeat very soon)")).toBeInTheDocument()
    expect(screen.getByText("Hard = 2 (remembered with strain, short gap)")).toBeInTheDocument()
    expect(screen.getByText("Good = 3 (normal recall, default step)")).toBeInTheDocument()
    expect(screen.getByText("Easy = 5 (effortless recall, longest jump)")).toBeInTheDocument()
  })

  it("maps review shortcuts to visible review controls", () => {
    render(
      <KeyboardShortcutsModal
        open
        onClose={vi.fn()}
        activeTab="review"
      />
    )

    expect(screen.getByText("Show answer")).toBeInTheDocument()
    expect(screen.getByText("Rate with Again, Hard, Good, or Easy")).toBeInTheDocument()
    expect(screen.getByText("Re-rate last card while the undo button is visible")).toBeInTheDocument()
  })
})
