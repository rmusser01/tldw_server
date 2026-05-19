import { render, screen } from "@testing-library/react"
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"
import { TakeQuizTab } from "../TakeQuizTab"
import {
  useAttemptsQuery,
  useQuizzesQuery,
  useQuizQuery,
  useStartAttemptMutation,
  useSubmitAttemptMutation
} from "../../hooks"

vi.mock("react-router-dom", () => ({
  Link: ({ to, children, ...props }: Record<string, unknown>) => <a href={to as string} {...props}>{children as React.ReactNode}</a>
}))

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

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("../../hooks", () => ({
  useAttemptsQuery: vi.fn(),
  useQuizzesQuery: vi.fn(),
  useQuizQuery: vi.fn(),
  useStartAttemptMutation: vi.fn(),
  useSubmitAttemptMutation: vi.fn()
}))

vi.mock("../../hooks/useQuizTimer", () => ({
  useQuizTimer: vi.fn(() => null)
}))

vi.mock("../../hooks/useQuizAutoSave", () => ({
  useQuizAutoSave: vi.fn(() => ({
    storageUnavailable: false,
    restoreSavedAnswers: vi.fn(async () => false),
    clearSavedProgress: vi.fn(async () => {}),
    hasSavedProgress: vi.fn(async () => false),
    getSavedProgress: vi.fn(async () => null),
    forceSave: vi.fn(async () => {})
  }))
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

describe("TakeQuizTab empty-state guidance", () => {
  const originalMatchMedia = window.matchMedia

  beforeAll(() => {
    if (typeof window.matchMedia !== "function") {
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
  })

  afterAll(() => {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: originalMatchMedia
    })
  })

  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    vi.mocked(useAttemptsQuery).mockReturnValue({
      data: { items: [], count: 0 }
    } as any)
    vi.mocked(useQuizzesQuery).mockReturnValue({
      data: { items: [], count: 0 },
      isLoading: false
    } as any)
    vi.mocked(useQuizQuery).mockReturnValue({
      data: null
    } as any)
    vi.mocked(useStartAttemptMutation).mockReturnValue({
      mutateAsync: vi.fn()
    } as any)
    vi.mocked(useSubmitAttemptMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
  })

  it("describes the take-flow state when no quizzes exist", () => {
    render(
      <MemoryRouter>
        <TakeQuizTab
          onNavigateToGenerate={() => {}}
          onNavigateToCreate={() => {}}
        />
      </MemoryRouter>
    )

    expect(
      screen.getByText("No quizzes available to take yet")
    ).toBeInTheDocument()
    const mediaLibraryLink = screen.getByRole("link", { name: /media library/i })
    expect(mediaLibraryLink).toHaveAttribute("href", "/media")
    expect(screen.getByText(/Generate one from your/)).toBeInTheDocument()
    expect(screen.getByText(/or create one manually/)).toBeInTheDocument()
  })
})
