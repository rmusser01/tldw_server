import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { FlashcardTagPicker } from "../FlashcardTagPicker"

const mockUseGlobalFlashcardTagSuggestionsQuery = vi.hoisted(() => vi.fn())

vi.mock("../../hooks", () => ({
  useGlobalFlashcardTagSuggestionsQuery: mockUseGlobalFlashcardTagSuggestionsQuery
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
      return defaultValueOrOptions?.defaultValue ?? key
    }
  })
}))

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

type HarnessProps = {
  active?: boolean
  initialValue?: string[]
  onChangeSpy?: ReturnType<typeof vi.fn>
  dataTestId?: string
}

function renderPicker({
  active = true,
  initialValue = [],
  onChangeSpy = vi.fn(),
  dataTestId
}: HarnessProps = {}) {
  const Harness = () => {
    const [value, setValue] = React.useState<string[]>(initialValue)

    return (
      <FlashcardTagPicker
        active={active}
        value={value}
        dataTestId={dataTestId}
        onChange={(next) => {
          onChangeSpy(next)
          setValue(next)
        }}
      />
    )
  }

  return {
    onChangeSpy,
    ...render(<Harness />)
  }
}

describe("FlashcardTagPicker", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("shows backend suggestions when opened", async () => {
    mockUseGlobalFlashcardTagSuggestionsQuery.mockImplementation(
      (_query: string | null | undefined, options?: { enabled?: boolean }) => ({
        data: options?.enabled
          ? {
              items: [
                { tag: "Biology", count: 4 },
                { tag: "bioinformatics", count: 2 }
              ],
              count: 2
            }
          : undefined,
        isError: false,
        isFetching: false,
        isLoading: false,
        error: null
      })
    )

    renderPicker({ dataTestId: "flashcards-create-tag-picker" })

    expect(screen.getByTestId("flashcards-create-tag-picker")).toBeInTheDocument()

    fireEvent.mouseDown(screen.getByRole("combobox"))

    await waitFor(() => {
      expect(screen.getByRole("option", { name: "Biology" })).toBeInTheDocument()
    })
    expect(
      await screen.findByTestId("flashcards-create-tag-picker-search-input")
    ).toBeInTheDocument()
    expect(screen.getByRole("option", { name: "bioinformatics" })).toBeInTheDocument()
  })

  it("selecting an existing suggestion emits normalized tags", async () => {
    mockUseGlobalFlashcardTagSuggestionsQuery.mockImplementation(
      (_query: string | null | undefined, options?: { enabled?: boolean }) => ({
        data: options?.enabled
          ? {
              items: [{ tag: "  Biology  ", count: 4 }],
              count: 1
            }
          : undefined,
        isError: false,
        isFetching: false,
        isLoading: false,
        error: null
      })
    )

    const { onChangeSpy } = renderPicker()

    fireEvent.mouseDown(screen.getByRole("combobox"))

    await waitFor(() => {
      expect(document.querySelector(".ant-select-item-option")).toBeInTheDocument()
    })
    const suggestion = document.querySelector(".ant-select-item-option") as HTMLElement
    fireEvent.mouseDown(suggestion)
    fireEvent.click(suggestion)

    await waitFor(() => {
      expect(onChangeSpy).toHaveBeenCalledWith(["Biology"])
    })
  })

  it("typing a new tag and pressing Enter works", async () => {
    mockUseGlobalFlashcardTagSuggestionsQuery.mockImplementation(() => ({
      data: undefined,
      isError: false,
      isFetching: false,
      isLoading: false,
      error: null
    }))

    const { onChangeSpy } = renderPicker()

    fireEvent.mouseDown(screen.getByRole("combobox"))

    const searchInput = await screen.findByTestId("flashcard-tag-picker-search-input")
    fireEvent.change(searchInput, { target: { value: "Neuroscience" } })
    fireEvent.keyDown(searchInput, {
      key: "Enter",
      code: "Enter",
      charCode: 13,
      keyCode: 13
    })

    await waitFor(() => {
      expect(onChangeSpy).toHaveBeenCalledWith(["Neuroscience"])
    })
  })

  it("ignores whitespace-only values", async () => {
    mockUseGlobalFlashcardTagSuggestionsQuery.mockImplementation(() => ({
      data: undefined,
      isError: false,
      isFetching: false,
      isLoading: false,
      error: null
    }))

    const { onChangeSpy } = renderPicker()

    fireEvent.mouseDown(screen.getByRole("combobox"))

    const searchInput = await screen.findByTestId("flashcard-tag-picker-search-input")
    fireEvent.change(searchInput, { target: { value: "   " } })
    fireEvent.keyDown(searchInput, {
      key: "Enter",
      code: "Enter",
      charCode: 13,
      keyCode: 13
    })

    await waitFor(() => {
      expect(onChangeSpy).toHaveBeenCalledWith([])
    })
  })

  it("collapses duplicate values case-insensitively", async () => {
    mockUseGlobalFlashcardTagSuggestionsQuery.mockImplementation(() => ({
      data: undefined,
      isError: false,
      isFetching: false,
      isLoading: false,
      error: null
    }))

    const { onChangeSpy } = renderPicker({ initialValue: ["Bio"] })

    fireEvent.mouseDown(screen.getByRole("combobox"))

    const searchInput = await screen.findByTestId("flashcard-tag-picker-search-input")
    fireEvent.change(searchInput, { target: { value: "bio" } })
    fireEvent.keyDown(searchInput, {
      key: "Enter",
      code: "Enter",
      charCode: 13,
      keyCode: 13
    })

    await waitFor(() => {
      expect(onChangeSpy).toHaveBeenCalledWith(["Bio"])
    })
  })

  it("allows free typing when fetch fails", async () => {
    mockUseGlobalFlashcardTagSuggestionsQuery.mockImplementation(() => ({
      data: undefined,
      isError: true,
      isFetching: false,
      isLoading: false,
      error: new Error("tag lookup failed")
    }))

    const { onChangeSpy } = renderPicker()

    fireEvent.mouseDown(screen.getByRole("combobox"))

    const searchInput = await screen.findByTestId("flashcard-tag-picker-search-input")
    fireEvent.change(searchInput, { target: { value: "Cardiology" } })
    fireEvent.keyDown(searchInput, {
      key: "Enter",
      code: "Enter",
      charCode: 13,
      keyCode: 13
    })

    await waitFor(() => {
      expect(onChangeSpy).toHaveBeenCalledWith(["Cardiology"])
    })
  })

  it("does not throw before Form.Item injects value and onChange", async () => {
    mockUseGlobalFlashcardTagSuggestionsQuery.mockImplementation(() => ({
      data: undefined,
      isError: false,
      isFetching: false,
      isLoading: false,
      error: null
    }))

    render(
      <FlashcardTagPicker
        active
        dataTestId="flashcards-form-controlled-tag-picker"
      />
    )

    fireEvent.mouseDown(screen.getByRole("combobox"))

    const searchInput = await screen.findByTestId(
      "flashcards-form-controlled-tag-picker-search-input"
    )
    fireEvent.change(searchInput, { target: { value: "Cardiology" } })

    expect(() => {
      fireEvent.keyDown(searchInput, {
        key: "Enter",
        code: "Enter",
        charCode: 13,
        keyCode: 13
      })
    }).not.toThrow()
  })
})
