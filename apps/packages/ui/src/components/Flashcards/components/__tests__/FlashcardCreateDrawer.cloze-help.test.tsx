import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { FlashcardCreateDrawer } from "../FlashcardCreateDrawer"
import {
  useCreateDeckMutation,
  useCreateFlashcardMutation,
  useCreateFlashcardTemplateMutation,
  useDecksQuery
} from "../../hooks"
import { FLASHCARDS_DRAWER_WIDTH_PX } from "../../constants"

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

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
    loading: vi.fn(),
    open: vi.fn(),
    destroy: vi.fn()
  })
}))

vi.mock("../../hooks", () => ({
  useDecksQuery: vi.fn(),
  useCreateFlashcardMutation: vi.fn(),
  useCreateDeckMutation: vi.fn(),
  useCreateFlashcardTemplateMutation: vi.fn(),
  useDebouncedFormField: vi.fn(() => undefined),
  useFlashcardDeckRecentCardsQuery: vi.fn(() => ({
    data: [],
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn()
  })),
  useFlashcardDeckSearchQuery: vi.fn(() => ({
    data: [],
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn()
  }))
}))

vi.mock("../MarkdownWithBoundary", () => ({
  MarkdownWithBoundary: ({ content }: { content: string }) => <div>{content}</div>
}))

vi.mock("../FlashcardTagPicker", () => ({
  FlashcardTagPicker: ({ dataTestId }: { dataTestId?: string }) => (
    <div data-testid={dataTestId ?? "flashcard-tag-picker"} />
  )
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

describe("FlashcardCreateDrawer cloze helper and validation", () => {
  const mutateAsync = vi.fn()
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    vi.clearAllMocks()
    consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    vi.mocked(useDecksQuery).mockReturnValue({
      data: [
        {
          id: 1,
          name: "Biology",
          description: null,
          deleted: false,
          client_id: "test",
          version: 1
        }
      ],
      isLoading: false
    } as any)
    vi.mocked(useCreateFlashcardMutation).mockReturnValue({
      mutateAsync,
      isPending: false
    } as any)
    vi.mocked(useCreateDeckMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useCreateFlashcardTemplateMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
  })

  afterEach(() => {
    expect(consoleErrorSpy).not.toHaveBeenCalled()
    consoleErrorSpy.mockRestore()
  })

  it("shows template guidance and blocks invalid cloze front syntax", async () => {
    render(
      <FlashcardCreateDrawer
        open
        onClose={vi.fn()}
        onSuccess={vi.fn()}
      />
    )

    const wrapper = document.querySelector(".ant-drawer-content-wrapper") as HTMLElement | null
    expect(wrapper?.style.width).toBe(`${FLASHCARDS_DRAWER_WIDTH_PX}px`)

    expect(
      screen.getByText(
        "Choose Basic for direct question and answer cards (facts, definitions, short prompts)."
      )
    ).toBeInTheDocument()

    fireEvent.mouseDown(screen.getByLabelText("Card model"))
    fireEvent.click(screen.getByText("Cloze (Fill in the blank)"))

    await waitFor(() => {
      expect(
        screen.getByText(
          "Choose Cloze when you want to hide key words inside a sentence or paragraph."
        )
      ).toBeInTheDocument()
    })
    await waitFor(() => {
      expect(
        screen.getByText("Cloze syntax: add at least one deletion like {{c1::answer}} in Front text.")
      ).toBeInTheDocument()
    })

    fireEvent.change(screen.getByPlaceholderText("Question or prompt..."), {
      target: { value: "This front has no cloze pattern" }
    })
    fireEvent.change(screen.getByPlaceholderText("Answer..."), {
      target: { value: "Back content" }
    })

    fireEvent.click(screen.getByRole("button", { name: "Create" }))

    await waitFor(() => {
      expect(
        screen.getByText(
          "For Cloze cards, include at least one deletion like {{c1::answer}}."
        )
      ).toBeInTheDocument()
    })
    expect(mutateAsync).not.toHaveBeenCalled()
  }, 15000)

  it("shows byte-limit guidance and blocks over-limit front content", async () => {
    render(
      <FlashcardCreateDrawer
        open
        onClose={vi.fn()}
        onSuccess={vi.fn()}
      />
    )

    fireEvent.change(screen.getByPlaceholderText("Question or prompt..."), {
      target: { value: "a".repeat(8192) }
    })
    fireEvent.change(screen.getByPlaceholderText("Answer..."), {
      target: { value: "Back content" }
    })

    await waitFor(() => {
      expect(screen.getByText("Front: 8192 / 8192 bytes. Approaching the 8192-byte limit.")).toBeInTheDocument()
    })

    fireEvent.change(screen.getByPlaceholderText("Question or prompt..."), {
      target: { value: "a".repeat(8193) }
    })
    fireEvent.click(screen.getByRole("button", { name: "Create" }))

    await waitFor(() => {
      expect(screen.getByText("Front must be 8192 bytes or fewer.")).toBeInTheDocument()
    })
    expect(mutateAsync).not.toHaveBeenCalled()
  }, 15000)

  it("does not require cloze syntax for basic cards", async () => {
    mutateAsync.mockResolvedValueOnce({
      uuid: "card-1"
    })

    render(
      <FlashcardCreateDrawer
        open
        onClose={vi.fn()}
        onSuccess={vi.fn()}
      />
    )

    fireEvent.change(screen.getByPlaceholderText("Question or prompt..."), {
      target: { value: "Plain front content without cloze syntax" }
    })
    fireEvent.change(screen.getByPlaceholderText("Answer..."), {
      target: { value: "Plain back content" }
    })

    fireEvent.click(screen.getByRole("button", { name: "Create" }))

    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledTimes(1)
    })
  }, 15000)
})
