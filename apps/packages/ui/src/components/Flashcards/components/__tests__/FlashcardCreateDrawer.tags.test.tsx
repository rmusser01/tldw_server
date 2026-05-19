import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { FlashcardCreateDrawer } from "../FlashcardCreateDrawer"
import {
  useCreateDeckMutation,
  useCreateFlashcardMutation,
  useDecksQuery
} from "../../hooks"

const mockFlashcardTagPicker = vi.hoisted(() => vi.fn())

vi.mock("../FlashcardTagPicker", () => ({
  FlashcardTagPicker: (props: Record<string, unknown>) => mockFlashcardTagPicker(props)
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
  useDebouncedFormField: vi.fn(() => undefined)
}))

vi.mock("../MarkdownWithBoundary", () => ({
  MarkdownWithBoundary: ({ content }: { content: string }) => <div>{content}</div>
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

const renderMockTagPicker = ({
  value = [],
  onChange,
  active,
  placeholder,
  dataTestId
}: {
  value?: string[]
  onChange?: (next: string[]) => void
  active?: boolean
  placeholder?: string
  dataTestId?: string
}) => (
  <div
    data-testid={dataTestId}
    data-active={String(active)}
    data-placeholder={placeholder}
  >
    <div data-testid={`${dataTestId}-value`}>{JSON.stringify(value)}</div>
    <button
      type="button"
      onClick={() => onChange?.([...(value ?? []), "Biology"])}
    >
      choose suggested tag
    </button>
    <input
      aria-label={`${dataTestId}-input`}
      data-testid={`${dataTestId}-input`}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          onChange?.([...(value ?? []), event.currentTarget.value])
        }
      }}
    />
  </div>
)

describe("FlashcardCreateDrawer tags", () => {
  const mutateAsync = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
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
    mockFlashcardTagPicker.mockImplementation(renderMockTagPicker)
  })

  it("choosing a suggested tag in Advanced options submits that tag", async () => {
    mutateAsync.mockResolvedValueOnce({ uuid: "card-1" })

    render(
      <FlashcardCreateDrawer
        open
        onClose={vi.fn()}
        onSuccess={vi.fn()}
      />
    )

    fireEvent.click(screen.getByText("Advanced options (tags, extra, notes)"))

    const picker = await screen.findByTestId("flashcards-create-tag-picker")
    expect(picker).toHaveAttribute("data-active", "true")
    expect(picker).toHaveAttribute("data-placeholder", "tag1, tag2")

    fireEvent.click(screen.getByRole("button", { name: "choose suggested tag" }))

    fireEvent.change(screen.getByPlaceholderText("Question or prompt..."), {
      target: { value: "Front content" }
    })
    fireEvent.change(screen.getByPlaceholderText("Answer..."), {
      target: { value: "Back content" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Create" }))

    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledWith(
        expect.objectContaining({
          tags: ["Biology"]
        })
      )
    })
  }, 15000)

  it("typing a new tag still submits successfully", async () => {
    mutateAsync.mockResolvedValueOnce({ uuid: "card-1" })

    render(
      <FlashcardCreateDrawer
        open
        onClose={vi.fn()}
        onSuccess={vi.fn()}
      />
    )

    fireEvent.click(screen.getByText("Advanced options (tags, extra, notes)"))

    const input = await screen.findByTestId("flashcards-create-tag-picker-input")
    fireEvent.change(input, { target: { value: "Neuroscience" } })
    fireEvent.keyDown(input, {
      key: "Enter",
      code: "Enter",
      charCode: 13,
      keyCode: 13
    })

    fireEvent.change(screen.getByPlaceholderText("Question or prompt..."), {
      target: { value: "Front content" }
    })
    fireEvent.change(screen.getByPlaceholderText("Answer..."), {
      target: { value: "Back content" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Create" }))

    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledWith(
        expect.objectContaining({
          tags: ["Neuroscience"]
        })
      )
    })
  }, 15000)
})
