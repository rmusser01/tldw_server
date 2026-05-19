// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { Form } from "antd"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { FlashcardCreateDrawer } from "../FlashcardCreateDrawer"
import {
  useCreateDeckMutation,
  useCreateFlashcardMutation,
  useCreateFlashcardTemplateMutation,
  useDecksQuery,
  useFlashcardTemplatesQuery
} from "../../hooks"
import type {
  DeckSchedulerSettingsEnvelope,
  FlashcardTemplate
} from "@/services/flashcards"

const defaultSchedulerSettings: DeckSchedulerSettingsEnvelope = {
  sm2_plus: {
    new_steps_minutes: [1, 10],
    relearn_steps_minutes: [10],
    graduating_interval_days: 1,
    easy_interval_days: 4,
    easy_bonus: 1.3,
    interval_modifier: 1,
    max_interval_days: 36500,
    leech_threshold: 8,
    enable_fuzz: false
  },
  fsrs: {
    target_retention: 0.9,
    maximum_interval_days: 36500,
    enable_fuzz: false
  }
}

const createFlashcardMutateAsync = vi.fn()
const createDeckMutateAsync = vi.fn()
const createTemplateMutateAsync = vi.fn()
const messageApi = {
  success: vi.fn(),
  error: vi.fn(),
  info: vi.fn(),
  warning: vi.fn(),
  loading: vi.fn(),
  open: vi.fn(),
  destroy: vi.fn()
}
let currentTemplates: FlashcardTemplate[] = []

const makeDeck = (id: number, name: string) =>
  ({
    id,
    name,
    description: null,
    deleted: false,
    client_id: "test",
    version: 1,
    scheduler_type: "sm2_plus",
    scheduler_settings_json: null,
    scheduler_settings: defaultSchedulerSettings
  }) as const

const template: FlashcardTemplate = {
  id: 11,
  name: "Definition scaffold",
  model_type: "basic",
  front_template: "What does {{term}} mean?",
  back_template: "{{definition}}",
  notes_template: "Hint: {{hint}}",
  extra_template: null,
  placeholder_definitions: [
    {
      key: "term",
      label: "Term",
      help_text: null,
      default_value: null,
      required: true,
      targets: ["front_template"]
    },
    {
      key: "definition",
      label: "Definition",
      help_text: null,
      default_value: null,
      required: true,
      targets: ["back_template"]
    },
    {
      key: "hint",
      label: "Hint",
      help_text: null,
      default_value: "Remember the cell biology context",
      required: false,
      targets: ["notes_template"]
    }
  ],
  created_at: "2026-04-15T00:00:00Z",
  last_modified: "2026-04-15T00:00:00Z",
  deleted: false,
  client_id: "test-client",
  version: 1
}

const reverseTemplate: FlashcardTemplate = {
  ...template,
  id: 12,
  name: "Reverse vocabulary",
  model_type: "basic_reverse",
  front_template: "{{term}}",
  back_template: "{{definition}}",
  notes_template: null,
  placeholder_definitions: template.placeholder_definitions.slice(0, 2)
}

const clozeTemplate: FlashcardTemplate = {
  ...template,
  id: 13,
  name: "Cloze scaffold",
  model_type: "cloze",
  front_template: "{{c1::{{term}}}} powers the cell",
  back_template: null,
  notes_template: null,
  placeholder_definitions: [
    {
      key: "term",
      label: "Term",
      help_text: null,
      default_value: null,
      required: true,
      targets: ["front_template"]
    }
  ]
}

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
  useAntdMessage: () => messageApi
}))

vi.mock("../../hooks", () => ({
  useDecksQuery: vi.fn(),
  useCreateFlashcardMutation: vi.fn(),
  useCreateDeckMutation: vi.fn(),
  useCreateFlashcardTemplateMutation: vi.fn(),
  useFlashcardTemplatesQuery: vi.fn(),
  useDebouncedFormField: vi.fn((form, field) => Form.useWatch(field, form)),
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

vi.mock("../FlashcardDeckReferenceSection", () => ({
  FlashcardDeckReferenceSection: () => null
}))

vi.mock("../DeckSchedulerSettingsEditor", () => ({
  DeckSchedulerSettingsEditor: () => null
}))

vi.mock("../FlashcardImageInsertButton", () => ({
  FlashcardImageInsertButton: ({ ariaLabel, buttonLabel }: { ariaLabel: string; buttonLabel: string }) => (
    <button type="button" aria-label={ariaLabel}>
      {buttonLabel}
    </button>
  )
}))

vi.mock("../FlashcardTagPicker", () => ({
  FlashcardTagPicker: ({
    value = [],
    onChange,
    dataTestId,
    placeholder
  }: {
    value?: string[]
    onChange?: (value: string[]) => void
    dataTestId?: string
    placeholder?: string
  }) => (
    <div data-testid={dataTestId ?? "flashcard-tag-picker"}>
      {value.map((tag) => (
        <span key={tag}>{tag}</span>
      ))}
      <input
        data-testid={`${dataTestId ?? "flashcard-tag-picker"}-search-input`}
        placeholder={placeholder}
        onKeyDown={(event) => {
          if (event.key !== "Enter") return
          event.preventDefault()
          const nextTag = event.currentTarget.value.trim()
          if (!nextTag) return
          onChange?.([...value, nextTag])
          event.currentTarget.value = ""
        }}
      />
    </div>
  )
}))

vi.mock("../utils/text-selection", () => ({
  getSelectionFromElement: () => ({ start: 0, end: 0 }),
  insertTextAtSelection: (value: string, _selection: { start: number; end: number }, inserted: string) => ({
    nextValue: `${value}${inserted}`,
    cursor: value.length + inserted.length
  }),
  restoreSelection: () => {}
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

describe("FlashcardCreateDrawer template flows", () => {
  beforeEach(() => {
    createFlashcardMutateAsync.mockReset()
    createDeckMutateAsync.mockReset()
    createTemplateMutateAsync.mockReset()
    vi.clearAllMocks()
    currentTemplates = [template]
    vi.mocked(useDecksQuery).mockReturnValue({
      data: [makeDeck(1, "Biology")],
      isLoading: false
    } as any)
    vi.mocked(useCreateFlashcardMutation).mockReturnValue({
      mutateAsync: createFlashcardMutateAsync,
      isPending: false
    } as any)
    vi.mocked(useCreateDeckMutation).mockReturnValue({
      mutateAsync: createDeckMutateAsync,
      isPending: false
    } as any)
    vi.mocked(useCreateFlashcardTemplateMutation).mockReturnValue({
      mutateAsync: createTemplateMutateAsync,
      isPending: false
    } as any)
    vi.mocked(useFlashcardTemplatesQuery).mockImplementation(() => ({
      data: {
        items: currentTemplates,
        count: currentTemplates.length,
        total: currentTemplates.length
      },
      isLoading: false,
      error: null
    } as any))
  })

  const openTemplateModal = async () => {
    fireEvent.click(screen.getByRole("button", { name: "Apply template" }))

    const modal = await waitFor(() => {
      const activeModal = screen.getAllByRole("dialog").find((dialog) => {
        return within(dialog).queryByLabelText("Template") !== null
      })

      expect(activeModal).toBeDefined()
      return activeModal as HTMLElement
    })

    return modal
  }

  const selectDeck = async (deckName: string) => {
    fireEvent.mouseDown(screen.getByLabelText("Deck"))
    fireEvent.click(await screen.findByText(deckName, { selector: ".ant-select-item-option-content" }))
  }

  const openAdvancedOptions = async () => {
    fireEvent.click(
      screen.getByText("Advanced options (tags, extra, notes)")
    )

    await waitFor(() => {
      expect(screen.getByTestId("flashcards-create-tag-picker-search-input")).toBeInTheDocument()
    })
  }

  it("applies a template through the value modal and preserves the current deck and tags", async () => {
    render(<FlashcardCreateDrawer open onClose={vi.fn()} onSuccess={vi.fn()} />)

    await selectDeck("Biology")
    await openAdvancedOptions()

    const tagsField = screen.getByTestId("flashcards-create-tag-picker-search-input")
    fireEvent.change(tagsField, {
      target: { value: "science" }
    })
    fireEvent.keyDown(tagsField, {
      key: "Enter",
      code: "Enter",
      charCode: 13
    })

    const modal = await openTemplateModal()

    fireEvent.change(within(modal).getByLabelText("Term"), {
      target: { value: "ATP" }
    })
    fireEvent.change(within(modal).getByLabelText("Definition"), {
      target: { value: "The cell's energy currency" }
    })
    await waitFor(() => {
      expect(within(modal).getByRole("button", { name: "Apply" })).not.toBeDisabled()
    })
    fireEvent.click(within(modal).getByRole("button", { name: "Apply" }))

    await waitFor(() => {
      expect(screen.getByDisplayValue("What does ATP mean?")).toBeInTheDocument()
    })

    const deckField = screen.getByText("Deck").closest(".ant-form-item")
    const modelField = screen.getByText("Card model").closest(".ant-form-item")

    expect(deckField).not.toBeNull()
    expect(modelField).not.toBeNull()
    expect(screen.getByDisplayValue("The cell's energy currency")).toBeInTheDocument()
    expect(screen.getByText("science")).toBeInTheDocument()
    expect(within(deckField as HTMLElement).getAllByText("Biology").length).toBeGreaterThan(0)
    expect(
      within(modelField as HTMLElement).getAllByText("Basic (Question - Answer)").length
    ).toBeGreaterThan(0)
  })

  it("disables template apply until required placeholder values are provided", async () => {
    render(<FlashcardCreateDrawer open onClose={vi.fn()} onSuccess={vi.fn()} />)

    const modal = await openTemplateModal()
    const applyButton = within(modal).getByRole("button", { name: "Apply" })

    expect(applyButton).toBeDisabled()

    fireEvent.change(within(modal).getByLabelText("Term"), {
      target: { value: "ATP" }
    })
    fireEvent.change(within(modal).getByLabelText("Definition"), {
      target: { value: "The cell's energy currency" }
    })

    await waitFor(() => {
      expect(applyButton).not.toBeDisabled()
    })
  })

  it("clears omitted notes and extra fields when applying a template", async () => {
    currentTemplates = [
      {
        ...template,
        notes_template: null,
        extra_template: null,
        placeholder_definitions: template.placeholder_definitions.slice(0, 2)
      }
    ]

    render(<FlashcardCreateDrawer open onClose={vi.fn()} onSuccess={vi.fn()} />)

    await openAdvancedOptions()

    fireEvent.change(screen.getByPlaceholderText("Optional hints or explanations..."), {
      target: { value: "Old extra text" }
    })
    fireEvent.change(screen.getByPlaceholderText("Internal notes (not shown during review)..."), {
      target: { value: "Old internal notes" }
    })

    const modal = await openTemplateModal()

    fireEvent.change(within(modal).getByLabelText("Term"), {
      target: { value: "ATP" }
    })
    fireEvent.change(within(modal).getByLabelText("Definition"), {
      target: { value: "The cell's energy currency" }
    })
    await waitFor(() => {
      expect(within(modal).getByRole("button", { name: "Apply" })).not.toBeDisabled()
    })
    fireEvent.click(within(modal).getByRole("button", { name: "Apply" }))

    await waitFor(() => {
      expect(screen.getByDisplayValue("What does ATP mean?")).toBeInTheDocument()
    })

    expect(screen.getByPlaceholderText("Optional hints or explanations...")).toHaveValue("")
    expect(screen.getByPlaceholderText("Internal notes (not shown during review)...")).toHaveValue("")
  })

  it("applies a basic reverse template and preserves the reverse model selection", async () => {
    currentTemplates = [reverseTemplate]

    render(<FlashcardCreateDrawer open onClose={vi.fn()} onSuccess={vi.fn()} />)

    const modal = await openTemplateModal()

    fireEvent.change(within(modal).getByLabelText("Term"), {
      target: { value: "ATP" }
    })
    fireEvent.change(within(modal).getByLabelText("Definition"), {
      target: { value: "The cell's energy currency" }
    })
    await waitFor(() => {
      expect(within(modal).getByRole("button", { name: "Apply" })).not.toBeDisabled()
    })
    fireEvent.click(within(modal).getByRole("button", { name: "Apply" }))

    await waitFor(() => {
      expect(screen.getByDisplayValue("ATP")).toBeInTheDocument()
    })

    const modelField = screen.getByText("Card model").closest(".ant-form-item")
    expect(modelField).not.toBeNull()
    expect(
      within(modelField as HTMLElement).getAllByText("Basic + Reverse (Both directions)").length
    ).toBeGreaterThan(0)
    expect(screen.getByDisplayValue("The cell's energy currency")).toBeInTheDocument()
  })

  it("applies a cloze template and enables the cloze-specific draft state", async () => {
    currentTemplates = [clozeTemplate]

    render(<FlashcardCreateDrawer open onClose={vi.fn()} onSuccess={vi.fn()} />)

    const modal = await openTemplateModal()

    fireEvent.change(within(modal).getByLabelText("Term"), {
      target: { value: "ATP" }
    })
    await waitFor(() => {
      expect(within(modal).getByRole("button", { name: "Apply" })).not.toBeDisabled()
    })
    fireEvent.click(within(modal).getByRole("button", { name: "Apply" }))

    await waitFor(() => {
      expect(screen.getByDisplayValue("{{c1::ATP}} powers the cell")).toBeInTheDocument()
    })

    const modelField = screen.getByText("Card model").closest(".ant-form-item")
    expect(modelField).not.toBeNull()
    expect(within(modelField as HTMLElement).getAllByText("Cloze (Fill in the blank)").length).toBeGreaterThan(0)
    expect(
      screen.getByText("Cloze syntax: add at least one deletion like {{c1::answer}} in Front text.")
    ).toBeInTheDocument()
  })

  it("saves the current draft as a template using the supported template payload", async () => {
    createTemplateMutateAsync.mockResolvedValue({
      ...template,
      name: "Saved from draft",
      front_template: "What does ATP mean?",
      back_template: "The cell's energy currency",
      notes_template: null
    })

    render(<FlashcardCreateDrawer open onClose={vi.fn()} onSuccess={vi.fn()} />)

    fireEvent.change(screen.getByPlaceholderText("Question or prompt..."), {
      target: { value: "What does ATP mean?" }
    })
    fireEvent.change(screen.getByPlaceholderText("Answer..."), {
      target: { value: "The cell's energy currency" }
    })

    fireEvent.click(screen.getByRole("button", { name: "Save as template" }))

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Save template" })).toBeInTheDocument()
    })

    fireEvent.change(screen.getByLabelText("Template name"), {
      target: { value: "Saved from draft" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save template" }))

    await waitFor(() => {
      expect(createTemplateMutateAsync).toHaveBeenCalledWith({
        name: "Saved from draft",
        model_type: "basic",
        front_template: "What does ATP mean?",
        back_template: "The cell's energy currency",
        notes_template: null,
        extra_template: null,
        placeholder_definitions: []
      })
    })
  })

  it("keeps entered template values when the template query refetches while the modal is open", async () => {
    const { rerender } = render(<FlashcardCreateDrawer open onClose={vi.fn()} onSuccess={vi.fn()} />)

    fireEvent.click(screen.getByRole("button", { name: "Apply template" }))

    await waitFor(() => {
      expect(screen.getByLabelText("Term")).toBeInTheDocument()
    })

    fireEvent.change(screen.getByLabelText("Term"), {
      target: { value: "ATP" }
    })

    currentTemplates = [...currentTemplates]
    rerender(<FlashcardCreateDrawer open onClose={vi.fn()} onSuccess={vi.fn()} />)

    expect(screen.getByLabelText("Term")).toHaveValue("ATP")
  })

  it("shows an error and keeps the save modal open when template creation fails", async () => {
    createTemplateMutateAsync.mockRejectedValueOnce(new Error("save failed"))

    render(<FlashcardCreateDrawer open onClose={vi.fn()} onSuccess={vi.fn()} />)

    fireEvent.change(screen.getByPlaceholderText("Question or prompt..."), {
      target: { value: "What does ATP mean?" }
    })
    fireEvent.change(screen.getByPlaceholderText("Answer..."), {
      target: { value: "The cell's energy currency" }
    })

    fireEvent.click(screen.getByRole("button", { name: "Save as template" }))

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Save template" })).toBeInTheDocument()
    })

    fireEvent.change(screen.getByLabelText("Template name"), {
      target: { value: "Broken draft save" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save template" }))

    await waitFor(() => {
      expect(messageApi.error).toHaveBeenCalledWith("save failed")
    })
    expect(screen.getByRole("button", { name: "Save template" })).toBeInTheDocument()
  })
})
