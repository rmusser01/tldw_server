import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { TemplatesTab } from "../TemplatesTab"
import {
  useCreateFlashcardTemplateMutation,
  useDeleteFlashcardTemplateMutation,
  useFlashcardTemplatesQuery,
  useUpdateFlashcardTemplateMutation
} from "../../hooks"
import type { FlashcardTemplate } from "@/services/flashcards"

const messageSpies = {
  success: vi.fn(),
  error: vi.fn(),
  info: vi.fn(),
  warning: vi.fn(),
  loading: vi.fn(),
  open: vi.fn(),
  destroy: vi.fn()
}

const createMutateAsync = vi.hoisted(() => vi.fn())
const updateMutateAsync = vi.hoisted(() => vi.fn())
const deleteMutateAsync = vi.hoisted(() => vi.fn())

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => messageSpies
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

vi.mock("../../hooks", () => ({
  useFlashcardTemplatesQuery: vi.fn(),
  useCreateFlashcardTemplateMutation: vi.fn(),
  useUpdateFlashcardTemplateMutation: vi.fn(),
  useDeleteFlashcardTemplateMutation: vi.fn()
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

if (!(Element.prototype as any).scrollIntoView) {
  ;(Element.prototype as any).scrollIntoView = vi.fn()
}

const buildTemplate = (overrides: Partial<FlashcardTemplate> = {}): FlashcardTemplate => ({
  id: 1,
  name: "Basic facts",
  model_type: "basic",
  front_template: "Question: {{prompt}}",
  back_template: "Answer: {{answer}}",
  notes_template: null,
  extra_template: null,
  placeholder_definitions: [
    {
      key: "prompt",
      label: "Prompt",
      help_text: "Main cue",
      default_value: null,
      required: true,
      targets: ["front_template"]
    },
    {
      key: "answer",
      label: "Answer",
      help_text: "Expected response",
      default_value: null,
      required: true,
      targets: ["back_template"]
    }
  ],
  created_at: "2026-04-15T00:00:00Z",
  last_modified: "2026-04-15T00:00:00Z",
  deleted: false,
  client_id: "test-client",
  version: 3,
  ...overrides
})

describe("TemplatesTab", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  beforeEach(() => {
    vi.restoreAllMocks()
    createMutateAsync.mockReset()
    updateMutateAsync.mockReset()
    deleteMutateAsync.mockReset()
    vi.clearAllMocks()

    vi.mocked(useFlashcardTemplatesQuery).mockReturnValue({
      data: { items: [], count: 0, total: 0 },
      isLoading: false,
      error: null
    } as any)
    vi.mocked(useCreateFlashcardTemplateMutation).mockReturnValue({
      mutateAsync: createMutateAsync,
      isPending: false
    } as any)
    vi.mocked(useUpdateFlashcardTemplateMutation).mockReturnValue({
      mutateAsync: updateMutateAsync,
      isPending: false
    } as any)
    vi.mocked(useDeleteFlashcardTemplateMutation).mockReturnValue({
      mutateAsync: deleteMutateAsync,
      isPending: false
    } as any)
  })

  it("shows an empty state when no templates exist", () => {
    render(<TemplatesTab />)

    expect(screen.getByText("No templates yet")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Create template" })).toBeInTheDocument()
  })

  it("lists existing templates and opens the selected template in the editor", () => {
    vi.mocked(useFlashcardTemplatesQuery).mockReturnValue({
      data: {
        items: [
          buildTemplate(),
          buildTemplate({
            id: 2,
            name: "Reverse vocabulary",
            model_type: "basic_reverse",
            version: 1
          })
        ],
        count: 2,
        total: 2
      },
      isLoading: false,
      error: null
    } as any)

    render(<TemplatesTab />)

    expect(screen.getByText("Basic facts")).toBeInTheDocument()
    expect(screen.getByText("Reverse vocabulary")).toBeInTheDocument()
    expect(screen.getByDisplayValue("Basic facts")).toBeInTheDocument()
    expect(screen.getByDisplayValue("Question: {{prompt}}")).toBeInTheDocument()
  })

  it("opens a create form from the create action", async () => {
    vi.mocked(useFlashcardTemplatesQuery).mockReturnValue({
      data: { items: [buildTemplate()], count: 1, total: 1 },
      isLoading: false,
      error: null
    } as any)

    render(<TemplatesTab />)

    fireEvent.click(screen.getByRole("button", { name: "Create template" }))

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Save template" })).toBeInTheDocument()
    })
    expect(screen.getByLabelText("Template name")).toHaveValue("")
  })

  it("revalidates back template requirements when the card type changes to cloze", async () => {
    render(<TemplatesTab />)

    fireEvent.click(screen.getByRole("button", { name: "Create template" }))

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Save template" })).toBeInTheDocument()
    })

    fireEvent.change(screen.getByLabelText("Template name"), {
      target: { value: "Cloze scaffold" }
    })
    fireEvent.change(screen.getByLabelText("Front template"), {
      target: { value: "{{c1::{{term}}}}" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save template" }))

    await waitFor(() => {
      expect(screen.getByText("Enter a back template.")).toBeInTheDocument()
    })

    fireEvent.mouseDown(screen.getByLabelText("Card type"))
    fireEvent.click(
      await screen.findByText("Fill-in-blank", {
        selector: ".ant-select-item-option-content"
      })
    )

    await waitFor(() => {
      expect(screen.queryByText("Enter a back template.")).not.toBeInTheDocument()
    })
  })

  it("lets an empty library return from create mode without saving", async () => {
    render(<TemplatesTab />)

    fireEvent.click(screen.getByRole("button", { name: "Create template" }))

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Save template" })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }))

    await waitFor(() => {
      expect(screen.getByText("No templates yet")).toBeInTheDocument()
    })
  })

  it("keeps a newly created template selected before the list query refreshes", async () => {
    createMutateAsync.mockResolvedValue(
      buildTemplate({
        id: 17,
        name: "Created template",
        front_template: "Question: {{prompt}}",
        back_template: "Answer: {{answer}}",
        version: 1
      })
    )

    render(<TemplatesTab />)

    fireEvent.click(screen.getByRole("button", { name: "Create template" }))

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Save template" })).toBeInTheDocument()
    })

    fireEvent.change(screen.getByLabelText("Template name"), {
      target: { value: "Created template" }
    })
    fireEvent.change(screen.getByLabelText("Front template"), {
      target: { value: "Question: {{prompt}}" }
    })
    fireEvent.change(screen.getByLabelText("Back template"), {
      target: { value: "Answer: {{answer}}" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save template" }))

    await waitFor(() => {
      expect(screen.getByDisplayValue("Created template")).toBeInTheDocument()
    })
    expect(screen.getByRole("button", { name: "Save changes" })).toBeInTheDocument()
  })

  it("updates the selected template from the editor form", async () => {
    vi.mocked(useFlashcardTemplatesQuery).mockReturnValue({
      data: { items: [buildTemplate()], count: 1, total: 1 },
      isLoading: false,
      error: null
    } as any)
    updateMutateAsync.mockResolvedValue(
      buildTemplate({
        name: "Updated template"
      })
    )

    render(<TemplatesTab />)

    fireEvent.change(screen.getByLabelText("Template name"), {
      target: { value: "Updated template" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }))

    await waitFor(() => {
      expect(updateMutateAsync).toHaveBeenCalledWith({
        templateId: 1,
        update: expect.objectContaining({
          name: "Updated template",
          expected_version: 3
        })
      })
    })
  })

  it("keeps the current template editor pinned while a dirty search filters it out", () => {
    vi.mocked(useFlashcardTemplatesQuery).mockReturnValue({
      data: {
        items: [
          buildTemplate(),
          buildTemplate({
            id: 2,
            name: "Chemistry facts",
            front_template: "Element: {{prompt}}",
            back_template: "Symbol: {{answer}}",
            version: 1
          })
        ],
        count: 2,
        total: 2
      },
      isLoading: false,
      error: null
    } as any)

    render(<TemplatesTab />)

    fireEvent.change(screen.getByLabelText("Template name"), {
      target: { value: "Unsaved rename" }
    })
    fireEvent.change(screen.getByPlaceholderText("Search templates"), {
      target: { value: "Chemistry" }
    })

    expect(screen.getByDisplayValue("Unsaved rename")).toBeInTheDocument()
    expect(screen.queryByDisplayValue("Chemistry facts")).not.toBeInTheDocument()
  })

  it("clears a stale search and keeps the edited template selected after rename", async () => {
    vi.mocked(useFlashcardTemplatesQuery).mockReturnValue({
      data: { items: [buildTemplate()], count: 1, total: 1 },
      isLoading: false,
      error: null
    } as any)
    updateMutateAsync.mockResolvedValue(
      buildTemplate({
        name: "Renamed template"
      })
    )

    render(<TemplatesTab />)

    fireEvent.change(screen.getByPlaceholderText("Search templates"), {
      target: { value: "Basic" }
    })
    fireEvent.change(screen.getByLabelText("Template name"), {
      target: { value: "Renamed template" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }))

    await waitFor(() => {
      expect(updateMutateAsync).toHaveBeenCalled()
    })

    expect(screen.getByPlaceholderText("Search templates")).toHaveValue("")
    expect(screen.getByDisplayValue("Renamed template")).toBeInTheDocument()
  })

  it("disables template navigation controls while a mutation is pending", () => {
    vi.mocked(useFlashcardTemplatesQuery).mockReturnValue({
      data: {
        items: [
          buildTemplate(),
          buildTemplate({
            id: 2,
            name: "Chemistry facts",
            front_template: "Element: {{prompt}}",
            back_template: "Symbol: {{answer}}",
            version: 1
          })
        ],
        count: 2,
        total: 2
      },
      isLoading: false,
      error: null
    } as any)
    vi.mocked(useUpdateFlashcardTemplateMutation).mockReturnValue({
      mutateAsync: updateMutateAsync,
      isPending: true
    } as any)

    render(<TemplatesTab />)

    expect(screen.getByRole("button", { name: "Create template" })).toBeDisabled()
    expect(screen.getByPlaceholderText("Search templates")).toBeDisabled()
    expect(screen.getByRole("button", { name: /^Basic facts/i })).toBeDisabled()
    expect(screen.getByRole("button", { name: /^Chemistry facts/i })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Delete template" })).toBeDisabled()
  })

  it("deletes the selected template from the editor", async () => {
    vi.mocked(useFlashcardTemplatesQuery).mockReturnValue({
      data: { items: [buildTemplate()], count: 1, total: 1 },
      isLoading: false,
      error: null
    } as any)
    deleteMutateAsync.mockResolvedValue(undefined)
    vi.spyOn(window, "confirm").mockReturnValue(true)

    render(<TemplatesTab />)

    fireEvent.click(screen.getByRole("button", { name: "Delete template" }))

    await waitFor(() => {
      expect(deleteMutateAsync).toHaveBeenCalledWith({
        templateId: 1,
        expectedVersion: 3
      })
    })

  })

  it("clears a stale search and selects another template after deleting the last match", async () => {
    vi.mocked(useFlashcardTemplatesQuery).mockReturnValue({
      data: {
        items: [
          buildTemplate(),
          buildTemplate({
            id: 2,
            name: "Chemistry facts",
            front_template: "Element: {{prompt}}",
            back_template: "Symbol: {{answer}}",
            version: 1
          })
        ],
        count: 2,
        total: 2
      },
      isLoading: false,
      error: null
    } as any)
    deleteMutateAsync.mockResolvedValue(undefined)
    vi.spyOn(window, "confirm").mockReturnValue(true)

    render(<TemplatesTab />)

    fireEvent.change(screen.getByPlaceholderText("Search templates"), {
      target: { value: "Basic" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Delete template" }))

    await waitFor(() => {
      expect(deleteMutateAsync).toHaveBeenCalledWith({
        templateId: 1,
        expectedVersion: 3
      })
    })

    expect(screen.getByPlaceholderText("Search templates")).toHaveValue("")
    expect(screen.getByDisplayValue("Chemistry facts")).toBeInTheDocument()

  })

  it("blocks save when a placeholder key is not used in any targeted field", async () => {
    render(<TemplatesTab />)

    fireEvent.click(screen.getByRole("button", { name: "Create template" }))

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Save template" })).toBeInTheDocument()
    })

    fireEvent.change(screen.getByLabelText("Template name"), {
      target: { value: "Template with placeholder" }
    })
    fireEvent.change(screen.getByLabelText("Front template"), {
      target: { value: "Question without token" }
    })
    fireEvent.change(screen.getByLabelText("Back template"), {
      target: { value: "Answer body" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Add placeholder" }))
    fireEvent.change(screen.getByLabelText("Key"), {
      target: { value: "prompt" }
    })
    fireEvent.change(screen.getByLabelText("Label"), {
      target: { value: "Prompt" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save template" }))

    await waitFor(() => {
      expect(createMutateAsync).not.toHaveBeenCalled()
    })
    await waitFor(() => {
      expect(
        screen.getByText(/at least one targeted template field/i)
      ).toBeInTheDocument()
    })
  })

  it("accepts placeholder tokens with internal whitespace", async () => {
    createMutateAsync.mockResolvedValue(
      buildTemplate({
        id: 18,
        name: "Spaced token template",
        front_template: "Question: {{ prompt }}",
        back_template: "Answer body",
        placeholder_definitions: [
          {
            key: "prompt",
            label: "Prompt",
            help_text: null,
            default_value: null,
            required: true,
            targets: ["front_template"]
          }
        ],
        version: 1
      })
    )

    render(<TemplatesTab />)

    fireEvent.click(screen.getByRole("button", { name: "Create template" }))

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Save template" })).toBeInTheDocument()
    })

    fireEvent.change(screen.getByLabelText("Template name"), {
      target: { value: "Spaced token template" }
    })
    fireEvent.change(screen.getByLabelText("Front template"), {
      target: { value: "Question: {{ prompt }}" }
    })
    fireEvent.change(screen.getByLabelText("Back template"), {
      target: { value: "Answer body" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Add placeholder" }))
    fireEvent.change(screen.getByLabelText("Key"), {
      target: { value: "prompt" }
    })
    fireEvent.change(screen.getByLabelText("Label"), {
      target: { value: "Prompt" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save template" }))

    await waitFor(() => {
      expect(createMutateAsync).toHaveBeenCalledWith(
        expect.objectContaining({
          front_template: "Question: {{ prompt }}"
        })
      )
    })
  })
})
