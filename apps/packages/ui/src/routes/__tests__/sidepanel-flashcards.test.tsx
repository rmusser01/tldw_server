import React from "react"
import userEvent from "@testing-library/user-event"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import SidepanelFlashcards, {
  readSelectedTextFromPage
} from "../sidepanel-flashcards"

type TemplateFixture = {
  id: number
  name: string
  model_type: "basic" | "basic_reverse" | "cloze"
  front_template: string
  back_template?: string | null
  notes_template?: string | null
  extra_template?: string | null
  placeholder_definitions: Array<{
    key: string
    label: string
    help_text?: string | null
    default_value?: string | null
    required?: boolean
    targets: string[]
  }>
  created_at?: string | null
  last_modified?: string | null
  deleted: boolean
  client_id: string
  version: number
}

const flashcardMocks = vi.hoisted(() => ({
  createFlashcardMutateAsync: vi.fn(),
  generateFlashcardsMutateAsync: vi.fn(),
  useFlashcardsEnabled: vi.fn(),
  useDecksQuery: vi.fn(),
  useFlashcardTemplatesQuery: vi.fn()
}))

const browserMocks = vi.hoisted(() => ({
  runtimeGetURL: vi.fn((path: string) => `chrome-extension://flashcards${path}`),
  tabsCreate: vi.fn(async () => undefined),
  tabsQuery: vi.fn(async () => [
    {
      id: 42,
      title: "Selection Source",
      url: "https://example.test/source"
    }
  ]),
  executeScript: vi.fn(async () => [
    {
      result: "Key concept from the active page"
    }
  ])
}))

const translationMocks = vi.hoisted(() => ({
  t: vi.fn(
    (
      key: string,
      fallbackOrOptions?:
        | string
        | Record<string, string | number | undefined>
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        const template = String(fallbackOrOptions.defaultValue || key)
        return template.replace(/{{(\w+)}}/g, (_, token: string) =>
          String(fallbackOrOptions[token] ?? "")
        )
      }
      return key
    }
  )
}))

vi.mock("@/components/Flashcards/hooks", () => ({
  useCreateFlashcardMutation: () => ({
    mutateAsync: flashcardMocks.createFlashcardMutateAsync,
    isPending: false
  }),
  useGenerateFlashcardsMutation: () => ({
    mutateAsync: flashcardMocks.generateFlashcardsMutateAsync,
    isPending: false
  }),
  useFlashcardsEnabled: () => flashcardMocks.useFlashcardsEnabled(),
  useDecksQuery: (...args: unknown[]) => flashcardMocks.useDecksQuery(...args),
  useFlashcardTemplatesQuery: (...args: unknown[]) =>
    flashcardMocks.useFlashcardTemplatesQuery(...args)
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      getURL: browserMocks.runtimeGetURL
    },
    tabs: {
      create: browserMocks.tabsCreate,
      query: browserMocks.tabsQuery
    },
    scripting: {
      executeScript: browserMocks.executeScript
    }
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: translationMocks.t
  })
}))

describe("sidepanel flashcards route", () => {
  let currentTemplates: TemplateFixture[]

  beforeEach(() => {
    vi.clearAllMocks()
    translationMocks.t.mockClear()
    browserMocks.runtimeGetURL.mockReset()
    browserMocks.tabsCreate.mockReset()
    browserMocks.tabsQuery.mockReset()
    browserMocks.executeScript.mockReset()
    flashcardMocks.createFlashcardMutateAsync.mockReset()
    flashcardMocks.generateFlashcardsMutateAsync.mockReset()
    flashcardMocks.useFlashcardTemplatesQuery.mockReset()
    currentTemplates = [
      {
        id: 21,
        name: "Source vocabulary",
        model_type: "basic_reverse",
        front_template: "What does {{term}} mean?",
        back_template: "{{definition}}",
        notes_template: "From {{source}}",
        extra_template: "Review after reading.",
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
            key: "source",
            label: "Source",
            help_text: null,
            default_value: "selected page",
            required: false,
            targets: ["notes_template"]
          }
        ],
        created_at: "2026-05-27T00:00:00Z",
        last_modified: "2026-05-27T00:00:00Z",
        deleted: false,
        client_id: "test-client",
        version: 1
      }
    ]
    browserMocks.runtimeGetURL.mockImplementation(
      (path: string) => `chrome-extension://flashcards${path}`
    )
    browserMocks.tabsCreate.mockResolvedValue(undefined)
    browserMocks.tabsQuery.mockResolvedValue([
      {
        id: 42,
        title: "Selection Source",
        url: "https://example.test/source"
      }
    ])
    browserMocks.executeScript.mockResolvedValue([
      {
        result: "Key concept from the active page"
      }
    ])
    flashcardMocks.createFlashcardMutateAsync.mockResolvedValue({
      uuid: "card-1",
      deck_id: 7
    })
    flashcardMocks.generateFlashcardsMutateAsync.mockResolvedValue({
      flashcards: [
        {
          front: "What does ATP do?",
          back: "ATP stores and transfers cellular energy.",
          tags: ["biology", "energy"],
          model_type: "basic",
          notes: "Generated from selected source text",
          extra: "Review with the source page open"
        },
        {
          front: "What is {{c1::mitochondrial respiration}}?",
          back: "A process that produces ATP from nutrients.",
          tags: ["biology"],
          model_type: "cloze"
        }
      ]
    })
    flashcardMocks.useFlashcardsEnabled.mockReturnValue({
      isOnline: true,
      capsLoading: false,
      flashcardsUnsupported: false,
      flashcardsEnabled: true
    })
    flashcardMocks.useDecksQuery.mockReturnValue({
      data: [
        {
          id: 7,
          name: "Biology",
          description: null,
          deleted: false,
          client_id: "test",
          version: 1
        }
      ],
      isLoading: false,
      isError: false
    })
    flashcardMocks.useFlashcardTemplatesQuery.mockImplementation(() => ({
      data: {
        items: currentTemplates,
        count: currentTemplates.length,
        total: currentTemplates.length
      },
      isLoading: false,
      error: null
    }))
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("shows explicit full-workspace and selected-text capture actions without auto-opening a tab", () => {
    render(<SidepanelFlashcards />)

    expect(
      screen.getByRole("heading", { name: "Flashcards" })
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        "Capture selected page text into a deck, or open the full Flashcards workspace."
      )
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Open full Flashcards" })
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        "Create one editable card, generate a small draft batch, apply templates to queued drafts, or use full Flashcards for imports, review, and management."
      )
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Capture page selection" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Generate from selection" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Generate draft cards" })
    ).toBeInTheDocument()
    expect(browserMocks.tabsCreate).not.toHaveBeenCalled()
  })

  it("opens the full Flashcards workspace when the user chooses that action", async () => {
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Open full Flashcards" })
    )

    expect(browserMocks.runtimeGetURL).toHaveBeenCalledWith(
      "/options.html#/flashcards"
    )
    expect(browserMocks.tabsCreate).toHaveBeenCalledWith({
      url: "chrome-extension://flashcards/options.html#/flashcards"
    })
  })

  it("shows an inline error when tab and fallback window opening fail", async () => {
    browserMocks.tabsCreate.mockRejectedValueOnce(new Error("tab open failed"))
    vi.spyOn(window, "open").mockReturnValue(null)
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Open full Flashcards" })
    )

    expect(window.open).toHaveBeenCalledWith(
      "chrome-extension://flashcards/options.html#/flashcards",
      "_blank"
    )
    expect(
      await screen.findByText(
        "Could not open Flashcards. Check popup permissions and try again."
      )
    ).toBeInTheDocument()
  })

  it("reads selected text from focused form controls in the injected page helper", () => {
    const input = document.createElement("input")
    input.value = "Before selected phrase after"
    document.body.append(input)

    try {
      input.focus()
      input.setSelectionRange(7, 22)

      expect(readSelectedTextFromPage()).toBe("selected phrase")
    } finally {
      input.remove()
    }
  })

  it("captures the active page selection into an editable sidepanel draft", async () => {
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )

    expect(browserMocks.tabsQuery).toHaveBeenCalledWith({
      active: true,
      currentWindow: true
    })
    expect(browserMocks.executeScript).toHaveBeenCalledWith({
      target: { tabId: 42 },
      func: expect.any(Function)
    })
    expect(browserMocks.tabsCreate).not.toHaveBeenCalled()
    expect(
      screen.getByRole("heading", { name: "Draft flashcard" })
    ).toBeInTheDocument()
    expect(screen.getByTestId("sidepanel-flashcards-deck-select")).toHaveTextContent(
      "Biology"
    )
    expect(screen.getByLabelText("Front")).toHaveValue("Selection Source")
    expect(screen.getByLabelText("Back")).toHaveValue(
      "Key concept from the active page"
    )
  })

  const openTemplateModal = async (buttonName: string | RegExp) => {
    await userEvent.setup().click(
      screen.getByRole("button", { name: buttonName })
    )

    return await waitFor(() => {
      const dialog = screen.getAllByRole("dialog").find((candidate) =>
        within(candidate).queryByLabelText("Template")
      )
      expect(dialog).toBeDefined()
      return dialog as HTMLElement
    })
  }

  const fillAndApplyTemplate = async (
    dialog: HTMLElement,
    values: {
      term: string
      definition: string
    }
  ) => {
    fireEvent.change(within(dialog).getByLabelText("Term"), {
      target: { value: values.term }
    })
    fireEvent.change(within(dialog).getByLabelText("Definition"), {
      target: { value: values.definition }
    })

    await waitFor(() => {
      expect(within(dialog).getByRole("button", { name: "Apply" })).not.toBeDisabled()
    })
    fireEvent.click(within(dialog).getByRole("button", { name: "Apply" }))
  }

  it("describes native template application in the sidepanel hint", () => {
    render(<SidepanelFlashcards />)

    expect(
      screen.getByText(/apply templates to queued drafts/)
    ).toBeInTheDocument()
    expect(
      screen.queryByText(/for templates, imports, and review/)
    ).not.toBeInTheDocument()
  })

  it("applies an existing template to a captured sidepanel draft before saving", async () => {
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )
    const modal = await openTemplateModal("Apply template to draft 1")
    await fillAndApplyTemplate(modal, {
      term: "ATP",
      definition: "The cell's energy currency"
    })

    await waitFor(() => {
      expect(screen.getByLabelText("Front")).toHaveValue("What does ATP mean?")
    })
    expect(screen.getByLabelText("Back")).toHaveValue("The cell's energy currency")
    await user.click(screen.getByRole("button", { name: "Save card" }))

    await waitFor(() => {
      expect(flashcardMocks.createFlashcardMutateAsync).toHaveBeenCalledTimes(1)
    })
    expect(flashcardMocks.createFlashcardMutateAsync).toHaveBeenCalledWith({
      deck_id: 7,
      front: "What does ATP mean?",
      back: "The cell's energy currency",
      notes: "From selected page",
      extra: "Review after reading.",
      model_type: "basic_reverse",
      is_cloze: false,
      reverse: true,
      source_ref_type: "manual",
      source_ref_id: "https://example.test/source"
    })
  })

  it("applies a template only to the selected queued draft", async () => {
    browserMocks.executeScript
      .mockResolvedValueOnce([{ result: "First captured concept" }])
      .mockResolvedValueOnce([{ result: "Second captured concept" }])
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )
    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )
    const modal = await openTemplateModal("Apply template to draft 2")
    await fillAndApplyTemplate(modal, {
      term: "Respiration",
      definition: "ATP production from nutrients"
    })

    await waitFor(() => {
      expect(screen.getAllByLabelText("Back")[1]).toHaveValue(
        "ATP production from nutrients"
      )
    })
    expect(screen.getAllByLabelText("Back")[0]).toHaveValue("First captured concept")
  })

  it("applies templates to generated drafts without losing generated tags", async () => {
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Generate draft cards" })
    )
    const modal = await openTemplateModal("Apply template to draft 1")
    await fillAndApplyTemplate(modal, {
      term: "ATP",
      definition: "The cell's energy currency"
    })
    await user.click(screen.getAllByRole("button", { name: "Save card" })[0])

    await waitFor(() => {
      expect(flashcardMocks.createFlashcardMutateAsync).toHaveBeenCalledTimes(1)
    })
    expect(flashcardMocks.createFlashcardMutateAsync).toHaveBeenCalledWith({
      deck_id: 7,
      front: "What does ATP mean?",
      back: "The cell's energy currency",
      tags: ["biology", "energy"],
      notes: "From selected page",
      extra: "Review after reading.",
      model_type: "basic_reverse",
      is_cloze: false,
      reverse: true,
      source_ref_type: "manual",
      source_ref_id: "https://example.test/source"
    })
  })

  it("preserves generated notes and extra when a template omits those fields", async () => {
    currentTemplates = [
      {
        ...currentTemplates[0],
        id: 22,
        name: "Sparse source vocabulary",
        notes_template: null,
        extra_template: null,
        placeholder_definitions: currentTemplates[0].placeholder_definitions.filter(
          (definition) => definition.key !== "source"
        )
      }
    ]
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Generate draft cards" })
    )
    const modal = await openTemplateModal("Apply template to draft 1")
    await fillAndApplyTemplate(modal, {
      term: "ATP",
      definition: "The cell's energy currency"
    })
    await user.click(screen.getAllByRole("button", { name: "Save card" })[0])

    await waitFor(() => {
      expect(flashcardMocks.createFlashcardMutateAsync).toHaveBeenCalledTimes(1)
    })
    expect(flashcardMocks.createFlashcardMutateAsync).toHaveBeenCalledWith({
      deck_id: 7,
      front: "What does ATP mean?",
      back: "The cell's energy currency",
      tags: ["biology", "energy"],
      notes: "Generated from selected source text",
      extra: "Review with the source page open",
      model_type: "basic_reverse",
      is_cloze: false,
      reverse: true,
      source_ref_type: "manual",
      source_ref_id: "https://example.test/source"
    })
  })

  it("keeps queued drafts visible when no templates exist", async () => {
    currentTemplates = []
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )
    await user.click(
      screen.getByRole("button", { name: "Apply template to draft 1" })
    )

    expect(await screen.findByText("No templates yet")).toBeInTheDocument()
    expect(
      screen.getByRole("heading", { name: "Draft flashcard" })
    ).toBeInTheDocument()
    expect(screen.getByLabelText("Back")).toHaveValue(
      "Key concept from the active page"
    )
  })

  it("opens full Flashcards generation with selected page text and source context", async () => {
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Generate from selection" })
    )

    const generatePath = browserMocks.runtimeGetURL.mock.calls
      .map(([path]) => path)
      .find((path) => path.includes("generate=1"))
    expect(generatePath).toBeTruthy()
    expect(generatePath).toContain("/options.html#/flashcards?")

    const query = generatePath?.slice(generatePath.indexOf("?") + 1) ?? ""
    const params = new URLSearchParams(query)
    expect(params.get("tab")).toBe("importExport")
    expect(params.get("generate")).toBe("1")
    expect(params.get("generate_text")).toBe("Key concept from the active page")
    expect(params.get("generate_source_id")).toBe("https://example.test/source")
    expect(params.get("generate_source_title")).toBe("Selection Source")
    expect(browserMocks.tabsCreate).toHaveBeenCalledWith({
      url: `chrome-extension://flashcards${generatePath}`
    })
    expect(
      screen.queryByRole("heading", { name: "Draft flashcard" })
    ).not.toBeInTheDocument()
  })

  it("generates selected page text into editable sidepanel drafts", async () => {
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Generate draft cards" })
    )

    await waitFor(() => {
      expect(flashcardMocks.generateFlashcardsMutateAsync).toHaveBeenCalledTimes(1)
    })
    expect(flashcardMocks.generateFlashcardsMutateAsync).toHaveBeenCalledWith({
      text: "Key concept from the active page",
      numCards: 3,
      cardType: "basic",
      difficulty: "mixed"
    })
    expect(browserMocks.tabsCreate).not.toHaveBeenCalled()
    expect(screen.getByText("Generated 2 draft cards.")).toBeInTheDocument()
    expect(screen.getByText("2 draft cards ready")).toBeInTheDocument()
    expect(screen.getAllByLabelText("Front")[0]).toHaveValue(
      "What does ATP do?"
    )
    expect(screen.getAllByLabelText("Back")[0]).toHaveValue(
      "ATP stores and transfers cellular energy."
    )
    expect(screen.getAllByLabelText("Front")[1]).toHaveValue(
      "What is {{c1::mitochondrial respiration}}?"
    )

    await user.click(screen.getAllByRole("button", { name: "Save card" })[0])

    await waitFor(() => {
      expect(flashcardMocks.createFlashcardMutateAsync).toHaveBeenCalledTimes(1)
    })
    expect(flashcardMocks.createFlashcardMutateAsync).toHaveBeenCalledWith({
      deck_id: 7,
      front: "What does ATP do?",
      back: "ATP stores and transfers cellular energy.",
      tags: ["biology", "energy"],
      notes: "Generated from selected source text",
      extra: "Review with the source page open",
      model_type: "basic",
      is_cloze: false,
      reverse: false,
      source_ref_type: "manual",
      source_ref_id: "https://example.test/source"
    })

    await user.click(screen.getByRole("button", { name: "Save card" }))

    await waitFor(() => {
      expect(flashcardMocks.createFlashcardMutateAsync).toHaveBeenCalledTimes(2)
    })
    expect(flashcardMocks.createFlashcardMutateAsync).toHaveBeenLastCalledWith({
      deck_id: 7,
      front: "What is {{c1::mitochondrial respiration}}?",
      back: "A process that produces ATP from nutrients.",
      tags: ["biology"],
      model_type: "cloze",
      is_cloze: true,
      reverse: false,
      source_ref_type: "manual",
      source_ref_id: "https://example.test/source"
    })
  })

  it("ignores duplicate native generation clicks before disabled state rerenders", async () => {
    flashcardMocks.generateFlashcardsMutateAsync.mockImplementation(
      () => new Promise(() => undefined)
    )
    render(<SidepanelFlashcards />)
    const generateDraftCardsButton = screen.getByRole("button", {
      name: "Generate draft cards"
    })

    await act(async () => {
      fireEvent.click(generateDraftCardsButton)
      fireEvent.click(generateDraftCardsButton)
    })

    await waitFor(() => {
      expect(flashcardMocks.generateFlashcardsMutateAsync).toHaveBeenCalledTimes(1)
    })
  })

  it("blocks the full-workspace generation handoff while native generation is in flight", async () => {
    flashcardMocks.generateFlashcardsMutateAsync.mockImplementation(
      () => new Promise(() => undefined)
    )
    render(<SidepanelFlashcards />)
    const generateDraftCardsButton = screen.getByRole("button", {
      name: "Generate draft cards"
    })
    const generateFromSelectionButton = screen.getByRole("button", {
      name: "Generate from selection"
    })

    await act(async () => {
      fireEvent.click(generateDraftCardsButton)
      fireEvent.click(generateFromSelectionButton)
    })

    await waitFor(() => {
      expect(flashcardMocks.generateFlashcardsMutateAsync).toHaveBeenCalledTimes(1)
    })
    expect(browserMocks.tabsCreate).not.toHaveBeenCalled()
  })

  it("shows empty-generation guidance when the generation response has no flashcards", async () => {
    flashcardMocks.generateFlashcardsMutateAsync.mockResolvedValueOnce(undefined)
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Generate draft cards" })
    )

    expect(
      await screen.findByText(
        "No draft cards were generated. Try selecting a longer passage."
      )
    ).toBeInTheDocument()
    expect(
      screen.queryByText(/Could not generate draft cards/)
    ).not.toBeInTheDocument()
  })

  it("uses localized whole-message copy for generated draft counts", async () => {
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Generate draft cards" })
    )

    expect(await screen.findByText("Generated 2 draft cards.")).toBeInTheDocument()
    const generatedStatusCalls = translationMocks.t.mock.calls.filter(
      ([key]) => String(key).startsWith("sidepanel:flashcards.generateDraftsSuccess")
    )
    expect(generatedStatusCalls).toContainEqual([
      "sidepanel:flashcards.generateDraftsSuccess_other",
      expect.objectContaining({
        count: 2,
        defaultValue: "Generated {{count}} draft cards."
      })
    ])
    for (const [, options] of generatedStatusCalls) {
      expect(options).not.toHaveProperty("cardLabel")
    }
  })

  it("appends repeated page selections into an editable draft queue", async () => {
    browserMocks.executeScript
      .mockResolvedValueOnce([{ result: "First captured concept" }])
      .mockResolvedValueOnce([{ result: "Second captured concept" }])
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )
    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )

    expect(screen.getByText("2 draft cards ready")).toBeInTheDocument()
    expect(
      screen.getAllByRole("heading", { name: "Draft flashcard" })
    ).toHaveLength(2)
    expect(screen.getAllByLabelText("Front")[0]).toHaveValue(
      "Selection Source"
    )
    expect(screen.getAllByLabelText("Back")[0]).toHaveValue(
      "First captured concept"
    )
    expect(screen.getAllByLabelText("Back")[1]).toHaveValue(
      "Second captured concept"
    )
  })

  it("edits one queued draft and removes another draft", async () => {
    browserMocks.executeScript
      .mockResolvedValueOnce([{ result: "Keep this answer" }])
      .mockResolvedValueOnce([{ result: "Remove this answer" }])
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )
    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )
    fireEvent.change(screen.getAllByLabelText("Front")[0], {
      target: { value: "Edited retained question" }
    })
    await user.click(screen.getByRole("button", { name: "Remove draft 2" }))

    expect(
      screen.getAllByRole("heading", { name: "Draft flashcard" })
    ).toHaveLength(1)
    expect(screen.getByLabelText("Front")).toHaveValue(
      "Edited retained question"
    )
    expect(screen.getByLabelText("Back")).toHaveValue("Keep this answer")
    expect(
      screen.queryByDisplayValue("Remove this answer")
    ).not.toBeInTheDocument()
  })

  it("saves the edited sidepanel draft with deck and page provenance", async () => {
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )
    fireEvent.change(screen.getByLabelText("Front"), {
      target: { value: "Edited question" }
    })
    fireEvent.change(screen.getByLabelText("Back"), {
      target: { value: "Edited answer" }
    })
    await user.click(screen.getByRole("button", { name: "Save card" }))

    await waitFor(() => {
      expect(flashcardMocks.createFlashcardMutateAsync).toHaveBeenCalledTimes(1)
    })
    expect(flashcardMocks.createFlashcardMutateAsync).toHaveBeenCalledWith({
      deck_id: 7,
      front: "Edited question",
      back: "Edited answer",
      model_type: "basic",
      is_cloze: false,
      reverse: false,
      source_ref_type: "manual",
      source_ref_id: "https://example.test/source"
    })
    expect(screen.getByText("Saved to Biology.")).toBeInTheDocument()
    expect(
      screen.queryByRole("heading", { name: "Draft flashcard" })
    ).not.toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Open full Flashcards" })
    ).toBeInTheDocument()
  })

  it("saves one queued draft without discarding unsaved drafts", async () => {
    browserMocks.executeScript
      .mockResolvedValueOnce([{ result: "First queued answer" }])
      .mockResolvedValueOnce([{ result: "Second queued answer" }])
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )
    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )
    await user.click(screen.getAllByRole("button", { name: "Save card" })[0])

    await waitFor(() => {
      expect(flashcardMocks.createFlashcardMutateAsync).toHaveBeenCalledTimes(1)
    })
    expect(flashcardMocks.createFlashcardMutateAsync).toHaveBeenCalledWith({
      deck_id: 7,
      front: "Selection Source",
      back: "First queued answer",
      model_type: "basic",
      is_cloze: false,
      reverse: false,
      source_ref_type: "manual",
      source_ref_id: "https://example.test/source"
    })
    expect(screen.getByText("Saved to Biology.")).toBeInTheDocument()
    expect(
      screen.queryByDisplayValue("First queued answer")
    ).not.toBeInTheDocument()
    expect(screen.getByDisplayValue("Second queued answer")).toBeInTheDocument()
  })

  it("ignores duplicate individual save clicks while the draft is saving", async () => {
    flashcardMocks.createFlashcardMutateAsync.mockImplementation(
      () => new Promise(() => undefined)
    )
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )
    const saveButton = screen.getByRole("button", { name: "Save card" })

    fireEvent.click(saveButton)
    fireEvent.click(saveButton)

    expect(flashcardMocks.createFlashcardMutateAsync).toHaveBeenCalledTimes(1)
  })

  it("preserves failed drafts after save-all partial failure", async () => {
    browserMocks.executeScript
      .mockResolvedValueOnce([{ result: "Saved answer" }])
      .mockResolvedValueOnce([{ result: "Failed answer" }])
    flashcardMocks.createFlashcardMutateAsync
      .mockResolvedValueOnce({ uuid: "card-1", deck_id: 7 })
      .mockRejectedValueOnce(new Error("save failed"))
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )
    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )
    await user.click(screen.getByRole("button", { name: "Save all cards" }))

    await waitFor(() => {
      expect(flashcardMocks.createFlashcardMutateAsync).toHaveBeenCalledTimes(2)
    })
    expect(
      screen.getByText("Saved 1 card to Biology. 1 draft still needs attention.")
    ).toBeInTheDocument()
    expect(screen.queryByDisplayValue("Saved answer")).not.toBeInTheDocument()
    expect(screen.getByDisplayValue("Failed answer")).toBeInTheDocument()
  })

  it("ignores duplicate save-all clicks while the queue is saving", async () => {
    browserMocks.executeScript
      .mockResolvedValueOnce([{ result: "First queued answer" }])
      .mockResolvedValueOnce([{ result: "Second queued answer" }])
    flashcardMocks.createFlashcardMutateAsync.mockImplementation(
      () => new Promise(() => undefined)
    )
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )
    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )
    const saveAllButton = screen.getByRole("button", {
      name: "Save all cards"
    })

    fireEvent.click(saveAllButton)
    fireEvent.click(saveAllButton)

    expect(flashcardMocks.createFlashcardMutateAsync).toHaveBeenCalledTimes(1)
  })

  it("locks draft editing and queue controls while a save is in progress", async () => {
    flashcardMocks.createFlashcardMutateAsync.mockImplementation(
      () => new Promise(() => undefined)
    )
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )
    fireEvent.click(screen.getByRole("button", { name: "Save card" }))

    await waitFor(() => {
      expect(screen.getByLabelText("Front")).toBeDisabled()
      expect(screen.getByLabelText("Back")).toBeDisabled()
      expect(
        screen.getByRole("button", { name: "Capture page selection" })
      ).toBeDisabled()
      expect(screen.getByTestId("sidepanel-flashcards-deck-select")).toHaveClass(
        "ant-select-disabled"
      )
    })
  })

  it("keeps the draft in place when sidepanel save fails", async () => {
    flashcardMocks.createFlashcardMutateAsync.mockRejectedValueOnce(
      new Error("save failed")
    )
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )
    await user.click(screen.getByRole("button", { name: "Save card" }))

    expect(
      await screen.findByText("Could not save flashcard. Check your connection and try again.")
    ).toBeInTheDocument()
    expect(screen.getByLabelText("Back")).toHaveValue(
      "Key concept from the active page"
    )
  })

  it("requires a deck before saving a sidepanel draft", async () => {
    flashcardMocks.useDecksQuery.mockReturnValue({
      data: [],
      isLoading: false,
      isError: false
    })
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )

    expect(
      screen.getByText("Create a deck in full Flashcards before saving here.")
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save card" })).toBeDisabled()
  })

  it("does not show true-empty deck guidance while decks are still loading", async () => {
    flashcardMocks.useDecksQuery.mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false
    })
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )

    expect(
      screen.queryByText("Create a deck in full Flashcards before saving here.")
    ).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save card" })).toBeDisabled()
  })

  it("differentiates deck load errors from true empty deck state", async () => {
    flashcardMocks.useDecksQuery.mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true
    })
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )

    expect(
      screen.getByText("Could not load decks. Check your connection and try again.")
    ).toBeInTheDocument()
    expect(
      screen.queryByText("Create a deck in full Flashcards before saving here.")
    ).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save card" })).toBeDisabled()
  })

  it("explains when flashcards are unavailable instead of saying there are no decks", async () => {
    flashcardMocks.useFlashcardsEnabled.mockReturnValue({
      isOnline: false,
      capsLoading: false,
      flashcardsUnsupported: false,
      flashcardsEnabled: false
    })
    flashcardMocks.useDecksQuery.mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: false
    })
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )

    expect(
      screen.getByText("Flashcards are unavailable. Check the server connection and try again.")
    ).toBeInTheDocument()
    expect(
      screen.queryByText("Create a deck in full Flashcards before saving here.")
    ).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save card" })).toBeDisabled()
  })

  it("keeps queued drafts when a later capture attempt fails", async () => {
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )
    expect(
      screen.getByRole("heading", { name: "Draft flashcard" })
    ).toBeInTheDocument()

    browserMocks.executeScript.mockResolvedValueOnce([{ result: "" }])
    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )

    expect(
      screen.getByText("Select text on the page first.")
    ).toBeInTheDocument()
    expect(
      screen.getByRole("heading", { name: "Draft flashcard" })
    ).toBeInTheDocument()
    expect(screen.getByLabelText("Back")).toHaveValue(
      "Key concept from the active page"
    )
  })

  it("keeps queued drafts when generate-from-selection capture fails", async () => {
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )
    expect(
      screen.getByRole("heading", { name: "Draft flashcard" })
    ).toBeInTheDocument()

    browserMocks.executeScript.mockResolvedValueOnce([{ result: "" }])
    await user.click(
      screen.getByRole("button", { name: "Generate from selection" })
    )

    expect(
      screen.getByText("Select text on the page first.")
    ).toBeInTheDocument()
    expect(
      screen.getByRole("heading", { name: "Draft flashcard" })
    ).toBeInTheDocument()
    expect(screen.getByLabelText("Back")).toHaveValue(
      "Key concept from the active page"
    )
  })

  it("keeps queued drafts when native generation fails", async () => {
    flashcardMocks.generateFlashcardsMutateAsync.mockRejectedValueOnce(
      new Error("Generation failed")
    )
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )
    expect(
      screen.getByRole("heading", { name: "Draft flashcard" })
    ).toBeInTheDocument()

    await user.click(
      screen.getByRole("button", { name: "Generate draft cards" })
    )

    expect(
      await screen.findByText(
        "Could not generate draft cards. Generation failed"
      )
    ).toBeInTheDocument()
    expect(
      screen.getByRole("heading", { name: "Draft flashcard" })
    ).toBeInTheDocument()
    expect(screen.getByLabelText("Back")).toHaveValue(
      "Key concept from the active page"
    )
  })

  it("uses capture wording when no active page tab is available", async () => {
    browserMocks.tabsQuery.mockResolvedValueOnce([])
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )

    expect(
      screen.getByText("Open a page tab before capturing page selection.")
    ).toBeInTheDocument()
  })

  it("keeps the user in place when no page text is selected", async () => {
    browserMocks.executeScript.mockResolvedValueOnce([{ result: "" }])
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )

    expect(
      screen.getByText("Select text on the page first.")
    ).toBeInTheDocument()
    expect(browserMocks.tabsCreate).not.toHaveBeenCalled()
  })
})
