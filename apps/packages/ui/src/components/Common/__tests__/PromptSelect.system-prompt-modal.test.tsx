import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { OPEN_PROMPT_SELECT_EVENT } from "@/utils/prompt-select-events"

const mocks = vi.hoisted(() => ({
  getAllPrompts: vi.fn(async () => []),
  getPromptById: vi.fn(async () => undefined),
  improvePrompt: vi.fn(),
  fetchPromptCapabilities: vi.fn()
}))

const registryLabels = vi.hoisted(() => ({
  loading: "Loading via registry"
}))

const commonLoadingResource = vi.hoisted(() => ({
  title: "Loading title from common",
  description: "Loading description from common",
  content: "Loading content from common"
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => {
      if (key === "common:loading") return commonLoadingResource
      if (key === "common:loading.title") return commonLoadingResource.title
      return fallback || key
    }
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) =>
    React.useState(defaultValue)
}))

vi.mock("@/db/dexie/helpers", () => ({
  getAllPrompts: mocks.getAllPrompts,
  getPromptById: mocks.getPromptById
}))

vi.mock("@/services/prompt-improvement", async (importActual) => {
  const actual =
    await importActual<typeof import("@/services/prompt-improvement")>()
  return {
    ...actual,
    improvePrompt: (...args: unknown[]) => mocks.improvePrompt(...args)
  }
})

vi.mock("@/services/prompts-api", async (importActual) => {
  const actual = await importActual<typeof import("@/services/prompts-api")>()
  return {
    ...actual,
    fetchPromptCapabilities: () => mocks.fetchPromptCapabilities()
  }
})

vi.mock("@/design-system", async (importActual) => {
  const actual = await importActual<typeof import("@/design-system")>()

  return {
    ...actual,
    getDesignSystemState: vi.fn(
      (key: Parameters<typeof actual.getDesignSystemState>[0]) => {
        const state = actual.getDesignSystemState(key)

        return {
          ...state,
          label: key === "loading" ? registryLabels.loading : state.label
        }
      }
    )
  }
})

vi.mock("antd", async () => {
  const React = await import("react")

  const Input = React.forwardRef<HTMLInputElement, any>((props, ref) => (
    <input
      ref={ref}
      aria-label={props["aria-label"] ?? props.placeholder}
      value={props.value}
      defaultValue={props.defaultValue}
      onChange={props.onChange}
      onKeyDownCapture={props.onKeyDownCapture}
      onKeyDown={props.onKeyDown}
    />
  ))

  const TextArea = React.forwardRef<HTMLTextAreaElement, any>((props, ref) => (
    <textarea
      ref={ref}
      aria-label={props["aria-label"] ?? props.placeholder ?? "System prompt"}
      value={props.value}
      defaultValue={props.defaultValue}
      onChange={props.onChange}
    />
  ))

  ;(Input as any).TextArea = TextArea

  const renderMenuItems = (items: any[] = []) =>
    items.map((item) => {
      if (!item) return null
      if (item.type === "group") {
        return (
          <div key={item.label}>
            <div>{item.label}</div>
            {renderMenuItems(item.children)}
          </div>
        )
      }
      if (item.key === "empty") {
        return <div key="empty">{item.label}</div>
      }
      return (
        <button
          key={item.key}
          type="button"
          role="menuitem"
          onClick={() => item.onClick?.()}
        >
          {item.label}
        </button>
      )
    })

  const Dropdown = ({
    open,
    onOpenChange,
    menu,
    popupRender,
    children
  }: any) => {
    const menuNode = <div role="menu">{renderMenuItems(menu?.items)}</div>

    return (
      <div>
        <div onClick={() => onOpenChange?.(!open)}>{children}</div>
        {open ? (popupRender ? popupRender(menuNode) : menuNode) : null}
      </div>
    )
  }

  const Modal = ({ open, title, children, footer }: any) =>
    open ? (
      <div role="dialog" aria-label={typeof title === "string" ? title : undefined}>
        <div>{title}</div>
        <div>{children}</div>
        <div>{footer}</div>
      </div>
    ) : null

  return {
    Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
    Dropdown,
    Empty: ({ description }: { description?: React.ReactNode }) => (
      <div>{description ?? "Empty"}</div>
    ),
    Input,
    Modal
  }
})

import { PromptSelect } from "../PromptSelect"

const buildPrompt = (overrides: Record<string, unknown> = {}) => ({
  id: "prompt-1",
  title: "Prompt One",
  content: "Template body",
  is_system: true,
  createdAt: Date.now(),
  ...overrides
})

const createDeferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((promiseResolve) => {
    resolve = promiseResolve
  })
  return { promise, resolve }
}

const improvementResponse = (
  operationId: string,
  improvedText = "Improved system draft"
) => ({
  schema_version: 1 as const,
  operation_id: operationId,
  status: "improved" as const,
  improved_text: improvedText,
  findings: [],
  review_required: false,
  warnings: [],
  resolved_model: {
    provider: "openai",
    model: "gpt-5-mini",
    display_name: "GPT-5 mini"
  },
  meta_prompt_version: "prompt-improvement-v1"
})

const renderPromptSelect = (overrides: Partial<React.ComponentProps<typeof PromptSelect>> = {}) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false
      }
    }
  })

  const props: React.ComponentProps<typeof PromptSelect> = {
    selectedSystemPrompt: "prompt-1",
    systemPrompt: "",
    setSystemPrompt: vi.fn(),
    setSelectedSystemPrompt: vi.fn(),
    setSelectedQuickPrompt: vi.fn(),
    selectedModel: "gpt-5-mini",
    currentProvider: "openai",
    promptAssistContextKey: "conversation-1",
    promptAssistBackendKey: "backend-a",
    onSelectModel: vi.fn(),
    ...overrides
  }

  return {
    ...render(
      <QueryClientProvider client={queryClient}>
        <PromptSelect {...props} />
      </QueryClientProvider>
    ),
    props,
    queryClient
  }
}

const openEditor = async (
  user: ReturnType<typeof userEvent.setup>,
  expectedValue = "Template body"
) => {
  await user.click(
    await screen.findByRole("button", { name: "selectAPrompt" })
  )
  await user.click(
    await screen.findByRole("menuitem", { name: /edit system prompt/i })
  )
  await screen.findByDisplayValue(expectedValue)
}

const applyImprovementNow = async (
  user: ReturnType<typeof userEvent.setup>
) => {
  await user.click(screen.getByRole("button", { name: "Improve prompt" }))
  await user.click(screen.getByRole("button", { name: /Improve now/ }))
  await screen.findByRole("button", { name: "Undo improvement" })
}

describe("PromptSelect system prompt modal", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.getAllPrompts.mockResolvedValue([buildPrompt()])
    mocks.getPromptById.mockResolvedValue(buildPrompt())
    mocks.fetchPromptCapabilities.mockResolvedValue({
      availability: "available",
      prompt_improvement_v1: {
        supported: true,
        limits: null
      },
      single_text_recipe_v2: { supported: false }
    })
    mocks.improvePrompt.mockImplementation(async (request) =>
      improvementResponse(request.operation_id)
    )
  })

  it("opens an editor modal with the effective selected template content", async () => {
    const user = userEvent.setup()
    renderPromptSelect()

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(await screen.findByRole("menuitem", { name: /edit system prompt/i }))

    expect(await screen.findByDisplayValue("Template body")).toBeInTheDocument()
  })

  it("keeps the prompt trigger visible while the prompt library is loading", async () => {
    mocks.getAllPrompts.mockReturnValue(new Promise(() => {}))

    renderPromptSelect({
      selectedSystemPrompt: undefined
    })

    expect(
      await screen.findByRole("button", { name: /loading prompts/i })
    ).toBeInTheDocument()
    expect(screen.getByText("Loading prompts")).toBeInTheDocument()
    expect(
      screen.queryByRole("status", { name: /loading prompts/i })
    ).not.toBeInTheDocument()
  })

  it("shows prompt library errors with a retry action", async () => {
    const user = userEvent.setup()
    mocks.getAllPrompts.mockRejectedValueOnce(new Error("dexie unavailable"))
    renderPromptSelect({
      selectedSystemPrompt: undefined
    })

    await user.click(
      await screen.findByRole("button", { name: /prompt library unavailable/i })
    )

    expect(
      await screen.findByRole("menuitem", {
        name: /prompt library unavailable/i
      })
    ).toBeInTheDocument()
    await user.click(
      screen.getByRole("menuitem", { name: /retry prompt library/i })
    )

    expect(mocks.getAllPrompts).toHaveBeenCalledTimes(2)
  })

  it("saves edited prompt content through setSystemPrompt", async () => {
    const user = userEvent.setup()
    const { props } = renderPromptSelect()

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(await screen.findByRole("menuitem", { name: /edit system prompt/i }))

    const textarea = await screen.findByDisplayValue("Template body")
    await user.clear(textarea)
    await user.type(textarea, "Conversation override")
    await user.click(screen.getByRole("button", { name: /save/i }))

    await waitFor(() => {
      expect(props.setSystemPrompt).toHaveBeenCalledWith("Conversation override")
    })
  })

  it("clears redundant overrides when the saved text matches the selected template", async () => {
    const user = userEvent.setup()
    const { props } = renderPromptSelect({
      systemPrompt: "Conversation override"
    })

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(await screen.findByRole("menuitem", { name: /edit system prompt/i }))

    const textarea = await screen.findByDisplayValue("Conversation override")
    await user.clear(textarea)
    await user.type(textarea, "Template body")
    await user.click(screen.getByRole("button", { name: /save/i }))

    await waitFor(() => {
      expect(props.setSystemPrompt).toHaveBeenCalledWith("")
    })
  })

  it("shows override-active copy when the live system prompt differs from the template", async () => {
    const user = userEvent.setup()
    renderPromptSelect({
      systemPrompt: "Conversation override"
    })

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(await screen.findByRole("menuitem", { name: /edit system prompt/i }))

    expect(await screen.findByText(/override active/i)).toBeInTheDocument()
  })

  it("resets to an empty prompt when the selected template cannot be resolved", async () => {
    const user = userEvent.setup()
    const { props } = renderPromptSelect({
      selectedSystemPrompt: "missing-prompt"
    })

    mocks.getAllPrompts.mockResolvedValue([])
    mocks.getPromptById.mockRejectedValue(new Error("missing"))

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(await screen.findByRole("menuitem", { name: /edit system prompt/i }))
    await user.click(screen.getByRole("button", { name: /reset/i }))

    await waitFor(() => {
      expect(props.setSystemPrompt).toHaveBeenCalledWith("")
    })
  })

  it("renders the scalar common loading title while resolving editor content", async () => {
    const user = userEvent.setup()
    mocks.getPromptById.mockReturnValue(new Promise(() => {}))
    renderPromptSelect()

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(await screen.findByRole("menuitem", { name: /edit system prompt/i }))

    expect(
      await screen.findByText("Loading title from common")
    ).toBeInTheDocument()
  })

  it("does not commit an editor-open lookup after the modal closes", async () => {
    const user = userEvent.setup()
    const pending = createDeferred<ReturnType<typeof buildPrompt>>()
    const nextOpen = createDeferred<ReturnType<typeof buildPrompt>>()
    mocks.getPromptById.mockReturnValueOnce(pending.promise)
    mocks.getPromptById.mockReturnValueOnce(nextOpen.promise)
    renderPromptSelect()

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )
    await user.click(screen.getByRole("button", { name: "Cancel" }))
    await act(async () => {
      pending.resolve(buildPrompt({ content: "Late open content" }))
      await pending.promise
    })

    await waitFor(() => {
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    })
    await user.click(screen.getByRole("button", { name: "selectAPrompt" }))
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )
    expect(
      screen.queryByDisplayValue("Late open content")
    ).not.toBeInTheDocument()
  })

  it("lets a newer editor-open lookup win over an older completion", async () => {
    const user = userEvent.setup()
    const older = createDeferred<ReturnType<typeof buildPrompt>>()
    const newer = createDeferred<ReturnType<typeof buildPrompt>>()
    mocks.getPromptById
      .mockReturnValueOnce(older.promise)
      .mockReturnValueOnce(newer.promise)
    renderPromptSelect()

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )
    await user.click(screen.getByRole("button", { name: "Cancel" }))
    await user.click(screen.getByRole("button", { name: "selectAPrompt" }))
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )

    await act(async () => {
      newer.resolve(buildPrompt({ content: "Newest open content" }))
      await newer.promise
    })
    expect(
      await screen.findByDisplayValue("Newest open content")
    ).toBeInTheDocument()
    await act(async () => {
      older.resolve(buildPrompt({ content: "Older open content" }))
      await older.promise
    })

    await waitFor(() => {
      expect(
        screen.getByDisplayValue("Newest open content")
      ).toBeInTheDocument()
      expect(
        screen.queryByDisplayValue("Older open content")
      ).not.toBeInTheDocument()
    })
  })

  it("does not let a late editor-open lookup overwrite typing", async () => {
    const user = userEvent.setup()
    const pending = createDeferred<ReturnType<typeof buildPrompt>>()
    mocks.getPromptById.mockReturnValueOnce(pending.promise)
    renderPromptSelect()

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )
    const editor = screen.getByRole("textbox", { name: "Enter system prompt" })
    await user.type(editor, "Typed while opening")
    await act(async () => {
      pending.resolve(buildPrompt({ content: "Late open content" }))
      await pending.promise
    })

    await waitFor(() => {
      expect(editor).toHaveValue("Typed while opening")
    })
  })

  it("does not commit a Reset lookup after close or unmount", async () => {
    const user = userEvent.setup()
    const afterClose = createDeferred<ReturnType<typeof buildPrompt>>()
    const afterUnmount = createDeferred<ReturnType<typeof buildPrompt>>()
    const first = renderPromptSelect({ systemPrompt: "Conversation override" })
    await openEditor(user, "Conversation override")
    mocks.getPromptById.mockReturnValueOnce(afterClose.promise)
    await user.click(screen.getByRole("button", { name: "Reset" }))
    await user.click(screen.getByRole("button", { name: "Cancel" }))
    await act(async () => {
      afterClose.resolve(buildPrompt({ content: "Late reset after close" }))
      await afterClose.promise
    })
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    )
    expect(first.props.setSystemPrompt).not.toHaveBeenCalled()

    first.unmount()
    const second = renderPromptSelect({ systemPrompt: "Conversation override" })
    await openEditor(user, "Conversation override")
    mocks.getPromptById.mockReturnValueOnce(afterUnmount.promise)
    await user.click(screen.getByRole("button", { name: "Reset" }))
    second.unmount()
    await act(async () => {
      afterUnmount.resolve(buildPrompt({ content: "Late reset after unmount" }))
      await afterUnmount.promise
    })

    await waitFor(() => {
      expect(second.props.setSystemPrompt).not.toHaveBeenCalled()
    })
  })

  it("lets a newer Reset win and rejects a Reset completion after typing", async () => {
    const user = userEvent.setup()
    const older = createDeferred<ReturnType<typeof buildPrompt>>()
    const newer = createDeferred<ReturnType<typeof buildPrompt>>()
    const afterTyping = createDeferred<ReturnType<typeof buildPrompt>>()
    const { props } = renderPromptSelect({
      systemPrompt: "Conversation override"
    })
    await openEditor(user, "Conversation override")
    mocks.getPromptById
      .mockReturnValueOnce(older.promise)
      .mockReturnValueOnce(newer.promise)
    await user.click(screen.getByRole("button", { name: "Reset" }))
    await user.click(screen.getByRole("button", { name: "Reset" }))
    await act(async () => {
      newer.resolve(buildPrompt({ content: "Newest reset content" }))
      await newer.promise
    })
    expect(
      await screen.findByDisplayValue("Newest reset content")
    ).toBeInTheDocument()
    await act(async () => {
      older.resolve(buildPrompt({ content: "Older reset content" }))
      await older.promise
    })
    await waitFor(() => {
      expect(
        screen.queryByDisplayValue("Older reset content")
      ).not.toBeInTheDocument()
    })

    mocks.getPromptById.mockReturnValueOnce(afterTyping.promise)
    await user.click(screen.getByRole("button", { name: "Reset" }))
    const editor = screen.getByRole("textbox", { name: "Enter system prompt" })
    await user.clear(editor)
    await user.type(editor, "Typed after reset")
    await act(async () => {
      afterTyping.resolve(buildPrompt({ content: "Late reset content" }))
      await afterTyping.promise
    })

    await waitFor(() => expect(editor).toHaveValue("Typed after reset"))
    expect(props.setSystemPrompt).toHaveBeenLastCalledWith(
      "Newest reset content"
    )
  })

  it.each([
    ["template", { selectedSystemPrompt: "prompt-2" }],
    ["model", { selectedModel: "gpt-5" }],
    ["provider", { currentProvider: "anthropic" }],
    ["context", { promptAssistContextKey: "conversation-2" }]
  ])(
    "rejects a Reset lookup after a %s lifecycle change",
    async (_change, changedProps) => {
      const user = userEvent.setup()
      const pending = createDeferred<ReturnType<typeof buildPrompt>>()
      const rendered = renderPromptSelect({
        systemPrompt: "Conversation override"
      })
      await openEditor(user, "Conversation override")
      mocks.getPromptById.mockReturnValueOnce(pending.promise)
      await user.click(screen.getByRole("button", { name: "Reset" }))

      rendered.rerender(
        <QueryClientProvider client={rendered.queryClient}>
          <PromptSelect {...rendered.props} {...changedProps} />
        </QueryClientProvider>
      )
      await act(async () => {
        pending.resolve(buildPrompt({ content: "Late lifecycle reset" }))
        await pending.promise
      })

      await waitFor(() => {
        expect(rendered.props.setSystemPrompt).not.toHaveBeenCalled()
      })
    }
  )

  it("closes the prompt dropdown when Escape is pressed from search", async () => {
    const user = userEvent.setup()
    renderPromptSelect()

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    expect(await screen.findByRole("menu")).toBeInTheDocument()

    const search = await screen.findByRole("textbox", {
      name: "Search prompts..."
    })
    search.focus()
    await user.keyboard("{Escape}")

    await waitFor(() => {
      expect(screen.queryByRole("menu")).not.toBeInTheDocument()
    })
  })

  it("returns focus to the launching rail trigger after prompt selection", async () => {
    const user = userEvent.setup()
    const { props } = renderPromptSelect({
      selectedSystemPrompt: undefined
    })
    render(
      <button type="button" data-testid="cockpit-prompt-select-trigger">
        Select prompt from rail
      </button>
    )
    const trigger = screen.getByTestId("cockpit-prompt-select-trigger")
    trigger.focus()

    window.dispatchEvent(
      new CustomEvent(OPEN_PROMPT_SELECT_EVENT, {
        detail: {
          returnFocusSelector: "[data-testid='cockpit-prompt-select-trigger']",
          source: "playground-cockpit"
        }
      })
    )

    await user.click(await screen.findByRole("menuitem", { name: /Prompt One/i }))

    await waitFor(() => {
      expect(props.setSelectedSystemPrompt).toHaveBeenCalledWith("prompt-1")
      expect(trigger).toHaveFocus()
    })
  })

  it("keeps current system prompt recovery actions visible when there are no saved prompts", async () => {
    const user = userEvent.setup()
    mocks.getAllPrompts.mockResolvedValue([])
    renderPromptSelect({
      selectedSystemPrompt: undefined,
      systemPrompt: "Stay in character."
    })

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )

    expect(await screen.findByText(/no saved prompts/i)).toBeInTheDocument()
    expect(
      screen.getByRole("menuitem", { name: /edit current system prompt/i })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("menuitem", { name: /clear current system prompt/i })
    ).toBeInTheDocument()
  })

  it("keeps current system prompt recovery actions visible when saved prompts exist", async () => {
    const user = userEvent.setup()
    renderPromptSelect({
      selectedSystemPrompt: undefined,
      systemPrompt: "Stay in character."
    })

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )

    expect(await screen.findByText(/Prompt One/i)).toBeInTheDocument()
    expect(
      screen.getByRole("menuitem", { name: /edit current system prompt/i })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("menuitem", { name: /clear current system prompt/i })
    ).toBeInTheDocument()
  })

  it("edits and saves a current custom prompt when the prompt library is empty", async () => {
    const user = userEvent.setup()
    mocks.getAllPrompts.mockResolvedValue([])
    const { props } = renderPromptSelect({
      selectedSystemPrompt: undefined,
      systemPrompt: "Stay in character."
    })

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", {
        name: /edit current system prompt/i
      })
    )

    const textarea = await screen.findByDisplayValue("Stay in character.")
    await user.clear(textarea)
    await user.type(textarea, "Speak as the station chief.")
    await user.click(screen.getByRole("button", { name: /save/i }))

    await waitFor(() => {
      expect(props.setSystemPrompt).toHaveBeenCalledWith(
        "Speak as the station chief."
      )
    })
  })

  it("clears a current custom prompt when the prompt library is empty", async () => {
    const user = userEvent.setup()
    mocks.getAllPrompts.mockResolvedValue([])
    const { props } = renderPromptSelect({
      selectedSystemPrompt: undefined,
      systemPrompt: "Stay in character."
    })

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", {
        name: /clear current system prompt/i
      })
    )

    expect(props.setSystemPrompt).toHaveBeenCalledWith("")
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "selectAPrompt" })).toHaveFocus()
    })
  })

  it("reviews the effective system draft in the existing modal and applies only a scoped override", async () => {
    const user = userEvent.setup()
    const { props } = renderPromptSelect()

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )
    await user.click(
      await screen.findByRole("button", { name: "Improve prompt" })
    )
    await user.click(screen.getByRole("button", { name: /Review changes/ }))

    expect(
      await screen.findByRole("textbox", {
        name: "Improved prompt candidate"
      })
    ).toHaveValue("Improved system draft")
    expect(screen.getAllByRole("dialog")).toHaveLength(1)
    expect(props.setSelectedSystemPrompt).not.toHaveBeenCalled()
    expect(mocks.improvePrompt).toHaveBeenCalledWith(
      expect.objectContaining({
        target: "system",
        text: "Template body",
        model_selection: {
          selected_model: "gpt-5-mini",
          provider_hint: "openai"
        }
      })
    )

    await user.click(screen.getByRole("button", { name: "Apply to draft" }))

    await waitFor(() => {
      expect(props.setSystemPrompt).toHaveBeenCalledWith(
        "Improved system draft"
      )
      expect(props.setSelectedSystemPrompt).not.toHaveBeenCalled()
      expect(screen.getByDisplayValue("Improved system draft")).toHaveFocus()
    })
    expect(screen.getByRole("button", { name: "Save" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Reset" })).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Undo improvement" })
    ).toBeInTheDocument()
  })

  it("normalizes an applied candidate matching the selected template to no override", async () => {
    const user = userEvent.setup()
    mocks.improvePrompt.mockImplementation(async (request) =>
      improvementResponse(request.operation_id, "Template body")
    )
    const { props } = renderPromptSelect({
      systemPrompt: "Conversation override"
    })

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )
    await user.click(
      await screen.findByRole("button", { name: "Improve prompt" })
    )
    await user.click(screen.getByRole("button", { name: /Review changes/ }))
    await user.click(
      await screen.findByRole("button", { name: "Apply to draft" })
    )

    expect(props.setSystemPrompt).toHaveBeenCalledWith("")
    expect(props.setSelectedSystemPrompt).not.toHaveBeenCalled()
  })

  it.each([undefined, "", "Custom override"])(
    "Undo restores the exact raw override state %s",
    async (rawOverride) => {
      const user = userEvent.setup()
      const { props } = renderPromptSelect({ systemPrompt: rawOverride })

      await user.click(
        await screen.findByRole("button", { name: "selectAPrompt" })
      )
      await user.click(
        await screen.findByRole("menuitem", { name: /edit system prompt/i })
      )
      await user.click(
        await screen.findByRole("button", { name: "Improve prompt" })
      )
      await user.click(screen.getByRole("button", { name: /Improve now/ }))
      await user.click(
        await screen.findByRole("button", { name: "Undo improvement" })
      )

      expect(props.setSystemPrompt).toHaveBeenLastCalledWith(rawOverride)
      expect(props.setSelectedSystemPrompt).not.toHaveBeenCalled()
    }
  )

  it("Reset restores the selected template without changing its identity", async () => {
    const user = userEvent.setup()
    const { props } = renderPromptSelect({
      systemPrompt: "Conversation override"
    })

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )
    await user.click(screen.getByRole("button", { name: "Reset" }))

    expect(await screen.findByDisplayValue("Template body")).toBeInTheDocument()
    expect(props.setSystemPrompt).toHaveBeenCalledWith("Template body")
    expect(props.setSelectedSystemPrompt).not.toHaveBeenCalled()
  })

  it("assist Cancel restores the draft captured on entry without stacking a modal", async () => {
    const user = userEvent.setup()
    const pending = createDeferred<ReturnType<typeof improvementResponse>>()
    mocks.improvePrompt.mockReturnValue(pending.promise)
    const { props } = renderPromptSelect()

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )
    const editor = await screen.findByDisplayValue("Template body")
    await user.clear(editor)
    await user.type(editor, "Unsaved editor draft")
    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
    await user.click(screen.getByRole("button", { name: /Review changes/ }))

    expect(screen.getAllByRole("dialog")).toHaveLength(1)
    await user.click(await screen.findByRole("button", { name: "Cancel" }))

    expect(
      await screen.findByDisplayValue("Unsaved editor draft")
    ).toHaveFocus()
    expect(props.setSystemPrompt).not.toHaveBeenCalled()
  })

  it("returns focus after confirming replacement of a stale reviewed draft", async () => {
    const user = userEvent.setup()
    const pending = createDeferred<ReturnType<typeof improvementResponse>>()
    mocks.improvePrompt.mockReturnValue(pending.promise)
    const rendered = renderPromptSelect()

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )
    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
    await user.click(screen.getByRole("button", { name: /Review changes/ }))

    pending.resolve(
      improvementResponse(
        mocks.improvePrompt.mock.calls[0][0].operation_id,
        "Replacement candidate"
      )
    )

    expect(
      await screen.findByRole("textbox", {
        name: "Improved prompt candidate"
      })
    ).toHaveValue("Replacement candidate")
    await user.type(
      screen.getByDisplayValue("Template body"),
      " changed while reviewing"
    )

    await user.click(screen.getByRole("button", { name: "Apply to draft" }))
    await user.click(
      await screen.findByRole("button", { name: "Replace current draft" })
    )
    await user.click(screen.getByRole("button", { name: "Confirm replace" }))

    expect(rendered.props.setSystemPrompt).toHaveBeenCalledWith(
      "Replacement candidate"
    )
    await waitFor(() => {
      expect(screen.getByDisplayValue("Replacement candidate")).toHaveFocus()
    })
  })

  it("keeps a polite no-change result visible in the editor", async () => {
    const user = userEvent.setup()
    mocks.improvePrompt.mockImplementation(async (request) => ({
      ...improvementResponse(request.operation_id, "Template body"),
      status: "no_change" as const
    }))
    renderPromptSelect()

    await openEditor(user)
    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
    await user.click(screen.getByRole("button", { name: /Improve now/ }))

    expect(
      await screen.findByText("No useful improvement found.")
    ).toBeInTheDocument()
    expect(screen.getByDisplayValue("Template body")).toBeInTheDocument()
  })

  it("typing after Apply invalidates Undo while keeping the editor usable", async () => {
    const user = userEvent.setup()
    renderPromptSelect()

    await openEditor(user)
    await applyImprovementNow(user)
    const editor = screen.getByDisplayValue("Improved system draft")
    await user.type(editor, " with a local edit")

    expect(
      screen.queryByRole("button", { name: "Undo improvement" })
    ).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save" })).toBeInTheDocument()
  })

  it("Reset after Apply clears Undo and restores the selected template", async () => {
    const user = userEvent.setup()
    const { props } = renderPromptSelect()

    await openEditor(user)
    await applyImprovementNow(user)
    await user.click(screen.getByRole("button", { name: "Reset" }))

    expect(await screen.findByDisplayValue("Template body")).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Undo improvement" })
    ).not.toBeInTheDocument()
    expect(props.setSystemPrompt).toHaveBeenLastCalledWith("Template body")
  })

  it("Save after Apply closes the modal and consumes Undo", async () => {
    const user = userEvent.setup()
    renderPromptSelect()

    await openEditor(user)
    await applyImprovementNow(user)
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    await openEditor(user)
    expect(
      screen.queryByRole("button", { name: "Undo improvement" })
    ).not.toBeInTheDocument()
  })

  it.each([
    ["template", { selectedSystemPrompt: "prompt-2" }],
    ["model", { selectedModel: "gpt-5" }],
    ["provider", { currentProvider: "anthropic" }],
    ["conversation", { promptAssistContextKey: "conversation-2" }]
  ])(
    "%s changes after Apply invalidate Undo",
    async (_change, changedProps) => {
      const user = userEvent.setup()
      const rendered = renderPromptSelect()

      await openEditor(user)
      await applyImprovementNow(user)

      rendered.rerender(
        <QueryClientProvider client={new QueryClient()}>
          <PromptSelect {...rendered.props} {...changedProps} />
        </QueryClientProvider>
      )

      await waitFor(() => {
        expect(
          screen.queryByRole("button", { name: "Undo improvement" })
        ).not.toBeInTheDocument()
      })
    }
  )

  it("fails closed while prompt-improvement capability is unknown", async () => {
    const user = userEvent.setup()
    mocks.fetchPromptCapabilities.mockReturnValue(new Promise(() => {}))
    renderPromptSelect()

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(
      await screen.findByRole("menuitem", { name: /edit system prompt/i })
    )
    await user.click(screen.getByRole("button", { name: "Improve prompt" }))

    expect(screen.getByRole("button", { name: /Improve now/ })).toBeDisabled()
    expect(mocks.improvePrompt).not.toHaveBeenCalled()
  })

  it("closes the editor and hands missing idle model recovery to its owner", async () => {
    const user = userEvent.setup()
    const { props } = renderPromptSelect({ selectedModel: null })

    await openEditor(user)
    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
    await user.click(screen.getByRole("button", { name: "Select model" }))

    expect(props.onSelectModel).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(mocks.improvePrompt).not.toHaveBeenCalled()
  })

  it("closes the editor and hands result-failure model recovery to its owner", async () => {
    const user = userEvent.setup()
    mocks.improvePrompt.mockRejectedValueOnce({
      code: "missing_model",
      retryable: false
    })
    const { props } = renderPromptSelect()

    await openEditor(user)
    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
    await user.click(screen.getByRole("button", { name: /Improve now/ }))
    await user.click(
      await screen.findByRole("button", { name: "Select model" })
    )

    expect(props.onSelectModel).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("fails closed without an authoritative backend identity", async () => {
    const user = userEvent.setup()
    renderPromptSelect({ promptAssistBackendKey: null })

    await openEditor(user)
    await user.click(screen.getByRole("button", { name: "Improve prompt" }))

    expect(screen.getByRole("button", { name: /Improve now/ })).toBeDisabled()
    expect(mocks.fetchPromptCapabilities).not.toHaveBeenCalled()
  })

  it("does not reuse supported capabilities after the backend identity changes", async () => {
    const user = userEvent.setup()
    const backendB = createDeferred<{
      availability: "available"
      prompt_improvement_v1: { supported: false; limits: null }
      single_text_recipe_v2: { supported: false }
    }>()
    mocks.fetchPromptCapabilities
      .mockResolvedValueOnce({
        availability: "available",
        prompt_improvement_v1: { supported: true, limits: null },
        single_text_recipe_v2: { supported: false }
      })
      .mockReturnValueOnce(backendB.promise)
    const rendered = renderPromptSelect()

    await openEditor(user)
    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
    expect(
      await screen.findByRole("button", { name: /Improve now/ })
    ).toBeEnabled()

    rendered.rerender(
      <QueryClientProvider client={rendered.queryClient}>
        <PromptSelect {...rendered.props} promptAssistBackendKey="backend-b" />
      </QueryClientProvider>
    )

    expect(screen.getByRole("button", { name: /Improve now/ })).toBeDisabled()
    expect(mocks.fetchPromptCapabilities).toHaveBeenCalledTimes(2)
    expect(
      rendered.queryClient.getQueryCache().find({
        queryKey: ["promptCapabilities", "backend-b"]
      })?.options.retry
    ).toBe(false)
  })

  it("ignores an old backend capability response while the new backend is unresolved", async () => {
    const user = userEvent.setup()
    const backendA =
      createDeferred<ReturnType<typeof mocks.fetchPromptCapabilities>>()
    const backendB =
      createDeferred<ReturnType<typeof mocks.fetchPromptCapabilities>>()
    mocks.fetchPromptCapabilities
      .mockReturnValueOnce(backendA.promise)
      .mockReturnValueOnce(backendB.promise)
    const rendered = renderPromptSelect()

    await openEditor(user)
    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
    expect(screen.getByRole("button", { name: /Improve now/ })).toBeDisabled()

    rendered.rerender(
      <QueryClientProvider client={rendered.queryClient}>
        <PromptSelect {...rendered.props} promptAssistBackendKey="backend-b" />
      </QueryClientProvider>
    )
    backendA.resolve({
      availability: "available",
      prompt_improvement_v1: { supported: true, limits: null },
      single_text_recipe_v2: { supported: false }
    })

    await waitFor(() => {
      expect(mocks.fetchPromptCapabilities).toHaveBeenCalledTimes(2)
    })
    expect(screen.getByRole("button", { name: /Improve now/ })).toBeDisabled()
  })

  it.each([
    ["template", { selectedSystemPrompt: "prompt-2" }],
    ["model", { selectedModel: "gpt-5" }],
    ["provider", { currentProvider: "anthropic" }],
    ["context", { promptAssistContextKey: "conversation-2" }],
    ["backend", { promptAssistBackendKey: "backend-b" }]
  ])(
    "%s changes invalidate an in-flight result without overwriting or stealing focus",
    async (_change, changedProps) => {
      const user = userEvent.setup()
      const pending = createDeferred<ReturnType<typeof improvementResponse>>()
      mocks.improvePrompt.mockReturnValue(pending.promise)
      const rendered = renderPromptSelect()
      render(<button type="button">Focus sentinel</button>)

      await user.click(
        await screen.findByRole("button", { name: "selectAPrompt" })
      )
      await user.click(
        await screen.findByRole("menuitem", { name: /edit system prompt/i })
      )
      await user.click(screen.getByRole("button", { name: "Improve prompt" }))
      await user.click(screen.getByRole("button", { name: /Improve now/ }))

      rendered.rerender(
        <QueryClientProvider client={new QueryClient()}>
          <PromptSelect {...rendered.props} {...changedProps} />
        </QueryClientProvider>
      )
      const focusSentinel = screen.getByRole("button", {
        name: "Focus sentinel"
      })
      focusSentinel.focus()
      pending.resolve(
        improvementResponse(
          mocks.improvePrompt.mock.calls[0][0].operation_id,
          "Late candidate"
        )
      )

      await waitFor(() => expect(mocks.improvePrompt).toHaveBeenCalledTimes(1))
      expect(rendered.props.setSystemPrompt).not.toHaveBeenCalled()
      expect(focusSentinel).toHaveFocus()
    }
  )
})
