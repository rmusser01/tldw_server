import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { OPEN_PROMPT_SELECT_EVENT } from "@/utils/prompt-select-events"

const mocks = vi.hoisted(() => ({
  getAllPrompts: vi.fn(async () => []),
  getPromptById: vi.fn(async () => undefined)
}))

const registryLabels = vi.hoisted(() => ({
  loading: "Loading via registry"
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
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
    ...overrides
  }

  return {
    ...render(
      <QueryClientProvider client={queryClient}>
        <PromptSelect {...props} />
      </QueryClientProvider>
    ),
    props
  }
}

describe("PromptSelect system prompt modal", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.getAllPrompts.mockResolvedValue([buildPrompt()])
    mocks.getPromptById.mockResolvedValue(buildPrompt())
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

  it("uses the design-system loading label while resolving editor content", async () => {
    const user = userEvent.setup()
    mocks.getPromptById.mockReturnValue(new Promise(() => {}))
    renderPromptSelect()

    await user.click(
      await screen.findByRole("button", { name: "selectAPrompt" })
    )
    await user.click(await screen.findByRole("menuitem", { name: /edit system prompt/i }))

    expect(await screen.findByText("Loading via registry")).toBeInTheDocument()
  })

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
})
