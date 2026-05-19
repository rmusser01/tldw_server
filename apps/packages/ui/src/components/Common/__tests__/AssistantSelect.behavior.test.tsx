import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  listAllCharacters: vi.fn(async () => []),
  listPersonaProfiles: vi.fn(async () => []),
  selectedAssistant: {
    value: null as null | {
      kind: "character" | "persona"
      id: string
      name: string
    }
  },
  setSelectedAssistant: vi.fn(async () => undefined)
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

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(async () => null),
    listAllCharacters: mocks.listAllCharacters,
    listPersonaProfiles: mocks.listPersonaProfiles
  }
}))

vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [
    mocks.selectedAssistant.value,
    mocks.setSelectedAssistant,
    { isLoading: false, setRenderValue: vi.fn() }
  ]
}))

vi.mock("antd", async () => {
  const React = await import("react")

  const Input = React.forwardRef<HTMLInputElement, any>((props, ref) => (
    <input
      ref={ref}
      aria-label={props["aria-label"] ?? props.placeholder}
      value={props.value}
      defaultValue={props.defaultValue}
      onChange={props.onChange}
      onKeyDown={props.onKeyDown}
    />
  ))

  const Tooltip = ({ children }: { children: React.ReactNode }) => <>{children}</>

  const Dropdown = ({
    open,
    onOpenChange,
    popupRender,
    children
  }: any) => {
    const containerRef = React.useRef<HTMLDivElement | null>(null)

    React.useEffect(() => {
      if (!open) return
      const onMouseDown = (event: MouseEvent) => {
        if (!containerRef.current?.contains(event.target as Node)) {
          onOpenChange?.(false)
        }
      }
      const onKeyDown = (event: KeyboardEvent) => {
        if (event.key === "Escape") {
          onOpenChange?.(false)
        }
      }
      document.addEventListener("mousedown", onMouseDown)
      document.addEventListener("keydown", onKeyDown)
      return () => {
        document.removeEventListener("mousedown", onMouseDown)
        document.removeEventListener("keydown", onKeyDown)
      }
    }, [open, onOpenChange])

    return (
      <div ref={containerRef}>
        <div onClick={() => onOpenChange?.(!open)}>{children}</div>
        {open ? popupRender?.(null) : null}
      </div>
    )
  }

  return {
    Dropdown,
    Input,
    Tooltip
  }
})

import { AssistantSelect } from "../AssistantSelect"

const renderAssistantSelect = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false
      }
    }
  })

  render(
    <QueryClientProvider client={queryClient}>
      <button id="assistant-rail-trigger" type="button">
        Runtime rail trigger
      </button>
      <AssistantSelect variant="dropdown" />
    </QueryClientProvider>
  )
}

describe("AssistantSelect behavior", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.selectedAssistant.value = null
    mocks.listAllCharacters.mockResolvedValue([
      { id: "char-1", name: "Alpha" },
      { id: "char-2", name: "Beta" }
    ])
    mocks.listPersonaProfiles.mockResolvedValue([
      { id: "persona-1", name: "Guide Persona" }
    ])
  })

  it("opens a searchable menu and filters visible characters", async () => {
    const user = userEvent.setup()
    renderAssistantSelect()

    await user.click(
      await screen.findByRole("button", { name: "Select character or persona" })
    )

    await user.type(
      await screen.findByRole("textbox", { name: /search characters and personas/i }),
      "beta"
    )

    expect(await screen.findByRole("button", { name: "Beta" })).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Alpha" })).toBeNull()
  })

  it("announces character and persona catalog loading in the selector", async () => {
    mocks.listAllCharacters.mockReturnValue(new Promise(() => {}))
    mocks.listPersonaProfiles.mockReturnValue(new Promise(() => {}))

    const user = userEvent.setup()
    renderAssistantSelect()

    await user.click(
      await screen.findByRole("button", { name: "Select character or persona" })
    )

    const status = await screen.findByRole("status", {
      name: /loading character and persona catalogs/i
    })
    expect(status).toHaveTextContent("Loading characters and personas")
  })

  it("shows a retryable character catalog failure instead of an empty list", async () => {
    mocks.listAllCharacters.mockRejectedValueOnce(new Error("characters failed"))

    const user = userEvent.setup()
    renderAssistantSelect()

    await user.click(
      await screen.findByRole("button", { name: "Select character or persona" })
    )

    expect(
      await screen.findByText(/could not load characters/i)
    ).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: /retry characters/i }))

    expect(mocks.listAllCharacters).toHaveBeenCalledTimes(2)
  })

  it("keeps loaded characters usable when persona catalog loading fails", async () => {
    mocks.listPersonaProfiles.mockRejectedValueOnce(new Error("personas failed"))

    const user = userEvent.setup()
    renderAssistantSelect()

    await user.click(
      await screen.findByRole("button", { name: "Select character or persona" })
    )

    expect(await screen.findByRole("button", { name: "Alpha" })).toBeInTheDocument()
    await user.click(await screen.findByRole("tab", { name: "Personas" }))
    expect(
      await screen.findByText(/could not load personas/i)
    ).toBeInTheDocument()
  })

  it("uses solid design-token backgrounds for the selector panel", async () => {
    const user = userEvent.setup()
    renderAssistantSelect()

    await user.click(
      await screen.findByRole("button", { name: "Select character or persona" })
    )

    const panel = await screen.findByTestId("assistant-select-panel")
    expect(panel).toHaveClass("bg-elevated")
    expect(panel).not.toHaveClass("bg-background")
    expect(await screen.findByRole("button", { name: "Alpha" })).toHaveClass(
      "bg-surface"
    )
  })

  it("does not select a character when its favorite star is clicked", async () => {
    const user = userEvent.setup()
    renderAssistantSelect()

    await user.click(
      await screen.findByRole("button", { name: "Select character or persona" })
    )

    await user.click(
      await screen.findByRole("button", { name: /add beta to favorites/i })
    )

    expect(mocks.setSelectedAssistant).not.toHaveBeenCalled()
  })

  it("opens the character tab when character chat requests character selection", async () => {
    renderAssistantSelect()

    window.dispatchEvent(
      new CustomEvent("tldw:open-assistant-select", {
        detail: { tab: "character", source: "chat-header" }
      })
    )

    expect(
      await screen.findByRole("button", { name: "Alpha" })
    ).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "Characters" })).toHaveAttribute(
      "aria-selected",
      "true"
    )
  })

  it("opens the persona tab from an assistant-select event", async () => {
    renderAssistantSelect()

    window.dispatchEvent(
      new CustomEvent("tldw:open-assistant-select", {
        detail: { tab: "persona", source: "playground-cockpit" }
      })
    )

    expect(
      await screen.findByRole("button", { name: "Guide Persona" })
    ).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "Personas" })).toHaveAttribute(
      "aria-selected",
      "true"
    )
  })

  it("returns focus to the requested rail trigger after selection", async () => {
    const user = userEvent.setup()
    renderAssistantSelect()
    const railTrigger = screen.getByRole("button", {
      name: "Runtime rail trigger"
    })

    window.dispatchEvent(
      new CustomEvent("tldw:open-assistant-select", {
        detail: {
          tab: "persona",
          source: "playground-cockpit",
          returnFocusSelector: "#assistant-rail-trigger"
        }
      })
    )

    await user.click(await screen.findByRole("button", { name: "Guide Persona" }))

    await waitFor(() => {
      expect(document.activeElement).toBe(railTrigger)
    })
  })

  it("closes the menu and restores focus without waiting for selection persistence", async () => {
    const user = userEvent.setup()
    let resolveSelection: (() => void) | undefined
    mocks.setSelectedAssistant.mockImplementationOnce(
      () =>
        new Promise<void>((resolve) => {
          resolveSelection = resolve
        })
    )

    renderAssistantSelect()
    const railTrigger = screen.getByRole("button", {
      name: "Runtime rail trigger"
    })

    window.dispatchEvent(
      new CustomEvent("tldw:open-assistant-select", {
        detail: {
          tab: "character",
          source: "playground-cockpit",
          returnFocusSelector: "#assistant-rail-trigger"
        }
      })
    )

    await user.click(await screen.findByRole("button", { name: "Alpha" }))

    await waitFor(() => {
      expect(screen.queryByTestId("assistant-select-menu")).toBeNull()
      expect(document.activeElement).toBe(railTrigger)
    })
    expect(mocks.setSelectedAssistant).toHaveBeenCalledWith(
      expect.objectContaining({ kind: "character", id: "char-1", name: "Alpha" })
    )

    resolveSelection?.()
  })

  it("returns focus to the requested rail trigger after Escape closes the menu", async () => {
    renderAssistantSelect()
    const railTrigger = screen.getByRole("button", {
      name: "Runtime rail trigger"
    })

    window.dispatchEvent(
      new CustomEvent("tldw:open-assistant-select", {
        detail: {
          tab: "character",
          source: "playground-cockpit",
          returnFocusSelector: "#assistant-rail-trigger"
        }
      })
    )

    expect(await screen.findByRole("button", { name: "Alpha" })).toBeInTheDocument()

    fireEvent.keyDown(document, { key: "Escape" })

    await waitFor(() => {
      expect(document.activeElement).toBe(railTrigger)
    })
  })

  it("labels identity and optional scene choices without mixing character and persona concepts", async () => {
    const user = userEvent.setup()
    renderAssistantSelect()

    await user.click(
      await screen.findByRole("button", { name: "Select character or persona" })
    )

    expect(
      screen.getByRole("tablist", { name: "Character or persona" })
    ).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "Characters" })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "Personas" })).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Optional scene context" })
    ).toBeInTheDocument()
  })

  it("keeps personas accessible and dispatches the actor footer action", async () => {
    const user = userEvent.setup()
    const actorListener = vi.fn()
    window.addEventListener("tldw:open-actor-settings", actorListener)

    renderAssistantSelect()

    await user.click(
      await screen.findByRole("button", { name: "Select character or persona" })
    )

    await user.click(await screen.findByRole("tab", { name: "Personas" }))
    expect(
      await screen.findByRole("button", { name: "Guide Persona" })
    ).toBeInTheDocument()

    await user.click(
      await screen.findByRole("button", { name: "Optional scene context" })
    )

    expect(actorListener).toHaveBeenCalledTimes(1)
    window.removeEventListener("tldw:open-actor-settings", actorListener)
  })

  it("lists and selects personas with canonical buddy summary payloads", async () => {
    const user = userEvent.setup()
    mocks.listPersonaProfiles.mockResolvedValue([
      {
        id: "persona-1",
        name: "Guide Persona",
        avatar_url: "https://example.com/guide.png",
        buddy_summary: {
          has_buddy: true,
          persona_name: "Guide Persona",
          role_summary: "Keeps the chat on course",
          visual: {
            species_id: "owl",
            silhouette_id: "perch",
            palette_id: "dawn"
          }
        }
      }
    ])

    renderAssistantSelect()

    await user.click(
      await screen.findByRole("button", { name: "Select character or persona" })
    )

    await user.click(await screen.findByRole("tab", { name: "Personas" }))

    const personaButton = await screen.findByRole("button", {
      name: "Guide Persona"
    })
    expect(within(personaButton).getByRole("img", { name: "Guide Persona" })).toHaveAttribute(
      "src",
      "https://example.com/guide.png"
    )

    await user.click(personaButton)

    expect(mocks.setSelectedAssistant).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "persona",
        id: "persona-1",
        name: "Guide Persona",
        avatar_url: "https://example.com/guide.png",
        buddy_summary: {
          has_buddy: true,
          persona_name: "Guide Persona",
          role_summary: "Keeps the chat on course",
          visual: {
            species_id: "owl",
            silhouette_id: "perch",
            palette_id: "dawn"
          }
        }
      })
    )
  })

  it.each([
    {
      label: "character to another character",
      source: { kind: "character" as const, id: "char-1", name: "Alpha" },
      targetTab: "Characters",
      targetName: "Beta",
      expected: { kind: "character", id: "char-2", name: "Beta" }
    },
    {
      label: "character to persona",
      source: { kind: "character" as const, id: "char-1", name: "Alpha" },
      targetTab: "Personas",
      targetName: "Guide Persona",
      expected: { kind: "persona", id: "persona-1", name: "Guide Persona" }
    },
    {
      label: "persona to character",
      source: { kind: "persona" as const, id: "persona-1", name: "Guide Persona" },
      targetTab: "Characters",
      targetName: "Alpha",
      expected: { kind: "character", id: "char-1", name: "Alpha" }
    },
    {
      label: "none to persona",
      source: null,
      targetTab: "Personas",
      targetName: "Guide Persona",
      expected: { kind: "persona", id: "persona-1", name: "Guide Persona" }
    }
  ])("supports $label from the assistant selector", async ({ source, targetTab, targetName, expected }) => {
    const user = userEvent.setup()
    mocks.selectedAssistant.value = source
    renderAssistantSelect()

    await user.click(
      await screen.findByRole("button", {
        name: source?.name ?? "Select character or persona"
      })
    )

    await user.click(await screen.findByRole("tab", { name: targetTab }))
    mocks.setSelectedAssistant.mockClear()
    await user.click(await screen.findByRole("button", { name: targetName }))

    expect(mocks.setSelectedAssistant).toHaveBeenCalledTimes(1)
    expect(mocks.setSelectedAssistant).toHaveBeenLastCalledWith(
      expect.objectContaining(expected)
    )
  })

  it("moves a favorited character ahead of other characters and closes on escape", async () => {
    const user = userEvent.setup()
    renderAssistantSelect()

    await user.click(
      await screen.findByRole("button", { name: "Select character or persona" })
    )

    await user.click(
      await screen.findByRole("button", { name: /add beta to favorites/i })
    )

    const menu = screen.getByTestId("assistant-select-menu")
    const characterButtons = within(menu).getAllByRole("button")
    expect(characterButtons[0]).toHaveTextContent("Beta")

    fireEvent.keyDown(document, { key: "Escape" })

    await waitFor(() => {
      expect(screen.queryByTestId("assistant-select-menu")).toBeNull()
    })
  })
})
