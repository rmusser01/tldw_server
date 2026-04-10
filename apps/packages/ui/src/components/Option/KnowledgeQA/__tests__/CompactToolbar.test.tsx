import { beforeEach, describe, expect, it, vi } from "vitest"
import { cleanup, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import React from "react"
import type { RagSource } from "@/services/rag/unified-rag"

vi.mock("@/libs/utils", () => ({
  cn: (...args: unknown[]) => args.filter(Boolean).join(" "),
}))

vi.mock("@/services/rag/unified-rag", () => ({}))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn().mockResolvedValue(undefined),
    getProviders: vi.fn().mockResolvedValue({
      default_provider: "openai",
      providers: [
        { name: "openai", display_name: "OpenAI", models: ["gpt-4o-mini"] },
      ],
    }),
  },
}))

import { CompactToolbar } from "../context/CompactToolbar"

const defaultProps = {
  sources: [] as RagSource[],
  preset: "balanced" as const,
  webEnabled: false,
  onToggleWeb: vi.fn(),
  onOpenSourceSelector: vi.fn(),
  onOpenSettings: vi.fn(),
  generationProvider: null as string | null,
  generationModel: null as string | null,
  onGenerationProviderChange: vi.fn(),
  onGenerationModelChange: vi.fn(),
  contextChangedSinceLastRun: false,
}

function renderToolbar(overrides: Partial<typeof defaultProps> = {}) {
  return render(<CompactToolbar {...defaultProps} {...overrides} />)
}

describe("CompactToolbar", () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders source summary "None" when sources is empty', () => {
    renderToolbar({ sources: [] })
    expect(screen.getByText(/Sources:.*None/)).toBeDefined()
  })

  it('renders single source label "Documents & Media" for media_db', () => {
    renderToolbar({ sources: ["media_db"] })
    expect(screen.getByText(/Sources:.*Documents & Media/)).toBeDefined()
  })

  it('renders "N selected" for 2-4 sources', () => {
    renderToolbar({ sources: ["media_db", "notes"] })
    expect(screen.getByText(/Sources:.*2 selected/)).toBeDefined()

    cleanup()
    renderToolbar({ sources: ["media_db", "notes", "characters", "chats"] })
    expect(screen.getByText(/Sources:.*4 selected/)).toBeDefined()
  })

  it('renders "All sources" for 5+ sources', () => {
    renderToolbar({
      sources: ["media_db", "notes", "characters", "chats", "kanban"],
    })
    expect(screen.getByText(/Sources:.*All sources/)).toBeDefined()
  })

  it('renders preset label "Balanced" for preset "balanced"', () => {
    renderToolbar({ preset: "balanced" })
    expect(screen.getByTitle(/Search preset: Balanced/)).toBeDefined()
  })

  it("falls back to raw preset name for unknown preset", () => {
    renderToolbar({ preset: "mystery" as any })
    expect(screen.getByTitle(/Search preset: mystery/)).toBeDefined()
    expect(screen.getByText("mystery")).toBeDefined()
  })

  it("calls onOpenSourceSelector when sources pill is clicked", async () => {
    const onOpenSourceSelector = vi.fn()
    renderToolbar({ onOpenSourceSelector })
    const btn = screen.getByText(/Sources:/)
    await userEvent.click(btn)
    expect(onOpenSourceSelector).toHaveBeenCalledOnce()
  })

  it("calls onToggleWeb when web pill is clicked", async () => {
    const onToggleWeb = vi.fn()
    renderToolbar({ onToggleWeb })
    const btn = screen.getByLabelText(/Web fallback/)
    await userEvent.click(btn)
    expect(onToggleWeb).toHaveBeenCalledOnce()
  })

  it("calls onOpenSettings when settings gear is clicked", async () => {
    const onOpenSettings = vi.fn()
    renderToolbar({ onOpenSettings })
    const btn = screen.getByLabelText("Open settings")
    await userEvent.click(btn)
    expect(onOpenSettings).toHaveBeenCalledOnce()
  })

  it("shows an answer model control in the toolbar", () => {
    renderToolbar()
    expect(screen.getByRole("button", { name: "Choose answer model" })).toBeInTheDocument()
  })

  it('shows "Scope changed" badge when contextChangedSinceLastRun is true', () => {
    renderToolbar({ contextChangedSinceLastRun: true })
    expect(screen.getByText("Scope changed")).toBeDefined()
  })

  it('does NOT show "Scope changed" when contextChangedSinceLastRun is false', () => {
    renderToolbar({ contextChangedSinceLastRun: false })
    expect(screen.queryByText("Scope changed")).toBeNull()
  })

  it("web pill has aria-pressed matching webEnabled state", () => {
    const { unmount } = renderToolbar({ webEnabled: false })
    const btn = screen.getByLabelText(/Web fallback/)
    expect(btn.getAttribute("aria-pressed")).toBe("false")
    unmount()

    renderToolbar({ webEnabled: true })
    const btnEnabled = screen.getByLabelText(/Web fallback/)
    expect(btnEnabled.getAttribute("aria-pressed")).toBe("true")
  })
})
