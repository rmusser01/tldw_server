// @vitest-environment jsdom
/**
 * XS-15 (#3101): the Pro composer took about 64% of a 400x900 side panel.
 * The review's fix, piece by piece:
 * - the model parameters collapse into one "Runtime" status chip with a
 *   popover, and remember which sections were open;
 * - Search & Context opens as an overlay sheet, not an in-flow card;
 * - the Pro textarea stops growing at 96px;
 * - the pinned composer dock is capped at 40% of the panel, so the
 *   conversation keeps at least 60% at any width (360–420px included).
 *
 * jsdom has no layout, so the dock budget is checked on the classes that set
 * it. It has not been measured in a browser.
 */
import { readFileSync } from "node:fs"
import path from "node:path"
import React from "react"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useStoreChatModelSettings } from "@/store/model"
import { useUiModeStore } from "@/store/ui-mode"
import { COMPOSER_CONSTANTS } from "@/config/ui-constants"

vi.mock("@/hooks/useSelectedModelSamplingSupport", () => ({
  useSelectedModelSamplingSupport: () => ({})
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string, options?: Record<string, unknown>) =>
      String(fallback || _key).replace(/\{\{(\w+)\}\}/g, (_match, name) =>
        String(options?.[name] ?? "")
      )
  })
}))
vi.mock("@/services/model-settings", () => ({
  getAllDefaultModelSettings: vi.fn(async () => ({}))
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { getLlmProviders: vi.fn(async () => ({ providers: {} })) }
}))
vi.mock("@/components/Knowledge", () => ({
  KnowledgePanel: ({ open, onOpenChange }: { open?: boolean; onOpenChange?: (open: boolean) => void }) =>
    open ? (
      <div data-testid="knowledge-panel">
        Search &amp; Context panel
        <button type="button" onClick={() => onOpenChange?.(false)}>Close</button>
      </div>
    ) : null
}))

import { ModelParamsPanel } from "../ModelParamsPanel"
import { MODEL_PARAMS_SECTIONS_STORAGE_KEY } from "../model-params-sections"
import { SidepanelRuntimeChip } from "../SidepanelRuntimeChip"
import { SidepanelKnowledgeSheet } from "../SidepanelKnowledgeSheet"

const withQuery = (ui: React.ReactElement) => (
  <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    {ui}
  </QueryClientProvider>
)

const source = (relative: string) =>
  readFileSync(path.resolve(__dirname, relative), "utf8")

beforeEach(() => {
  localStorage.removeItem(MODEL_PARAMS_SECTIONS_STORAGE_KEY)
  useStoreChatModelSettings.getState().reset()
  useUiModeStore.setState({ mode: "pro" })
})

describe("model parameter sections are remembered", () => {
  it("reopens the sections the user left open", () => {
    const first = render(withQuery(<ModelParamsPanel selectedModel="openai/gpt-4o" />))
    fireEvent.click(screen.getByRole("button", { name: "Model Parameters" }))
    expect(screen.getByRole("button", { name: "Model Parameters" })).toHaveAttribute("aria-expanded", "true")
    first.unmount()

    render(withQuery(<ModelParamsPanel selectedModel="openai/gpt-4o" />))
    expect(screen.getByRole("button", { name: "Model Parameters" })).toHaveAttribute("aria-expanded", "true")
    expect(screen.getByRole("button", { name: "Provider & API" })).toHaveAttribute("aria-expanded", "false")
  })
})

describe("SidepanelRuntimeChip", () => {
  it("shows the parameters as one Runtime chip, not inline sections", () => {
    useStoreChatModelSettings.getState().setTemperature(0.4)
    render(withQuery(<SidepanelRuntimeChip selectedModel="openai/gpt-4o" />))
    const chip = screen.getByRole("button", { name: /Runtime/ })
    expect(chip).toHaveTextContent("temp 0.4")
    expect(chip).toHaveAttribute("aria-expanded", "false")
    expect(screen.queryByTestId("model-params-panel")).toBeNull()
  })

  it("says the defaults are in use when nothing is set here", () => {
    render(withQuery(<SidepanelRuntimeChip selectedModel="openai/gpt-4o" />))
    expect(screen.getByRole("button", { name: /Runtime/ })).toHaveTextContent("Defaults")
    expect(screen.getByRole("button", { name: /Runtime/ })).not.toHaveTextContent("Model defaults")
  })

  // Review fix: the chip shows what is set here, and counts the rest.
  it("shows the first two values set here and counts the others", () => {
    const state = useStoreChatModelSettings.getState()
    state.setTemperature(0.4)
    state.setTopP(0.9)
    state.setSeed(7)
    state.setJsonMode(true)
    render(withQuery(<SidepanelRuntimeChip selectedModel="openai/gpt-4o" />))
    expect(screen.getByRole("button", { name: /Runtime/ })).toHaveTextContent("temp 0.4 · top-p 0.9 · +2")
  })

  it("is keyboard-operable: focus moves in, Escape closes and returns to the chip", async () => {
    render(withQuery(<SidepanelRuntimeChip selectedModel="openai/gpt-4o" />))
    const chip = screen.getByRole("button", { name: /Runtime/ })
    expect(chip).toHaveAttribute("aria-haspopup", "dialog")
    chip.focus()
    await act(async () => { fireEvent.click(chip) })
    const dialog = await screen.findByRole("dialog", { name: "Runtime" })
    await waitFor(() => expect(within(dialog).getByRole("button", { name: "Provider & API" })).toHaveFocus())
    await act(async () => { fireEvent.keyDown(within(dialog).getByRole("button", { name: "Provider & API" }), { key: "Escape" }) })
    await waitFor(() => expect(chip).toHaveAttribute("aria-expanded", "false"))
    await waitFor(() => expect(chip).toHaveFocus())
  })

  it("opens the parameters in a popover", async () => {
    render(withQuery(<SidepanelRuntimeChip selectedModel="openai/gpt-4o" />))
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: /Runtime/ })) })
    expect(screen.getByRole("button", { name: /Runtime/ })).toHaveAttribute("aria-expanded", "true")
    expect(await screen.findByTestId("model-params-panel")).toBeInTheDocument()
  })

  it("is hidden outside Pro, like the panel it replaces", () => {
    useUiModeStore.setState({ mode: "casual" })
    render(withQuery(<SidepanelRuntimeChip selectedModel="openai/gpt-4o" />))
    expect(screen.queryByRole("button", { name: /Runtime/ })).toBeNull()
  })
})

describe("SidepanelKnowledgeSheet", () => {
  it("opens Search & Context as an overlay sheet outside the composer's flow", () => {
    const onOpenChange = vi.fn()
    render(
      <div data-testid="composer">
        <SidepanelKnowledgeSheet open onOpenChange={onOpenChange} />
      </div>
    )
    const sheet = screen.getByRole("dialog", { name: "Search & Context" })
    expect(within(sheet).getByTestId("knowledge-panel")).toBeInTheDocument()
    expect(screen.getByTestId("composer")).not.toContainElement(sheet)
    expect(sheet.className).toMatch(/\bfixed\b/)
    expect(sheet.className).toMatch(/max-h-/)
  })

  it("closes on Escape", () => {
    const onOpenChange = vi.fn()
    render(<SidepanelKnowledgeSheet open onOpenChange={onOpenChange} />)
    fireEvent.keyDown(screen.getByRole("dialog", { name: "Search & Context" }), { key: "Escape" })
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  // Review fixes: a modal sheet that can be left at once and gives focus back.
  it("is a modal dialog that takes focus as it opens, so Escape works at once", async () => {
    const onOpenChange = vi.fn()
    render(<SidepanelKnowledgeSheet open onOpenChange={onOpenChange} />)
    const sheet = screen.getByRole("dialog", { name: "Search & Context" })
    expect(sheet).toHaveAttribute("aria-modal", "true")
    await waitFor(() => expect(sheet.contains(document.activeElement)).toBe(true))
    fireEvent.keyDown(document.activeElement as Element, { key: "Escape" })
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it("returns focus to the control that opened it", async () => {
    const onOpenChange = vi.fn()
    const view = render(<><button type="button">Search & Context toggle</button><SidepanelKnowledgeSheet open={false} onOpenChange={onOpenChange} /></>)
    const toggle = screen.getByRole("button", { name: "Search & Context toggle" })
    toggle.focus()
    view.rerender(<><button type="button">Search & Context toggle</button><SidepanelKnowledgeSheet open onOpenChange={onOpenChange} /></>)
    await waitFor(() => expect(toggle).not.toHaveFocus())
    view.rerender(<><button type="button">Search & Context toggle</button><SidepanelKnowledgeSheet open={false} onOpenChange={onOpenChange} /></>)
    await waitFor(() => expect(toggle).toHaveFocus())
  })

  it("closes on a click outside the sheet", () => {
    const onOpenChange = vi.fn()
    render(<SidepanelKnowledgeSheet open onOpenChange={onOpenChange} />)
    fireEvent.mouseDown(screen.getByTestId("sidepanel-knowledge-sheet-backdrop"))
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it("keeps the panel mounted while closed, so its open event still arrives", () => {
    render(<SidepanelKnowledgeSheet open={false} onOpenChange={vi.fn()} />)
    expect(screen.queryByRole("dialog")).toBeNull()
    expect(document.querySelector('[data-testid="sidepanel-knowledge-sheet"]')).not.toBeNull()
  })
})

describe("Pro composer share budget (source check)", () => {
  // Review fix: 96px was below Casual's cap; the dock budget holds without it.
  it("lets the Pro textarea grow at least as far as Casual's", () => {
    expect(COMPOSER_CONSTANTS.TEXTAREA_MAX_HEIGHT_PRO).toBeGreaterThanOrEqual(
      COMPOSER_CONSTANTS.TEXTAREA_MAX_HEIGHT_CASUAL
    )
  })

  // Coordinator's XS-15 decision: the textarea and Send always stay in view.
  // Optional rows scroll in one region at the top; the essential rows never
  // scroll, and the dock grows past 40% to fit them.
  it("splits the dock into a scrolling optional region and fixed essential rows", () => {
    const route = source("../../../../routes/sidepanel-chat.tsx")
    expect(route).toMatch(/data-testid="sidepanel-composer-dock"\s+className="[^"]*max-h-\[max\(40%,var\(--sp-dock-fixed-h,0px\)\)\][^"]*overflow-hidden/)
    const form = source("../form.tsx")
    expect(form).toMatch(/data-sidepanel-dock-optional[^>]*className="[^"]*min-h-0[^"]*overflow-y-auto/)
    expect(form).toMatch(/data-sidepanel-dock-essential[^>]*className="[^"]*shrink-0/)
  })

  it("sizes control-row controls by pointer, not width: 32px fine, 44px coarse", () => {
    const row = source("../ControlRow.tsx")
    expect(row).not.toContain("min-h-[44px] sm:min-h-0")
    expect(row).toContain("min-h-8 [@media(pointer:coarse)]:min-h-11")
  })

  it("records the essential rows' height so the dock can grow to fit them", async () => {
    const callbacks: Array<() => void> = []
    vi.stubGlobal("ResizeObserver", class { constructor(cb: () => void) { callbacks.push(cb) } observe() {} disconnect() {} })
    try {
      const { useComposerDockHeight } = await import("@/hooks/useComposerDockHeight")
      const Dock = () => {
        const { ref } = useComposerDockHeight()
        return <div ref={ref} data-testid="dock"><div data-testid="content"><div data-sidepanel-dock-optional data-testid="optional" /></div></div>
      }
      render(<Dock />)
      // Intrinsic heights: a capped dock clips the rows, but scrollHeight still counts them.
      Object.defineProperty(screen.getByTestId("content"), "scrollHeight", { configurable: true, get: () => 300 })
      // The optional region scrolls, so only its visible box (clientHeight) is in the content's height.
      Object.defineProperty(screen.getByTestId("optional"), "scrollHeight", { configurable: true, get: () => 60 })
      Object.defineProperty(screen.getByTestId("optional"), "clientHeight", { configurable: true, get: () => 40 })
      act(() => callbacks.forEach((cb) => cb()))
      // 300 - 40 visible optional + a 56px floor of the optional region, so a hint stays reachable.
      expect(screen.getByTestId("dock").style.getPropertyValue("--sp-dock-fixed-h")).toBe("316px")
    } finally {
      vi.unstubAllGlobals()
    }
  })

  it("renders the chip and the sheet in the composer, not the inline panels", () => {
    const form = source("../form.tsx")
    expect(form).toContain("<SidepanelRuntimeChip")
    expect(form).not.toContain("<ModelParamsPanel")
    expect(form).toContain("<SidepanelKnowledgeSheet")
    expect(form).not.toContain("<KnowledgePanel")
  })
})
