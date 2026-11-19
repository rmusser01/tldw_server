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
import { act, fireEvent, render, screen, within } from "@testing-library/react"
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

  it("says the model's defaults are in use when nothing is set here", () => {
    render(withQuery(<SidepanelRuntimeChip selectedModel="openai/gpt-4o" />))
    expect(screen.getByRole("button", { name: /Runtime/ })).toHaveTextContent("Model defaults")
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

  it("keeps the panel mounted while closed, so its open event still arrives", () => {
    render(<SidepanelKnowledgeSheet open={false} onOpenChange={vi.fn()} />)
    expect(screen.queryByRole("dialog")).toBeNull()
    expect(document.querySelector('[data-testid="sidepanel-knowledge-sheet"]')).not.toBeNull()
  })
})

describe("Pro composer share budget (source check)", () => {
  it("caps the Pro textarea at about 96px", () => {
    expect(COMPOSER_CONSTANTS.TEXTAREA_MAX_HEIGHT_PRO).toBeLessThanOrEqual(96)
  })

  it("caps the pinned dock at 40% so the conversation keeps at least 60%", () => {
    const route = source("../../../../routes/sidepanel-chat.tsx")
    expect(route).toContain("max-h-[40%]")
    expect(route).not.toContain("max-h-[60%]")
  })

  it("renders the chip and the sheet in the composer, not the inline panels", () => {
    const form = source("../form.tsx")
    expect(form).toContain("<SidepanelRuntimeChip")
    expect(form).not.toContain("<ModelParamsPanel")
    expect(form).toContain("<SidepanelKnowledgeSheet")
    expect(form).not.toContain("<KnowledgePanel")
  })
})
