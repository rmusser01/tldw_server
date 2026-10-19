// @vitest-environment jsdom
import React from "react"
import { act, fireEvent, render, screen, within } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ModelParamsPanel } from "../ModelParamsPanel"
import { useStoreChatModelSettings } from "@/store/model"
import { useUiModeStore } from "@/store/ui-mode"

// XS-16: unset parameters are sent as "provider default", so the side panel
// must not show invented numbers for them. It shows "Default" until the user
// sets a value, and each set value can be reset.

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string, options?: Record<string, unknown>) =>
      String(fallback || _key).replace(/\{\{(\w+)\}\}/g, (_match, name) =>
        String(options?.[name] ?? "")
      )
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { getLlmProviders: vi.fn(async () => ({ providers: {} })) }
}))

const renderPanel = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={queryClient}>
      <ModelParamsPanel selectedModel="openai/gpt-4o" />
    </QueryClientProvider>
  )
  fireEvent.click(screen.getByRole("button", { name: "Model Parameters" }))
  return screen.getByTestId("model-params-panel")
}

const field = (panel: HTMLElement, label: string) =>
  within(panel).getByText(label).closest("[data-param-field]") as HTMLElement

describe("ModelParamsPanel inherited defaults (XS-16)", () => {
  beforeEach(() => {
    useStoreChatModelSettings.getState().reset()
    useUiModeStore.setState({ mode: "pro" })
  })

  it.each(["Temperature", "Top P", "Frequency Penalty", "Presence Penalty", "Min P"])(
    "shows %s as Default when it is unset",
    (label) => {
      const panel = renderPanel()
      const row = field(panel, label)
      expect(within(row).getByText("Default")).toBeInTheDocument()
      expect(within(row).queryByText(/^\d/)).not.toBeInTheDocument()
      expect(
        within(row).queryByRole("button", { name: `Reset ${label} to default` })
      ).not.toBeInTheDocument()
    }
  )

  it("shows a set value and resets it to the provider default", () => {
    useStoreChatModelSettings.getState().updateSetting("temperature", 1.2)
    const panel = renderPanel()
    const row = field(panel, "Temperature")
    expect(within(row).getByText("1.2")).toBeInTheDocument()

    fireEvent.click(within(row).getByRole("button", { name: "Reset Temperature to default" }))

    expect(useStoreChatModelSettings.getState().temperature).toBeUndefined()
    expect(within(field(panel, "Temperature")).getByText("Default")).toBeInTheDocument()
  })

  it("clears Top K back to the provider default instead of inventing 40", () => {
    useStoreChatModelSettings.getState().updateSetting("topK", 12)
    const panel = renderPanel()
    const input = within(field(panel, "Top K")).getByRole("spinbutton")
    act(() => {
      fireEvent.change(input, { target: { value: "" } })
      fireEvent.blur(input)
    })
    expect(useStoreChatModelSettings.getState().topK).toBeUndefined()
    expect(input).toHaveAttribute("placeholder", "Default")
  })
})
