// @vitest-environment jsdom
import React from "react"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ModelParamsPanel } from "../ModelParamsPanel"
import { useStoreChatModelSettings } from "@/store/model"
import { useUiModeStore } from "@/store/ui-mode"

// XS-16: the side panel must not show invented numbers for unset parameters.
// An unset Temperature, Top P or penalty shows "Default (x)", where x is what
// the request sends (pageAssistModel.sampling-defaults.test.ts pins that), and
// the muted slider rests at x. Min P is not sent, so it shows "Default". Each
// value set for this model can be reset; a value set for all models is shown
// as such and has no reset here.

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string, options?: Record<string, unknown>) =>
      String(fallback || _key).replace(/\{\{(\w+)\}\}/g, (_match, name) =>
        String(options?.[name] ?? "")
      )
  })
}))

const modelDefaults = vi.hoisted(() => ({
  value: {} as Record<string, unknown>
}))

vi.mock("@/services/model-settings", () => ({
  getAllDefaultModelSettings: vi.fn(async () => modelDefaults.value)
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
    modelDefaults.value = {}
    useStoreChatModelSettings.getState().reset()
    useUiModeStore.setState({ mode: "pro" })
  })

  it.each([
    ["Temperature", "Default (0.7)"],
    ["Top P", "Default (1)"],
    ["Frequency Penalty", "Default (0)"],
    ["Presence Penalty", "Default (0)"],
    ["Min P", "Default"]
  ])("shows %s as %s when it is unset", async (label, readout) => {
    const panel = renderPanel()
    const row = field(panel, label)
    expect(await within(row).findByText(readout)).toBeInTheDocument()
    expect(within(row).queryByText(/^\d/)).not.toBeInTheDocument()
    expect(
      within(row).queryByRole("button", { name: `Reset ${label} to default` })
    ).not.toBeInTheDocument()
  })

  it("rests the muted Temperature slider at the value that is sent", async () => {
    const panel = renderPanel()
    const slider = within(field(panel, "Temperature")).getByRole("slider")
    await waitFor(() => expect(slider).toHaveAttribute("aria-valuenow", "0.7"))
    expect(slider).toHaveAttribute("aria-valuetext", "Default (0.7)")
  })

  it("shows the user's saved default model settings as the default", async () => {
    modelDefaults.value = { temperature: 0.4, topP: 0.8 }
    const panel = renderPanel()
    expect(
      await within(field(panel, "Temperature")).findByText("Default (0.4)")
    ).toBeInTheDocument()
    expect(within(field(panel, "Top P")).getByText("Default (0.8)")).toBeInTheDocument()
    expect(
      within(field(panel, "Temperature")).getByRole("slider")
    ).toHaveAttribute("aria-valuenow", "0.4")
  })

  it("shows a value set for all models without a reset that cannot clear it", async () => {
    useStoreChatModelSettings.getState().updateSetting("temperature", 0.5)
    useStoreChatModelSettings.getState().setActiveSettingsScope("openai/gpt-4o")
    const panel = renderPanel()
    const row = field(panel, "Temperature")
    expect(within(row).getByText("0.5 (all models)")).toBeInTheDocument()
    expect(
      within(row).queryByRole("button", { name: "Reset Temperature to default" })
    ).not.toBeInTheDocument()
  })

  it("resets a value set for this model back to the all-models value", async () => {
    useStoreChatModelSettings.getState().updateSetting("temperature", 0.5)
    useStoreChatModelSettings.getState().setActiveSettingsScope("openai/gpt-4o")
    useStoreChatModelSettings.getState().updateSetting("temperature", 1.2)
    const panel = renderPanel()
    const row = field(panel, "Temperature")
    expect(within(row).getByText("1.2")).toBeInTheDocument()

    fireEvent.click(within(row).getByRole("button", { name: "Reset Temperature to default" }))

    expect(useStoreChatModelSettings.getState().temperature).toBe(0.5)
    const after = field(panel, "Temperature")
    expect(within(after).getByText("0.5 (all models)")).toBeInTheDocument()
    expect(
      within(after).queryByRole("button", { name: "Reset Temperature to default" })
    ).not.toBeInTheDocument()
  })

  it("shows a set value and resets it to the default", async () => {
    useStoreChatModelSettings.getState().updateSetting("temperature", 1.2)
    const panel = renderPanel()
    const row = field(panel, "Temperature")
    expect(within(row).getByText("1.2")).toBeInTheDocument()

    fireEvent.click(within(row).getByRole("button", { name: "Reset Temperature to default" }))

    expect(useStoreChatModelSettings.getState().temperature).toBeUndefined()
    expect(
      await within(field(panel, "Temperature")).findByText("Default (0.7)")
    ).toBeInTheDocument()
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
