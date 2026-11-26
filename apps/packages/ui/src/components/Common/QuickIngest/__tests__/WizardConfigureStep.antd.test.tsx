// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  IngestWizardProvider,
  useIngestWizard,
} from "@/components/Common/QuickIngest/IngestWizardContext"
import { resolvePresetMap } from "@/components/Common/QuickIngest/presets"
import { WizardConfigureStep } from "@/components/Common/QuickIngest/WizardConfigureStep"
import { WebIngestDraft } from "../WebIngestDraft"
import type { WizardQueueItem } from "../types"

const mocks = vi.hoisted(() => ({
  getProvidersStatus: vi.fn(),
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?: string | { defaultValue?: string; destination?: string }
    ) =>
      typeof defaultValueOrOptions === "string"
        ? defaultValueOrOptions
        : (defaultValueOrOptions?.defaultValue ?? key).replace("{{destination}}", defaultValueOrOptions?.destination ?? ""),
  }),
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getProvidersStatus: (...args: unknown[]) =>
      mocks.getProvidersStatus(...args),
    getTranscriptionModels: vi.fn().mockResolvedValue({ all_models: [] }),
  },
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({ capabilities: {} }),
}))

const ProviderValue = () => {
  const { state } = useIngestWizard()
  return (
    <output data-testid="selected-analysis-provider">
      {String(state.presetConfig.advancedValues?.api_name ?? "")}
    </output>
  )
}

const RunOptions = () => {
  const { state } = useIngestWizard()
  return <output data-testid="run-analysis">{state.presetConfig.common.perform_analysis ? "on" : "off"}</output>
}

const sources: WizardQueueItem[] = [{
  id: "notes", kind: "url", url: "https://example.org/notes", detectedType: "web",
  icon: "Globe", fileSize: 0, validation: { valid: true },
}]

function renderWebDraft(isOnlineForIngest: boolean, onRetryConnection = vi.fn()) {
  const savedStandard = resolvePresetMap().standard
  render(
    <IngestWizardProvider initialState={{ presetConfig: savedStandard, queueItems: sources }}>
      <div role="dialog">
        <WebIngestDraft processingItems={sources} replacementItems={[]} visible
          isOnlineForIngest={isOnlineForIngest} isCheckingConnection={false}
          onRetryConnection={onRetryConnection} connectionRecoveryMessage="Cannot reach the server."
          onProcess={vi.fn()} />
      </div>
      <RunOptions />
    </IngestWizardProvider>
  )
  return savedStandard
}

describe("WizardConfigureStep Ant Design provider selection", () => {
  beforeEach(() => {
    mocks.getProvidersStatus.mockReset()
    mocks.getProvidersStatus.mockResolvedValue({
      providers: [
        { name: "openai", configured: true },
        { name: "anthropic", configured: true },
      ],
      any_configured: true,
    })
  })

  it("puts both provider remedies beside submission and focuses the enabled action after a run-only skip", async () => {
    const user = userEvent.setup()
    const savedStandard = renderWebDraft(true)

    const actions = within(screen.getByRole("group", { name: "Import actions" }))
    const submit = actions.getByRole("button", { name: "Save 1 source" })
    expect(submit).toBeDisabled()
    expect(screen.getAllByText("AI analysis needs a provider")).toEqual([actions.getByText("AI analysis needs a provider")])
    const provider = screen.getByRole("combobox", { name: "Analysis provider" })
    expect(provider).toHaveAttribute("aria-invalid", "true")
    await user.click(actions.getByRole("button", { name: "Choose provider" }))
    expect(provider).toHaveFocus()
    await user.click(actions.getByRole("button", { name: "Skip analysis for this import" }))
    expect(submit).toBeEnabled()
    expect(submit).toHaveFocus()
    expect(screen.queryByRole("combobox", { name: "Analysis provider" })).toBeNull()
    expect(screen.getByTestId("run-analysis")).toHaveTextContent("off")
    expect(savedStandard.common.perform_analysis).toBe(true)
  })

  it("prioritizes connection recovery in the footer and keeps the source editor available", async () => {
    const user = userEvent.setup()
    const retry = vi.fn()
    renderWebDraft(false, retry)
    const actions = within(screen.getByRole("group", { name: "Import actions" }))
    const submit = actions.getByRole("button", { name: "Save 1 source" })
    expect(submit).toBeDisabled()
    expect(submit).toHaveAccessibleDescription(/server offline/i)
    expect(actions.queryByRole("button", { name: "Choose provider" })).toBeNull()
    const linkInput = screen.getByRole("textbox", { name: "URL input area" })
    await user.type(linkInput, "https://example.org/next")
    expect(linkInput).toHaveValue("https://example.org/next")
    await user.click(actions.getByRole("button", { name: "Retry connection" }))
    expect(retry).toHaveBeenCalledTimes(1)
  })

  it("keeps the compact provider editor focused until a custom name is complete", async () => {
    const user = userEvent.setup()
    render(
      <IngestWizardProvider initialState={{ presetConfig: resolvePresetMap().standard }}>
        <WizardConfigureStep inline />
        <ProviderValue />
      </IngestWizardProvider>
    )
    const provider = screen.getByRole("combobox", { name: "Analysis provider" })
    await user.type(provider, "local-service")
    expect(provider).toHaveFocus()
    expect(screen.getByTestId("selected-analysis-provider")).toHaveTextContent("local-service")
    expect(provider).not.toHaveAttribute("aria-invalid", "true")
  })

  it("shows relevant advanced settings without separate step navigation", async () => {
    const user = userEvent.setup()
    render(
      <IngestWizardProvider initialState={{ presetConfig: resolvePresetMap().quick }}>
        <WizardConfigureStep inline expanded processingItems={[{
          id: "doc", fileName: "notes.pdf", icon: "FileText", detectedType: "pdf",
          fileSize: 10, validation: { valid: true }
        }]} />
      </IngestWizardProvider>
    )
    await user.click(screen.getByRole("button", { name: "Advanced options" }))
    expect(screen.getByRole("switch", { name: "OCR toggle" })).toBeVisible()
    expect(screen.queryByRole("combobox", { name: "Audio language" })).toBeNull()
    expect(screen.queryByRole("button", { name: "Next" })).toBeNull()
  })

  it("lets a user select review-first and extract-only destinations without opening advanced settings", async () => {
    const user = userEvent.setup()
    const Destination = () => {
      const { state: { presetConfig } } = useIngestWizard()
      return <output data-testid="destination">{JSON.stringify([presetConfig.storeRemote, presetConfig.reviewBeforeStorage])}</output>
    }
    render(
      <IngestWizardProvider initialState={{ presetConfig: resolvePresetMap().quick }}>
        <WizardConfigureStep inline expanded />
        <Destination />
      </IngestWizardProvider>
    )
    const destination = screen.getByRole("combobox", { name: "Destination" })
    const overwrite = () => screen.queryByRole("switch", { name: /overwrite existing/i })
    expect(overwrite()).toBeVisible()
    await user.selectOptions(destination, screen.getByRole("option", { name: "Create review drafts" }))
    expect(screen.getByTestId("destination")).toHaveTextContent("[true,true]")
    expect(overwrite()).toBeNull()
    await user.selectOptions(destination, screen.getByRole("option", { name: "Extract without saving" }))
    expect(screen.getByTestId("destination")).toHaveTextContent("[false,false]")
    expect(overwrite()).toBeNull()
    await user.selectOptions(screen.getByRole("combobox", { name: "Processing preset" }), "quick")
    expect(destination).toHaveValue("media")
    expect(screen.getByText("This preset changed the destination to Save to Media.")).toHaveAttribute("role", "status")
    expect(overwrite()).toBeVisible()
  })

  it("selects a configured provider from the real AutoComplete with the keyboard", async () => {
    const user = userEvent.setup()
    render(
      <IngestWizardProvider
        initialState={{
          currentStep: 2,
          highestStep: 2,
          selectedPreset: "standard",
          customBasePreset: "standard",
          presetConfig: resolvePresetMap().standard,
        }}
      >
        <WizardConfigureStep />
        <ProviderValue />
      </IngestWizardProvider>
    )

    await waitFor(() => {
      expect(mocks.getProvidersStatus).toHaveBeenCalledTimes(1)
    })
    const provider = screen.getByRole("combobox", {
      name: "Analysis provider",
    })
    expect(provider.closest(".ant-select")).toHaveClass("w-full")
    await user.click(provider)
    await screen.findByRole("option", { name: "openai" })
    fireEvent.keyDown(provider, {
      key: "ArrowDown",
      code: "ArrowDown",
      keyCode: 40,
      which: 40,
    })
    fireEvent.keyDown(provider, {
      key: "Enter",
      code: "Enter",
      keyCode: 13,
      which: 13,
    })

    await waitFor(() => {
      expect(screen.getByTestId("selected-analysis-provider")).toHaveTextContent(
        "openai"
      )
    })
  })
})
