// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  IngestWizardProvider,
  useIngestWizard,
} from "@/components/Common/QuickIngest/IngestWizardContext"
import { resolvePresetMap } from "@/components/Common/QuickIngest/presets"
import { WizardConfigureStep } from "@/components/Common/QuickIngest/WizardConfigureStep"

const mocks = vi.hoisted(() => ({
  getProvidersStatus: vi.fn(),
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?: string | { defaultValue?: string }
    ) =>
      typeof defaultValueOrOptions === "string"
        ? defaultValueOrOptions
        : defaultValueOrOptions?.defaultValue ?? key,
  }),
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getProvidersStatus: (...args: unknown[]) =>
      mocks.getProvidersStatus(...args),
    getTranscriptionModels: vi.fn().mockResolvedValue({ all_models: [] }),
  },
}))

vi.mock("@/components/Common/QuickIngest/PresetSelector", () => ({
  PresetSelector: () => null,
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

  it("offers a run-only skip for the missing provider before commit", async () => {
    const user = userEvent.setup()
    const savedStandard = resolvePresetMap().standard
    render(
      <IngestWizardProvider initialState={{ presetConfig: savedStandard }}>
        <WizardConfigureStep inline />
        <RunOptions />
      </IngestWizardProvider>
    )

    const provider = screen.getByRole("combobox", { name: "Analysis provider" })
    expect(provider).toHaveAttribute("aria-invalid", "true")
    expect(screen.queryByRole("button", { name: "Next" })).toBeNull()
    await user.click(screen.getByRole("button", { name: "Skip analysis for this import" }))
    expect(screen.queryByRole("combobox", { name: "Analysis provider" })).toBeNull()
    expect(screen.getByTestId("run-analysis")).toHaveTextContent("off")
    expect(savedStandard.common.perform_analysis).toBe(true)
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
    await user.selectOptions(destination, screen.getByRole("option", { name: "Create review drafts" }))
    expect(screen.getByTestId("destination")).toHaveTextContent("[true,true]")
    await user.selectOptions(destination, screen.getByRole("option", { name: "Extract without saving" }))
    expect(screen.getByTestId("destination")).toHaveTextContent("[false,false]")
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
