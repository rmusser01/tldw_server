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
