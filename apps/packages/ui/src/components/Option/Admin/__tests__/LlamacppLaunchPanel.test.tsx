import React from "react"
import { describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen } from "@testing-library/react"
import { LlamacppAssetsPanel } from "../LlamacppAssetsPanel"
import { LlamacppInventoryPanel } from "../LlamacppInventoryPanel"
import { LlamacppLaunchPanel } from "../LlamacppLaunchPanel"
import { LlamacppProfilesPanel } from "../LlamacppProfilesPanel"
import { LlamacppReadinessPanel } from "../LlamacppReadinessPanel"
import { LlamacppRuntimePanel } from "../LlamacppRuntimePanel"
import type { LlamacppServerArgsInput } from "@/utils/build-llamacpp-server-args"

type LaunchPanelProps = React.ComponentProps<typeof LlamacppLaunchPanel>

const baseSettings: LlamacppServerArgsInput = {
  contextSize: 4096,
  gpuLayers: 0,
  cacheType: "f16",
  splitMode: "layer",
  rowSplit: false,
  mlock: false,
  noMmap: false,
  noKvOffload: false,
  streamingLlm: false,
  cpuMoe: false,
  mmprojAuto: true,
  mmprojOffload: true,
  flashAttn: "auto",
  customArgs: {}
}

const renderLaunchPanel = (props: Partial<LaunchPanelProps> = {}) =>
  render(
    <LlamacppLaunchPanel
      initialSettings={baseSettings}
      selectedModelId="gguf:selected"
      isRunning={false}
      actionLoading={false}
      inventoryUnavailable={false}
      adminUnavailable={false}
      hardwareWarnings={[]}
      presetNotice={null}
      onStart={vi.fn()}
      onStartWithDefaults={vi.fn()}
      onExportPreset={vi.fn()}
      onOpenImportPreset={vi.fn()}
      importPresetInput={null}
      chatAction={null}
      {...props}
    />
  )

describe("LlamacppLaunchPanel", () => {
  it("keeps hardware warnings advisory and preserves advanced launch controls", () => {
    const onStart = vi.fn()

    renderLaunchPanel({
      initialSettings: baseSettings,
      hardwareWarnings: ["GPU probe unavailable."],
      onStart
    })

    expect(screen.getByText("GPU probe unavailable.")).toBeTruthy()
    const guidanceAlert = screen
      .getByText("Hardware guidance")
      .closest('[data-ds-component="Alert"]')
    expect(guidanceAlert).toHaveAttribute("role", "status")
    expect(guidanceAlert).toHaveAttribute("aria-live", "polite")
    expect(screen.getByText("Other Options")).toBeTruthy()
    expect(screen.getByText("Multimodal (vision)")).toBeTruthy()
    expect(screen.getByText("Speculative decoding")).toBeTruthy()
    expect(screen.getByText("Network & Runtime")).toBeTruthy()
    expect(screen.getByText("Raw argument overrides")).toBeTruthy()

    fireEvent.click(screen.getByRole("button", { name: "Start Server" }))

    expect(onStart).toHaveBeenCalled()
  })

  it("test_six_llamacpp_panels_are_react_memo_wrapped", () => {
    const memoPanels = [
      LlamacppAssetsPanel,
      LlamacppInventoryPanel,
      LlamacppLaunchPanel,
      LlamacppProfilesPanel,
      LlamacppReadinessPanel,
      LlamacppRuntimePanel
    ]
    for (const Panel of memoPanels) {
      expect((Panel as unknown as { $$typeof: symbol }).$$typeof).toBe(
        Symbol.for("react.memo")
      )
    }
  })

  it("test_launch_panel_edits_stay_local_and_sync_settings_ref", () => {
    const settingsRef: React.MutableRefObject<LlamacppServerArgsInput> = {
      current: { ...baseSettings }
    }

    renderLaunchPanel({ settingsRef })

    fireEvent.click(screen.getByText("Other Options"))
    const tensorSplitInput = screen.getByPlaceholderText(
      "e.g. 38,62"
    ) as HTMLInputElement
    fireEvent.change(tensorSplitInput, { target: { value: "38,62" } })

    expect(tensorSplitInput.value).toBe("38,62")
    expect(settingsRef.current.tensorSplit).toBe("38,62")
  })

  it("test_launch_panel_applies_imported_preset_when_token_changes", () => {
    const { rerender } = renderLaunchPanel({
      appliedPreset: {
        token: 7,
        settings: { ...baseSettings, contextSize: 16384, gpuLayers: 10 }
      }
    })

    expect(
      (screen.getByDisplayValue("16384") as HTMLInputElement).value
    ).toBe("16384")
    expect(
      (screen.getByDisplayValue("10") as HTMLInputElement).value
    ).toBe("10")

    rerender(
      <LlamacppLaunchPanel
        initialSettings={baseSettings}
        selectedModelId="gguf:selected"
        isRunning={false}
        actionLoading={false}
        inventoryUnavailable={false}
        adminUnavailable={false}
        hardwareWarnings={[]}
        presetNotice={null}
        onStart={vi.fn()}
        onStartWithDefaults={vi.fn()}
        onExportPreset={vi.fn()}
        onOpenImportPreset={vi.fn()}
        importPresetInput={null}
        chatAction={null}
        appliedPreset={{
          token: 8,
          settings: { ...baseSettings, contextSize: 2048 }
        }}
      />
    )

    expect(
      (screen.getByDisplayValue("2048") as HTMLInputElement).value
    ).toBe("2048")
  })
})
