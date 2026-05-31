import React from "react"
import { describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen } from "@testing-library/react"
import { LlamacppLaunchPanel } from "../LlamacppLaunchPanel"

describe("LlamacppLaunchPanel", () => {
  it("keeps hardware warnings advisory and preserves advanced launch controls", () => {
    const onStart = vi.fn()

    render(
      <LlamacppLaunchPanel
        settings={{
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
        }}
        onSettingsChange={vi.fn()}
        selectedModelId="gguf:selected"
        isRunning={false}
        actionLoading={false}
        inventoryUnavailable={false}
        adminUnavailable={false}
        hardwareWarnings={["GPU probe unavailable."]}
        presetNotice={null}
        onStart={onStart}
        onStartWithDefaults={vi.fn()}
        onExportPreset={vi.fn()}
        onOpenImportPreset={vi.fn()}
        importPresetInput={null}
        chatAction={null}
      />
    )

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
})
