import React from "react"
import { describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen } from "@testing-library/react"
import { LlamacppInventoryPanel } from "../LlamacppInventoryPanel"

describe("LlamacppInventoryPanel", () => {
  it("renders inventory metadata and selects by model_id", () => {
    const onSelect = vi.fn()

    render(
      <LlamacppInventoryPanel
        inventory={{
          models: [
            {
              model_id: "gguf:stable-id",
              display_name: "Mistral 7B Instruct Q4_K_M",
              basename: "mistral-7b-instruct-q4_k_m.gguf",
              source: "registered_path",
              path: "/models/mistral-7b-instruct-q4_k_m.gguf",
              size_bytes: 4_000_000_000,
              modified_at: null,
              metadata: {
                quantization: "Q4_K_M",
                parameter_hint: "7B",
                context_hint: null
              },
              warnings: ["Outside models directory but allowed."]
            }
          ],
          warnings: [],
          scan_limited: false
        }}
        selectedModelId={undefined}
        activeModel="different.gguf"
        loading={false}
        registering={false}
        onSelectModel={onSelect}
        onRegisterPath={vi.fn()}
        onReload={vi.fn()}
      />
    )

    expect(screen.getByText("Mistral 7B Instruct Q4_K_M")).toBeTruthy()
    expect(screen.getByText("registered_path")).toBeTruthy()
    expect(screen.getByText("Q4_K_M")).toBeTruthy()
    expect(screen.getByText("Outside models directory but allowed.")).toBeTruthy()

    fireEvent.click(screen.getByRole("button", { name: "Select" }))

    expect(onSelect).toHaveBeenCalledWith("gguf:stable-id")
  })
})
