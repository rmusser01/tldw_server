import fs from "node:fs"
import path from "node:path"
import React from "react"
import { render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: (selector?: (state: any) => unknown) => {
    const state = {
      temperature: 0.7,
      topP: 0.9,
      topK: 40,
      frequencyPenalty: 0,
      presencePenalty: 0,
      repeatPenalty: 1,
      updateSettings: vi.fn()
    }
    return selector ? selector(state) : state
  }
}))

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) =>
    React.createElement(React.Fragment, null, children),
  Segmented: ({
    options,
    value,
    onChange,
    "aria-label": ariaLabel
  }: {
    options: Array<{ label: React.ReactNode; value: string }>
    value: string
    onChange: (value: string) => void
    "aria-label"?: string
  }) =>
    React.createElement(
      "div",
      { role: "radiogroup", "aria-label": ariaLabel },
      options.map((option) =>
        React.createElement(
          "label",
          { key: option.value },
          React.createElement("input", {
            type: "radio",
            checked: value === option.value,
            onChange: () => onChange(option.value)
          }),
          option.label
        )
      )
    )
}))

import { ParameterPresets } from "../ParameterPresets"

describe("ParameterPresets guard", () => {
  it("keeps explicit preset parameter detail rows in tooltip content", () => {
    const sourcePath = path.resolve(__dirname, "../ParameterPresets.tsx")
    const source = fs.readFileSync(sourcePath, "utf8")

    expect(source).toContain("formatPresetSettingEntries")
    expect(source).toContain("PRESET_SETTING_LABELS")
    expect(source).toContain("Frequency penalty")
    expect(source).toContain("Presence penalty")
  })

  it("labels compact segmented controls and icon-only options for assistive tech", () => {
    render(React.createElement(ParameterPresets, { compact: true }))

    const group = screen.getByRole("radiogroup", {
      name: "Generation style"
    })
    expect(
      within(group).getByRole("radio", { name: "Creative" })
    ).toBeInTheDocument()
    expect(
      within(group).getByRole("radio", { name: "Balanced" })
    ).toBeInTheDocument()
    expect(
      within(group).getByRole("radio", { name: "Precise" })
    ).toBeInTheDocument()
    expect(
      within(group).getByRole("radio", { name: "Custom" })
    ).toBeInTheDocument()
  })
})
