import React from "react"
import { describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      defaultValueOrOptions?:
        | string
        | { defaultValue?: string; [key: string]: unknown },
      interpolation?: Record<string, unknown>
    ) => {
      if (typeof defaultValueOrOptions === "string") {
        return defaultValueOrOptions.replace(/\{\{(\w+)\}\}/g, (_m, token) =>
          String(interpolation?.[token] ?? "")
        )
      }
      return String(defaultValueOrOptions?.defaultValue || "")
    }
  })
}))

import { PresetSelector } from "../PresetSelector"

const qi = (
  _key: string,
  defaultValue: string,
  options?: Record<string, unknown>
) =>
  defaultValue.replace(/\{\{(\w+)\}\}/g, (_m, token) =>
    String(options?.[token] ?? "")
  )

describe("PresetSelector", () => {
  it("does not predict model duration from a tiny document size", () => {
    render(<PresetSelector qi={qi} value="standard" onChange={vi.fn()} queueItems={[{
      id: "cedar", kind: "file", fileName: "cedar.txt", detectedType: "document",
      icon: "File", fileSize: 275, validation: { valid: true },
    }]} />)
    expect(screen.queryByText(/~\d+ (sec|min|hr)/)).not.toBeInTheDocument()
  })

  it("explains that presets are optional starting points and custom changes are allowed", () => {
    render(
      <PresetSelector
        qi={qi}
        value="standard"
        onChange={vi.fn()}
      />
    )

    expect(
      screen.getByText(
        /presets are starting points\. adjust any settings below or in advanced options/i
      )
    ).toBeInTheDocument()
  })
})
