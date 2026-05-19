// @vitest-environment jsdom

import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { AudioReadinessStrip } from "../AudioReadinessStrip"

describe("AudioReadinessStrip", () => {
  it("renders accessible readiness status labels and source details", () => {
    render(
      <AudioReadinessStrip
        label="STT readiness"
        items={[
          {
            id: "models",
            label: "STT models",
            state: "ready",
            detail: "2 listed, 1 ready, 1 unknown",
            source: "health"
          }
        ]}
      />
    )

    expect(screen.getByRole("status", { name: "STT readiness" })).toHaveTextContent(
      "STT models: Ready"
    )
    expect(
      screen.getByLabelText(
        "STT models: Ready. 2 listed, 1 ready, 1 unknown Source: model health."
      )
    ).toBeInTheDocument()
  })
})
