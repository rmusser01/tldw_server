import React from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { VoiceTranscriptComposer } from "../VoiceTranscriptComposer"

const mocks = vi.hoisted(() => ({
  t: vi.fn(
    (
      _key: string,
      options?: { defaultValue?: string } | string
    ) => {
      if (typeof options === "string") return options
      return options?.defaultValue ?? _key
    }
  ),
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: mocks.t,
  }),
}))

describe("VoiceTranscriptComposer product-state UI", () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it("renders unsupported voice state through the canonical design-system Alert", () => {
    render(
      <VoiceTranscriptComposer
        transcript=""
        isListening={false}
        supported={false}
        isSubmitting={false}
        onTranscriptChange={vi.fn()}
        onStartListening={vi.fn()}
        onStopListening={vi.fn()}
        onCancel={vi.fn()}
        onSubmit={vi.fn()}
      />
    )

    const unavailableMessage = screen.getByText(
      "Voice transcript is unavailable in this browser."
    )

    expect(unavailableMessage).toBeInTheDocument()
    expect(
      unavailableMessage.closest('[data-ds-component="Alert"]')
    ).toBeInTheDocument()
    expect(screen.getByRole("alert")).toContainElement(unavailableMessage)
  })
})
