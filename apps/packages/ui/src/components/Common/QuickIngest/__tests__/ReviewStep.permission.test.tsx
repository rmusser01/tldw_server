import { render, screen } from "@testing-library/react"
import React from "react"
import { describe, expect, it, vi } from "vitest"

import { IngestWizardProvider } from "../IngestWizardContext"
import { ReviewStep } from "../ReviewStep"
import { DEFAULT_PRESETS } from "../presets"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, value: any) =>
      typeof value === "string" ? value : value.defaultValue
  })
}))
describe("replacement permission review", () => {
  it.each([false, true])(
    "shows explicit replacement choice independent of Deep (%s)",
    (overwrite) => {
      render(
        <IngestWizardProvider
          initialState={{
            selectedPreset: "deep",
            presetConfig: {
              ...DEFAULT_PRESETS.deep,
              common: {
                ...DEFAULT_PRESETS.deep.common,
                overwrite_existing: overwrite
              }
            }
          }}>
          <ReviewStep />
        </IngestWizardProvider>
      )
      expect(
        screen.getByText(
          overwrite
            ? "Replacement allowed: matching saved sources may be replaced. Affected saved items have not been confirmed."
            : "Replacement disabled: existing saved sources are preserved."
        )
      ).toBeInTheDocument()
    }
  )
})
