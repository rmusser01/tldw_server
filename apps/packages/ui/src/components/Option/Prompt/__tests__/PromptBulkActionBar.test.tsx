import React from "react"
import { describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import { PromptBulkActionBar } from "../PromptBulkActionBar"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: { defaultValue?: string }) =>
      key === "option:bulkActions" ? "Translated bulk actions" : options?.defaultValue ?? key
  })
}))

describe("PromptBulkActionBar", () => {
  it("uses the localized bulk actions aria-label for the legacy toolbar", () => {
    render(
      <PromptBulkActionBar mode="legacy">
        <button type="button">Legacy action</button>
      </PromptBulkActionBar>
    )

    expect(screen.getByTestId("prompts-bulk-action-bar-legacy")).toHaveAttribute(
      "aria-label",
      "Translated bulk actions"
    )
  })

  it("uses the localized bulk actions aria-label for the scaffold toolbar", () => {
    render(
      <PromptBulkActionBar
        selectedCount={2}
        onBulkExport={vi.fn()}
        onBulkTag={vi.fn()}
        onBulkFavoriteToggle={vi.fn()}
        onBulkDelete={vi.fn()}
        onClearSelection={vi.fn()}
      />
    )

    expect(screen.getByTestId("prompts-bulk-action-bar-scaffold")).toHaveAttribute(
      "aria-label",
      "Translated bulk actions"
    )
  })
})
