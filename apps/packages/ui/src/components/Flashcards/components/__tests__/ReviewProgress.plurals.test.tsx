import React from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { createInstance } from "i18next"
import { I18nextProvider } from "react-i18next"
import { afterEach, describe, expect, it } from "vitest"

import ICU from "@/i18n/icu-format"
import option from "@/assets/locale/en/option.json"
import { ReviewProgress } from "../ReviewProgress"

afterEach(cleanup)

describe("ReviewProgress localized remaining count", () => {
  it.each([
    { remaining: 1, label: "card remaining", announcement: "1 card remaining, 4 reviewed" },
    { remaining: 2, label: "cards remaining", announcement: "2 cards remaining, 4 reviewed" }
  ])("keeps visible and announced wording aligned at $remaining", async ({ remaining, label, announcement }) => {
    const i18n = createInstance().use(ICU)
    await i18n.init({ lng: "en", resources: { en: { option } }, interpolation: { escapeValue: false } })
    render(
      <I18nextProvider i18n={i18n}>
        <ReviewProgress remainingCount={remaining} reviewedCount={4} />
      </I18nextProvider>
    )

    expect(screen.getByText(label, { exact: true })).toBeVisible()
    expect(screen.getByRole("status").querySelector(".sr-only")).toHaveTextContent(announcement)
  })

  it("does not announce remaining work for an empty queue", async () => {
    const i18n = createInstance().use(ICU)
    await i18n.init({ lng: "en", resources: { en: { option } } })
    render(
      <I18nextProvider i18n={i18n}>
        <ReviewProgress remainingCount={0} reviewedCount={5} />
      </I18nextProvider>
    )
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
  })
})
