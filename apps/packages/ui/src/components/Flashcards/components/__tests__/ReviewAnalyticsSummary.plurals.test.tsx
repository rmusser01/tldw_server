import { cleanup, render, screen } from "@testing-library/react"
import { createInstance } from "i18next"
import { I18nextProvider } from "react-i18next"
import { afterEach, describe, expect, it } from "vitest"

import option from "@/assets/locale/en/option.json"
import ICU from "@/i18n/icu-format"
import type { FlashcardAnalyticsSummary } from "@/services/flashcards"
import { ReviewAnalyticsSummary } from "../ReviewAnalyticsSummary"

afterEach(cleanup)

const summaryFor = (days: number): FlashcardAnalyticsSummary => ({
  reviewed_today: 12,
  retention_rate_today: 87.5,
  lapse_rate_today: 12.5,
  avg_answer_time_ms_today: 1900,
  study_streak_days: days,
  generated_at: "2026-09-17T01:00:00Z",
  decks: []
})

const localizer = async (resourcePresent: boolean) => {
  const flashcards: Record<string, unknown> = { ...option.flashcards }
  if (!resourcePresent) delete flashcards.studyStreakDays
  const instance = createInstance().use(ICU)
  await instance.init({
    lng: "en",
    fallbackLng: false,
    resources: { en: { option: { ...option, flashcards } } },
    interpolation: { escapeValue: false }
  })
  return instance
}

describe.each([
  { resourcePresent: true, source: "production English resource" },
  { resourcePresent: false, source: "missing-key component fallback" }
])("ReviewAnalyticsSummary streak with $source", ({ resourcePresent }) => {
  it.each([
    { days: 0, label: "0 days" },
    { days: 1, label: "1 day" },
    { days: 6, label: "6 days" }
  ])("renders the supplied $days-day value with actual ICU", async ({ days, label }) => {
    const instance = await localizer(resourcePresent)
    render(
      <I18nextProvider i18n={instance}>
        <ReviewAnalyticsSummary summary={summaryFor(days)} />
      </I18nextProvider>
    )

    expect(screen.getByText(label, { exact: true })).toBeVisible()
  })
})

it("updates the same streak metric without changing other analytics values", async () => {
  const instance = await localizer(true)
  const view = render(
    <I18nextProvider i18n={instance}>
      <ReviewAnalyticsSummary summary={summaryFor(0)} />
    </I18nextProvider>
  )
  expect(screen.getByText("0 days", { exact: true })).toBeVisible()

  for (const [days, label] of [[1, "1 day"], [6, "6 days"]] as const) {
    view.rerender(
      <I18nextProvider i18n={instance}>
        <ReviewAnalyticsSummary summary={summaryFor(days)} />
      </I18nextProvider>
    )
    expect(screen.getByText(label, { exact: true })).toBeVisible()
    expect(screen.getByText("12", { exact: true })).toBeVisible()
    expect(screen.getByText("87.5%", { exact: true })).toBeVisible()
    expect(screen.getByText("12.5%", { exact: true })).toBeVisible()
    expect(screen.getByText("1.9s", { exact: true })).toBeVisible()
  }
})
