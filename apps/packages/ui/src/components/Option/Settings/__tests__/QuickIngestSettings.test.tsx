import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { QuickIngestSettings } from "../QuickIngestSettings"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue ?? _key
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, fallback: unknown) => [fallback, vi.fn()]
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: () => ({ data: [], isLoading: false })
}))

describe("QuickIngestSettings", () => {
  it("owns OCR asset and default language settings", () => {
    render(<QuickIngestSettings />)

    expect(screen.getByText("OCR defaults")).toBeInTheDocument()
    expect(screen.getByText("generalSettings.settings.enableOcrAssets.label")).toBeInTheDocument()
    expect(screen.getByText("generalSettings.settings.ocrLanguage.label")).toBeInTheDocument()
  })
})
