import { render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { PreferencesSettings } from "../preferences-settings"

const setSettingMock = vi.fn()
const setUserPersonaMock = vi.fn()
const resetTutorialProgressMock = vi.fn()

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue ?? _key
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, fallback: unknown) => [fallback, vi.fn()]
}))

vi.mock("@/hooks/useI18n", () => ({
  useI18n: () => ({
    changeLocale: vi.fn(),
    locale: "en",
    supportLanguage: [{ label: "English", value: "en" }]
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({ success: vi.fn(), error: vi.fn() })
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => ({ userPersona: null }),
  useConnectionActions: () => ({ setUserPersona: setUserPersonaMock })
}))

vi.mock("@/store/tutorials", () => ({
  useTutorialCompletion: () => ({
    completedTutorials: [],
    resetProgress: resetTutorialProgressMock
  })
}))

vi.mock("@/hooks/useSetting", () => ({
  useSetting: () => [[], setSettingMock]
}))

vi.mock("../search-mode", () => ({
  SearchModeSettings: () => <div>Web search defaults</div>
}))

describe("PreferencesSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("owns personal defaults and web search without setup, theme, OCR, extension promo, or destructive reset", () => {
    const { container } = render(
      <MemoryRouter>
        <PreferencesSettings />
      </MemoryRouter>
    )

    expect(screen.getByText("General preferences")).toBeInTheDocument()
    expect(screen.getByText("generalSettings.settings.language.label")).toBeInTheDocument()
    expect(
      screen.getByText("generalSettings.settings.sendNotificationAfterIndexing.label")
    ).toBeInTheDocument()
    expect(screen.getByText("generalSettings.settings.ollamaStatus.label")).toBeInTheDocument()
    expect(
      screen.getByText("Auto-finish onboarding after successful connection")
    ).toBeInTheDocument()
    expect(screen.getByText("Reset tutorial progress")).toBeInTheDocument()
    expect(screen.getByText("Persona")).toBeInTheDocument()
    expect(screen.getByText("Web search defaults")).toBeInTheDocument()
    expect(screen.queryByText("Connection")).not.toBeInTheDocument()
    expect(screen.queryByText("Theme picker")).not.toBeInTheDocument()
    expect(screen.queryByText("Browser Extension Available")).not.toBeInTheDocument()
    expect(screen.queryByText(/OCR assets/i)).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: /reset all/i })
    ).not.toBeInTheDocument()
    expect(container.querySelector("dl")).not.toBeInTheDocument()
  })
})
