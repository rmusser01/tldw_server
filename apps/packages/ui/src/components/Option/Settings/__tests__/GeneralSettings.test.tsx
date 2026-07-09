import { render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { describe, expect, it, vi } from "vitest"

import { GeneralSettings } from "../general-settings"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      return fallbackOrOptions?.defaultValue ?? key
    }
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) => [defaultValue, vi.fn()]
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    error: vi.fn(),
    success: vi.fn()
  })
}))

vi.mock("@/hooks/useI18n", () => ({
  useI18n: () => ({
    changeLocale: vi.fn(),
    locale: "en",
    supportLanguage: [{ label: "English", value: "en" }]
  })
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => ({
    userPersona: null
  }),
  useConnectionActions: () => ({
    setUserPersona: vi.fn()
  })
}))

vi.mock("@/store/tutorials", () => ({
  useTutorialCompletion: () => ({
    completedTutorials: [],
    resetProgress: vi.fn()
  })
}))

vi.mock("@/utils/browser-runtime", () => ({
  isExtensionRuntime: () => false
}))

vi.mock("@/hooks/useSetting", () => ({
  useSetting: () => [[], vi.fn()]
}))

vi.mock("../search-mode", () => ({
  SearchModeSettings: () => <div>Search preferences</div>
}))

describe("GeneralSettings", () => {
  it("keeps the legacy export pointing at preferences", () => {
    render(
      <MemoryRouter>
        <GeneralSettings />
      </MemoryRouter>
    )

    expect(screen.getByText("General preferences")).toBeInTheDocument()
    expect(screen.queryByText("Connection")).not.toBeInTheDocument()
  })
})
