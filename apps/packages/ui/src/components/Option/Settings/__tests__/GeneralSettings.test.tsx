import { render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { GeneralSettings } from "../general-settings"

const setStorageMock = vi.fn()
const setSettingMock = vi.fn()
const restartOnboardingMock = vi.fn()
const setUserPersonaMock = vi.fn()
const resetTutorialProgressMock = vi.fn()
const mutateMock = vi.fn()
const storageOverrides = new Map<string, unknown>()

const expectDesignSystemAlert = (text: string | RegExp) => {
  const node =
    typeof text === "string"
      ? screen.getByText(text, { exact: false })
      : screen.getByText(text)

  expect(node.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
}

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
  useStorage: (key: string, defaultValue: unknown) => [
    storageOverrides.has(key)
      ? storageOverrides.get(key)
      : key === "settingsIntroDismissed"
        ? true
        : defaultValue,
    setStorageMock
  ]
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
    serverUrl: "http://127.0.0.1:8000",
    userPersona: null
  }),
  useConnectionActions: () => ({
    restartOnboarding: restartOnboardingMock,
    setUserPersona: setUserPersonaMock
  })
}))

vi.mock("@/store/tutorials", () => ({
  useTutorialCompletion: () => ({
    completedTutorials: [],
    resetProgress: resetTutorialProgressMock
  })
}))

vi.mock("@/utils/browser-runtime", () => ({
  isExtensionRuntime: () => false
}))

vi.mock("@/hooks/useSetting", () => ({
  useSetting: () => [[], setSettingMock]
}))

vi.mock("@/context/FontSizeProvider", () => ({
  useFontSize: () => ({
    decrease: vi.fn(),
    increase: vi.fn(),
    scale: 1
  })
}))

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => ({
    clearChat: vi.fn()
  })
}))

vi.mock("@tanstack/react-query", () => ({
  useMutation: () => ({
    isPending: false,
    mutate: mutateMock
  }),
  useQueryClient: () => ({
    invalidateQueries: vi.fn()
  })
}))

vi.mock("@/utils/is-private-mode", () => ({
  isFireFox: false,
  isFireFoxPrivateMode: false
}))

vi.mock("@/components/Common/Settings/ThemePicker", () => ({
  ThemePicker: () => <div>Theme picker</div>
}))

vi.mock("../search-mode", () => ({
  SearchModeSettings: () => <div>Search preferences</div>
}))

describe("GeneralSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    storageOverrides.clear()
  })

  it("keeps routine preferences separate from destructive data actions", () => {
    render(
      <MemoryRouter>
        <GeneralSettings />
      </MemoryRouter>
    )

    expect(screen.getByText("Connection")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /reset all/i })).not.toBeInTheDocument()
  })

  it("renders extension promotion and disabled OCR assets guidance through design-system alerts", () => {
    render(
      <MemoryRouter>
        <GeneralSettings />
      </MemoryRouter>
    )

    expectDesignSystemAlert("Browser Extension Available")
    expectDesignSystemAlert("Get the tldw browser extension")
    expect(screen.getByRole("link", { name: "Learn More" })).toHaveAttribute(
      "href",
      "https://github.com/rmusser01/tldw_server"
    )
    expectDesignSystemAlert(
      "Enable to download OCR language assets for image text recognition"
    )
  })

  it("renders enabled OCR assets guidance through the design-system alert", () => {
    storageOverrides.set("enableOcrAssets", true)

    render(
      <MemoryRouter>
        <GeneralSettings />
      </MemoryRouter>
    )

    expectDesignSystemAlert("OCR assets enabled and ready")
  })
})
