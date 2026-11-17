import { render, screen } from "@testing-library/react"
import { createInstance } from "i18next"
import { I18nextProvider } from "react-i18next"
import { expect, it, vi } from "vitest"

import common from "@/assets/locale/en/common.json"
import option from "@/assets/locale/en/option.json"
import settings from "@/assets/locale/en/settings.json"
import ICUWithInterpolation from "@/i18n/icu-format"
import { SystemSettings } from "../system-settings"

vi.mock("@/context/FontSizeProvider", () => ({
  useFontSize: () => ({ increase: vi.fn(), decrease: vi.fn(), scale: 1 })
}))
vi.mock("@/hooks/chat/useClearChat", () => ({ useClearChat: () => vi.fn() }))
vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({ error: vi.fn() })
}))
vi.mock("@/hooks/useSetting", () => ({
  useSetting: (setting: { defaultValue: unknown }) => [
    setting.defaultValue,
    vi.fn()
  ]
}))
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, fallback: unknown) => [fallback, vi.fn()]
}))

it("labels background upload with only the System route's loaded English resources", async () => {
  const i18n = createInstance()
  await i18n.use(ICUWithInterpolation).init({
    lng: "en",
    fallbackLng: false,
    ns: ["common", "option", "settings"],
    defaultNS: "settings",
    resources: { en: { common, option, settings } },
    interpolation: { escapeValue: false }
  })
  render(
    <I18nextProvider i18n={i18n}>
      <SystemSettings />
    </I18nextProvider>
  )
  expect(
    screen.getByLabelText("Upload image", { selector: "input[type=file]" })
  ).toBeInTheDocument()
  expect(screen.queryByText("form.uploadFile.label")).not.toBeInTheDocument()
})
