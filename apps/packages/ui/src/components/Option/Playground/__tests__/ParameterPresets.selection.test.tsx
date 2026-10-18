import playground from "@/assets/locale/en/playground.json"
import { useStoreChatModelSettings } from "@/store/model"
import { fireEvent, render, screen } from "@testing-library/react"
import i18next from "i18next"
import React from "react"
import { I18nextProvider } from "react-i18next"
import { beforeEach, expect, it } from "vitest"

import { ParameterPresetsDropdown } from "../ParameterPresets"

const translation = i18next.createInstance()
await translation.init({ lng: "en", resources: { en: { playground } } })
beforeEach(() => useStoreChatModelSettings.getState().reset())
it("announces the current generation choice before and after a deliberate preset change", () => {
  useStoreChatModelSettings
    .getState()
    .updateSettings({ temperature: 0.7, topP: 0.9, topK: 40 })
  render(
    <I18nextProvider i18n={translation}>
      <ParameterPresetsDropdown />
    </I18nextProvider>
  )
  expect(screen.getByRole("button", { name: /Balanced/ })).toHaveAttribute(
    "aria-pressed",
    "true"
  )
  const creative = screen.getByRole("button", { name: /Creative/ })
  expect(creative).toHaveAttribute("aria-pressed", "false")
  fireEvent.click(creative)
  expect(creative).toHaveAttribute("aria-pressed", "true")
  expect(screen.getByRole("button", { name: /Balanced/ })).toHaveAttribute(
    "aria-pressed",
    "false"
  )
})
