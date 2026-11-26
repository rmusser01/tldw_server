import { act, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { createInstance } from "i18next"
import { I18nextProvider } from "react-i18next"
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest"

import option from "@/assets/locale/en/option.json"
import settings from "@/assets/locale/en/settings.json"
import type { PresetMap } from "@/components/Common/QuickIngest/presets"
import ICUWithInterpolation from "@/i18n/icu-format"
import { Storage } from "../../../../../../../tldw-frontend/extension/shims/plasmo-storage"
import { QuickIngestSettings } from "../QuickIngestSettings"

vi.mock("@plasmohq/storage/hook", () =>
  import("../../../../../../../tldw-frontend/extension/shims/plasmo-storage-hook")
)
vi.mock("@plasmohq/storage", () =>
  import("../../../../../../../tldw-frontend/extension/shims/plasmo-storage")
)

vi.mock("@tanstack/react-query", () => ({
  useQuery: () => ({ data: [], isLoading: false })
}))

const i18n = createInstance()
const presetsKey = "quickIngestPresetConfigs"
const storedPresets = (): PresetMap =>
  JSON.parse(window.localStorage.getItem(presetsKey)!)
const renderSettings = () =>
  render(
    <I18nextProvider i18n={i18n}>
      <QuickIngestSettings />
    </I18nextProvider>
  )

beforeAll(async () => {
  await i18n.use(ICUWithInterpolation).init({
    lng: "en",
    fallbackLng: false,
    ns: ["settings", "option"],
    defaultNS: "settings",
    resources: { en: { settings, option } },
    interpolation: { escapeValue: false }
  })
})

beforeEach(() => window.localStorage.clear())

describe("QuickIngestSettings", () => {
  it("owns OCR asset and default language settings", async () => {
    const user = userEvent.setup()
    renderSettings()

    expect(screen.getByRole("heading", { name: "OCR defaults" })).toBeVisible()
    expect(screen.getByRole("combobox", { name: /default OCR language/i })).toBeVisible()
    await user.click(screen.getByRole("switch", { name: /enable OCR/i }))
    expect(window.localStorage.getItem("enableOcrAssets")).toBe("true")
  })

  it("finds a shared OCR language by its visible name and saves its language code", async () => {
    const user = userEvent.setup()
    renderSettings()

    await user.type(screen.getByRole("combobox", { name: /default OCR language/i }), "Dutch")
    await user.click(await screen.findByText("Dutch; Flemish", { selector: ".ant-select-item-option-content" }))
    expect(window.localStorage.getItem("defaultOCRLanguage")).toBe(JSON.stringify("nld"))
  })

  it("starts on Standard and changes the single named editor with the arrow keys", async () => {
    const user = userEvent.setup()
    renderSettings()

    const group = within(screen.getByRole("radiogroup", { name: "Preset to edit" }))
    expect(group.getAllByRole("radio")).toHaveLength(3)
    expect(group.getByRole("radio", { name: "Quick" })).not.toBeChecked()
    const standard = group.getByRole("radio", { name: "Standard" })
    expect(standard).toBeChecked()
    const standardEditor = within(screen.getByRole("region", { name: "Standard preset" }))
    expect(standardEditor.getByRole("heading", { name: "Standard preset" })).toBeVisible()
    await user.click(standard)
    await user.keyboard("{ArrowRight}")

    expect(group.getByRole("radio", { name: "Deep" })).toBeChecked()
    expect(screen.getAllByRole("region", { name: / preset$/i })).toHaveLength(1)
    const deepEditor = within(screen.getByRole("region", { name: "Deep preset" }))
    expect(deepEditor.getByRole("heading", { name: "Deep preset" })).toBeVisible()
  })

  it("maps each destination to the existing storage flags and restores it on re-open", async () => {
    const user = userEvent.setup()
    const view = renderSettings()
    const destination = screen.getByRole("combobox", { name: "Destination" })
    await waitFor(() => expect(destination).toBeEnabled())
    const choices = [
      ["draft", true, true],
      ["media", true, false],
      ["extract", false, false]
    ] as const

    for (const [value, storeRemote, reviewBeforeStorage] of choices) {
      await user.selectOptions(destination, value)
      expect(storedPresets().standard).toMatchObject({ storeRemote, reviewBeforeStorage })
    }

    view.unmount()
    renderSettings()
    expect(await screen.findByDisplayValue("Extract without saving")).toHaveAccessibleName("Destination")
  })

  it("persists both edits when the first preset storage write is delayed", async () => {
    const user = userEvent.setup()
    await act(async () => { renderSettings() })
    const pending = Promise.withResolvers<void>()
    const persist = Storage.prototype.set
    const write = vi.spyOn(Storage.prototype, "set").mockImplementationOnce(async function (key, value) {
      await pending.promise
      await persist.call(this, key, value)
    })

    try {
      await user.selectOptions(screen.getByRole("combobox", { name: "Destination" }), "extract")
      await waitFor(() => expect(write).toHaveBeenCalledWith(presetsKey, expect.anything()))
      await user.click(screen.getByRole("switch", { name: "Analysis" }))
    } finally {
      await act(async () => { pending.resolve() })
    }

    await waitFor(() => expect(storedPresets().standard).toMatchObject({
      storeRemote: false,
      reviewBeforeStorage: false,
      common: { perform_analysis: false }
    }))
  })

  it.each([undefined, "none"])("explains missing provider %s until a custom name is saved", async (api_name) => {
    window.localStorage.setItem(presetsKey, JSON.stringify({
      standard: { advancedValues: { api_name, custom_prompt: "Keep this prompt" } }
    }))
    const user = userEvent.setup()
    // Let the real storage hook finish reading the seeded legacy preset.
    await act(async () => { renderSettings() })

    const analysis = screen.getByRole("switch", { name: "Analysis" })
    expect(analysis).toBeChecked()
    const provider = screen.getByRole("combobox", { name: "Analysis provider" })
    const prerequisite = /choose an analysis provider/i
    expect(provider).toHaveAccessibleDescription(prerequisite)
    await user.clear(provider)
    await user.type(provider, "lab-provider")
    await user.tab()
    expect(provider).not.toHaveAccessibleDescription(prerequisite)
    expect(storedPresets().standard.advancedValues).toMatchObject({
      api_name: "lab-provider", custom_prompt: "Keep this prompt"
    })

    await user.click(analysis)
    expect(storedPresets().standard.common.perform_analysis).toBe(false)
    await user.click(analysis)
    expect(screen.getByRole("combobox", { name: "Analysis provider" })).toHaveValue("lab-provider")
  })

  it("keeps Analysis provider as the single provider control when advanced options open", async () => {
    const user = userEvent.setup()
    renderSettings()

    await user.click(screen.getByRole("button", { name: "Advanced options" }))
    expect(screen.getByRole("combobox", { name: "Analysis provider" })).toBeVisible()
    expect(screen.queryByText(/^api name$/i)).not.toBeInTheDocument()
  })

  it("undoes a named reset without losing a later change to another preset", async () => {
    const user = userEvent.setup()
    renderSettings()

    const destination = screen.getByRole("combobox", { name: "Destination" })
    await waitFor(() => expect(destination).toBeEnabled())
    await user.selectOptions(destination, "extract")
    await user.click(screen.getByRole("button", { name: "Reset Standard preset" }))
    expect(storedPresets().standard).toMatchObject({ storeRemote: true, reviewBeforeStorage: false })
    expect(screen.getByRole("status")).toHaveTextContent(/Standard/)
    await user.click(screen.getByRole("radio", { name: "Quick" }))
    await user.selectOptions(screen.getByRole("combobox", { name: "Destination" }), "draft")
    await user.click(screen.getByRole("button", { name: "Undo" }))

    expect(storedPresets()).toMatchObject({
      standard: { storeRemote: false, reviewBeforeStorage: false },
      quick: { storeRemote: true, reviewBeforeStorage: true }
    })
  })

  it("resets all three presets and restores their saved values without reverting shared OCR", async () => {
    const user = userEvent.setup()
    renderSettings()

    for (const name of ["Quick", "Standard", "Deep"]) {
      await user.click(screen.getByRole("radio", { name }))
      await user.selectOptions(screen.getByRole("combobox", { name: "Destination" }), "extract")
    }
    const beforeReset = storedPresets()
    await user.click(screen.getByRole("button", { name: "Reset all presets" }))
    const status = screen.getByRole("status")
    for (const name of ["Quick", "Standard", "Deep"]) {
      expect(status).toHaveTextContent(name)
    }
    expect(storedPresets()).toMatchObject({
      quick: { storeRemote: true, reviewBeforeStorage: false },
      standard: { storeRemote: true, reviewBeforeStorage: false },
      deep: { storeRemote: true, reviewBeforeStorage: true }
    })
    await user.click(screen.getByRole("switch", { name: /enable OCR/i }))
    await user.click(screen.getByRole("button", { name: "Undo" }))

    expect(storedPresets()).toEqual(beforeReset)
    expect(window.localStorage.getItem("enableOcrAssets")).toBe("true")
  })

  it("keeps a preset edited by another storage client after Reset all while undoing the other reset", async () => {
    const external = new Storage()
    await external.set(presetsKey, {
      quick: { advancedValues: { author: "Quick author" } },
      deep: { advancedValues: { author: "Deep author" } }
    })
    const user = userEvent.setup()
    await act(async () => { renderSettings() })
    await user.click(screen.getByRole("button", { name: "Reset all presets" }))
    const reset = storedPresets()
    expect(reset.quick.advancedValues).toEqual({})
    expect(reset.deep.advancedValues).toEqual({})
    await act(async () => {
      await external.set(presetsKey, {
        ...reset,
        deep: { ...reset.deep, advancedValues: { author: "New author" } }
      })
    })
    await user.click(screen.getByRole("button", { name: "Undo" }))

    expect(storedPresets().quick.advancedValues).toEqual({ author: "Quick author" })
    expect(storedPresets().deep.advancedValues).toEqual({ author: "New author" })
  })

  it.each(["Reset Standard preset", "Undo"])("reports a failed %s write and lets the user retry it", async (action) => {
    const user = userEvent.setup()
    await act(async () => { renderSettings() })
    const destination = screen.getByRole("combobox", { name: "Destination" })
    await user.selectOptions(destination, "extract")
    const undoingReset = action === "Undo"
    if (undoingReset) {
      await user.click(screen.getByRole("button", { name: "Reset Standard preset" }))
    }
    const beforeWrite = storedPresets()
    vi.spyOn(window.Storage.prototype, "setItem").mockImplementationOnce(() => {
      throw new DOMException("Storage quota reached", "QuotaExceededError")
    })

    await user.click(screen.getByRole("button", { name: action }))
    expect(await screen.findByRole("alert")).toHaveTextContent(undoingReset ? /restore|undo/i : /reset/i)
    expect(storedPresets()).toEqual(beforeWrite)
    expect(destination).toHaveValue(undoingReset ? "media" : "extract")
    const retry = screen.getByRole("button", { name: action })
    expect(retry).toBeEnabled()
    await user.click(retry)

    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    expect(storedPresets().standard.storeRemote).toBe(!undoingReset)
    expect(destination).toHaveValue(undoingReset ? "extract" : "media")
  })
})
