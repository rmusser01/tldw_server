import { createInstance } from "i18next"
import { describe, expect, it, vi } from "vitest"

import commonEn from "../../assets/locale/en/common.json"

import ICUWithInterpolation from "../icu-format"

const createI18n = async () => {
  const instance = createInstance()
  await instance.use(ICUWithInterpolation).init({
    lng: "en",
    fallbackLng: false,
    resources: {
      en: {
        translation: {
          action: "View {{name}}",
          count: "{{count}} skills",
          shortcut: "Press {{shortcut}}",
          steps: "{count, plural, one {# step} other {# steps}}",
          literal: "Use {{value}}, {{...}}, or {{ item.title }} in a template."
        }
      }
    },
    interpolation: {
      escapeValue: false,
      defaultVariables: { shortcut: "Cmd+K" }
    }
  })
  return instance
}

describe("ICUWithInterpolation", () => {
  it("preserves object-valued English resources without stringifying them", async () => {
    const i18n = createInstance()
    await i18n.use(ICUWithInterpolation).init({
      lng: "en",
      fallbackLng: false,
      resources: { en: { common: commonEn } }
    })

    expect(i18n.t("common:loading", { returnObjects: true })).toEqual({
      title: "Loading…",
      description: "Please wait while we get things ready.",
      content: "Loading content…"
    })
  })

  it("delegates syntax-tree arrays to ICU for formatting and repeated values", async () => {
    const i18n = createInstance()
    await i18n.use(ICUWithInterpolation).init({
      lng: "en",
      fallbackLng: false,
      resources: {
        en: {
          translation: {
            compiled: [
              { type: 0, value: "Hello " },
              { type: 1, value: "name" }
            ]
          }
        }
      }
    })

    expect(i18n.t("compiled", { name: "Alice" })).toBe("Hello Alice")
    expect(i18n.t("compiled", { name: "Bob" })).toBe("Hello Bob")
  })

  it("delegates object resources to the configured upstream parse-error handler", async () => {
    const resource = { label: "Object message" }
    const parseErrorHandler = vi.fn(
      (..._args: [Error, string, unknown, object]) => "Translation unavailable"
    )
    const i18n = createInstance()
    await i18n.use(ICUWithInterpolation).init({
      lng: "en",
      fallbackLng: false,
      resources: { en: { translation: { object: resource } } },
      i18nFormat: { parseErrorHandler }
    })

    expect(i18n.t("object")).toBe("Translation unavailable")
    expect(parseErrorHandler).toHaveBeenCalledWith(
      expect.any(Error),
      "object",
      resource,
      expect.any(Object)
    )
    expect(parseErrorHandler.mock.calls[0][2]).toBe(resource)
  })

  it("does not cache the first interpolation values for repeated keys", async () => {
    const i18n = await createI18n()

    expect(i18n.t("action", { name: "first-skill" })).toBe("View first-skill")
    expect(i18n.t("action", { name: "second-skill" })).toBe("View second-skill")
    expect(i18n.t("count", { count: 0 })).toBe("0 skills")
    expect(i18n.t("count", { count: 30 })).toBe("30 skills")
    expect(i18n.t("shortcut")).toBe("Press Cmd+K")
  })

  it("keeps ICU plural formatting and literal template braces intact", async () => {
    const i18n = await createI18n()

    expect(i18n.t("steps", { count: 1 })).toBe("1 step")
    expect(i18n.t("steps", { count: 2 })).toBe("2 steps")
    expect(i18n.t("literal")).toBe(
      "Use {{value}}, {{...}}, or {{ item.title }} in a template."
    )
  })
})
