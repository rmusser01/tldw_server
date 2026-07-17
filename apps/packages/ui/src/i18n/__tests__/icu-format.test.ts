import { createInstance } from "i18next"
import { describe, expect, it } from "vitest"

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
