import { describe, expect, it } from "vitest"
import enOption from "@/assets/locale/en/option.json"
import publicEnOption from "@/public/_locales/en/option.json"

const flattenStrings = (
  value: unknown,
  prefix: string[] = []
): Record<string, string> => {
  if (typeof value === "string") {
    return { [prefix.join("_")]: value }
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {}
  }

  return Object.entries(value as Record<string, unknown>).reduce(
    (result, [key, nestedValue]) => ({
      ...result,
      ...flattenStrings(nestedValue, [...prefix, key])
    }),
    {} as Record<string, string>
  )
}

describe("Skills locale keys", () => {
  it("keeps the complete English Skills namespace mirrored for the extension", () => {
    const webuiSkills = flattenStrings(enOption.skills, ["skills"])
    const extensionSkills = Object.fromEntries(
      Object.entries(publicEnOption)
        .filter(([key]) => key.startsWith("skills_"))
        .map(([key, entry]) => [
          key,
          String((entry as { message?: unknown }).message ?? "")
        ])
    )

    expect(Object.keys(extensionSkills).sort()).toEqual(Object.keys(webuiSkills).sort())
    for (const [key, value] of Object.entries(webuiSkills)) {
      expect(value.trim(), `Empty WebUI locale key: ${key}`).not.toBe("")
      expect(extensionSkills[key], `Mismatched extension locale key: ${key}`).toBe(value)
    }
  })

  it("keeps page, field, and visibility copy semantically distinct", () => {
    expect(enOption.skills.description).toBe(
      "Discover, test, create, import, and manage reusable instructions."
    )
    expect(enOption.skills.descriptionLabel).toBe("Description")
    expect(enOption.skills.visibleState).toBe("Visible")
    expect(enOption.skills.visibleInChatState).toBe("Visible in chat")
    expect(enOption.skills.hiddenState).toBe("Hidden")
    expect(enOption.skills.hiddenFromChatState).toBe("Hidden from chat")
  })
})
