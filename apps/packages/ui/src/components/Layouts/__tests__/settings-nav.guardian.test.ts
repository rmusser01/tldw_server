import { readdir, readFile } from "node:fs/promises"
import path from "node:path"
import { describe, expect, it, vi } from "vitest"

import {
  getSettingsNavGroups,
  isSettingsAnnouncementBadgeActive
} from "../settings-nav"
import enOption from "@/assets/locale/en/option.json"
import enSettings from "@/assets/locale/en/settings.json"
import type { ServerCapabilities } from "@/services/tldw/server-capabilities"
import {
  FAMILY_WIZARD_SETTINGS_PATH,
  GUARDIAN_SETTINGS_PATH
} from "@/routes/route-capabilities"

vi.mock("@/routes/route-registry", () => {
  const MockIcon = () => null
  return {
    optionRoutes: [
      {
        kind: "options",
        path: "/settings/chat",
        nav: {
          group: "server",
          labelToken: "settings:chatSettingsNav",
          icon: MockIcon,
          order: 1
        }
      },
      {
        kind: "options",
        path: "/settings/family-guardrails",
        nav: {
          group: "server",
          labelToken: "settings:familyGuardrailsWizardNav",
          icon: MockIcon,
          order: 2
        }
      },
      {
        kind: "options",
        path: "/settings/guardian",
        nav: {
          group: "server",
          labelToken: "settings:guardianNav",
          icon: MockIcon,
          order: 3
        }
      },
      {
        kind: "options",
        path: "/settings/evaluations",
        nav: {
          group: "server",
          labelToken: "settings:evaluationsSettings.title",
          icon: MockIcon,
          beta: true,
          order: 4
        }
      },
      {
        kind: "options",
        path: "/research-workspace",
        nav: {
          group: "workspace",
          labelToken: "settings:researchWorkspaceNav",
          icon: MockIcon,
          beta: true,
          order: 0
        }
      }
    ]
  }
})

const makeCapabilities = (
  overrides: Partial<ServerCapabilities> = {}
): ServerCapabilities =>
  ({
    hasGuardian: false,
    hasSelfMonitoring: false,
    ...overrides
  } as ServerCapabilities)

const flattenPaths = (caps?: ServerCapabilities | null): string[] =>
  getSettingsNavGroups(caps).flatMap((group) => group.items.map((item) => item.to))

const localeNamespaces: Record<string, unknown> = {
  option: enOption,
  settings: enSettings
}

const getPathValue = (source: unknown, keyPath: string): unknown =>
  keyPath.split(".").reduce<unknown>((current, segment) => {
    if (!current || typeof current !== "object") return undefined
    return (current as Record<string, unknown>)[segment]
  }, source)

const resolveLocaleToken = (token: string): unknown => {
  const [namespace, keyPath] = token.split(":")
  if (!namespace || !keyPath) return undefined

  return getPathValue(localeNamespaces[namespace], keyPath)
}

describe("settings nav guardian gating", () => {
  it("uses label tokens that resolve to user-facing English locale copy", () => {
    const missingTokens = getSettingsNavGroups(undefined)
      .flatMap((group) => group.items.map((item) => item.labelToken))
      .filter((token) => {
        const value = resolveLocaleToken(token)
        return typeof value !== "string" || value.trim().length === 0
      })

    expect(missingTokens).toEqual([])
  })

  it("keeps settings navigation locale keys present across locale directories", async () => {
    const localeRoot = path.resolve(
      __dirname,
      "../../../assets/locale"
    )
    const requiredSettingsKeys = [
      "navigation.setupRecovery",
      "navigation.preferencesWorkflow",
      "navigation.dataAdmin",
      "providerKeys.navTitle",
      "dataManagement.navTitle",
      "quickIngestSettings.ocrDefaults.title",
      "quickIngestSettings.ocrDefaults.description",
      "sidepanelSettings.shortcuts.title",
      "sidepanelSettings.shortcuts.description",
      "sidepanelSettings.shortcuts.setupRecovery.label",
      "sidepanelSettings.shortcuts.setupRecovery.description",
      "sidepanelSettings.shortcuts.preferences.label",
      "sidepanelSettings.shortcuts.preferences.description",
      "sidepanelSettings.shortcuts.ui.label",
      "sidepanelSettings.shortcuts.ui.description",
      "sidepanelSettings.shortcuts.dataAdmin.label",
      "sidepanelSettings.shortcuts.dataAdmin.description",
      "sidepanelSettings.shortcuts.serverAuth.label",
      "sidepanelSettings.shortcuts.serverAuth.description",
      "sidepanelSettings.shortcuts.health.label",
      "sidepanelSettings.shortcuts.health.description",
      "sidepanelSettings.local.title"
    ]

    const locales = (await readdir(localeRoot, { withFileTypes: true }))
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name)

    for (const locale of locales) {
      const settingsPath = path.join(localeRoot, locale, "settings.json")
      const parsed = JSON.parse(await readFile(settingsPath, "utf8")) as unknown
      for (const keyPath of requiredSettingsKeys) {
        const value = getPathValue(parsed, keyPath)
        expect(
          typeof value,
          `Missing or non-string locale key: ${locale}.${keyPath}`
        ).toBe("string")
        expect(String(value).trim().length).toBeGreaterThan(0)
      }
    }
  })

  it("groups settings routes by user task", () => {
    const groups = getSettingsNavGroups(undefined)
    const pathsByGroup = Object.fromEntries(
      groups.map((group) => [group.key, group.items.map((item) => item.to)])
    )

    expect(groups.map((group) => group.key)).toEqual([
      "setupRecovery",
      "preferencesWorkflow",
      "dataAdmin"
    ])
    expect(pathsByGroup.setupRecovery).toEqual(
      expect.arrayContaining([
        "/settings",
        "/settings/tldw",
        "/settings/provider-keys",
        "/settings/model",
        "/settings/health"
      ])
    )
    expect(pathsByGroup.preferencesWorkflow).toEqual(
      expect.arrayContaining([
        "/settings/preferences",
        "/settings/chat",
        "/settings/ui",
        "/settings/prompt",
        "/settings/rag"
      ])
    )
    expect(pathsByGroup.dataAdmin).toEqual(
      expect.arrayContaining(["/settings/data", "/settings/about"])
    )
  })

  it("keeps Workflow prompts in Preferences & Workflow with its own label token", () => {
    const groups = getSettingsNavGroups(undefined)
    const preferences = groups.find((group) => group.key === "preferencesWorkflow")
    const prompt = preferences?.items.find((item) => item.to === "/settings/prompt")

    expect(prompt?.labelToken).toBe("settings:servicePrompts.title")
    expect(resolveLocaleToken(prompt?.labelToken ?? "")).toBe("Workflow prompts")
  })

  it("keeps only settings-prefixed routes in settings navigation", () => {
    const paths = flattenPaths(undefined)
    expect(paths).toContain("/settings/chat")
    expect(paths).not.toContain("/research-workspace")
  })

  it("includes guardian route by default when capabilities are not resolved", () => {
    const paths = flattenPaths(undefined)
    expect(paths).toContain(GUARDIAN_SETTINGS_PATH)
  })

  it("includes family wizard route by default when capabilities are not resolved", () => {
    const paths = flattenPaths(undefined)
    expect(paths).toContain(FAMILY_WIZARD_SETTINGS_PATH)
  })

  it("hides family wizard route when guardian capability is unavailable", () => {
    const paths = flattenPaths(
      makeCapabilities({
        hasGuardian: false,
        hasSelfMonitoring: true
      })
    )
    expect(paths).not.toContain(FAMILY_WIZARD_SETTINGS_PATH)
  })

  it("keeps family wizard route when guardian exists without self-monitoring", () => {
    const paths = flattenPaths(
      makeCapabilities({
        hasGuardian: true,
        hasSelfMonitoring: false
      })
    )
    expect(paths).toContain(FAMILY_WIZARD_SETTINGS_PATH)
  })

  it("hides guardian route when capabilities resolve to unavailable", () => {
    const paths = flattenPaths(null)
    expect(paths).not.toContain(GUARDIAN_SETTINGS_PATH)
  })

  it("hides guardian route when guardian/self-monitoring endpoints are unavailable", () => {
    const paths = flattenPaths(
      makeCapabilities({
        hasGuardian: false,
        hasSelfMonitoring: false
      })
    )
    expect(paths).not.toContain(GUARDIAN_SETTINGS_PATH)
  })

  it("keeps guardian route when both guardian capabilities are present", () => {
    const paths = flattenPaths(
      makeCapabilities({
        hasGuardian: true,
        hasSelfMonitoring: true
      })
    )
    expect(paths).toContain(GUARDIAN_SETTINGS_PATH)
  })

  it("limits beta badge visibility to active settings announcements", () => {
    const groups = getSettingsNavGroups(undefined)
    const byPath = Object.fromEntries(
      groups.flatMap((group) => group.items.map((item) => [item.to, item]))
    )

    expect(byPath["/settings/guardian"]?.beta).toBeUndefined()
    expect(byPath["/settings/evaluations"]?.beta).toBeUndefined()
  })
})

describe("settings announcement windows", () => {
  it("treats announcements as active before their window expires", () => {
    expect(
      isSettingsAnnouncementBadgeActive(
        "/settings/prompt-studio",
        new Date("2026-06-01T00:00:00Z")
      )
    ).toBe(true)
  })

  it("expires announcements after their window closes", () => {
    expect(
      isSettingsAnnouncementBadgeActive(
        "/settings/prompt-studio",
        new Date("2027-01-01T00:00:00Z")
      )
    ).toBe(false)
  })

  it("guardian and family-guardrails no longer have announcement windows", () => {
    expect(
      isSettingsAnnouncementBadgeActive(
        "/settings/guardian",
        new Date("2026-06-01T00:00:00Z")
      )
    ).toBe(false)
    expect(
      isSettingsAnnouncementBadgeActive(
        "/settings/family-guardrails",
        new Date("2026-06-01T00:00:00Z")
      )
    ).toBe(false)
  })
})
