import { describe, expect, it, vi } from "vitest"

import {
  getSettingsNavGroups,
  isSettingsAnnouncementBadgeActive
} from "../settings-nav"
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

describe("settings nav guardian gating", () => {
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
