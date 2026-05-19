import { describe, expect, it } from "vitest"

import {
  CLIPPER_PATH,
  COMPANION_CONVERSATION_PATH,
  COMPANION_PATH,
  FAMILY_WIZARD_SETTINGS_PATH,
  GUARDIAN_SETTINGS_PATH,
  PERSONA_DOCK_PATH,
  isClipperAvailable,
  isCompanionAvailable,
  isGuardianSettingsAvailable,
  isPersonaDockAvailable,
  isRouteEnabledForCapabilities
} from "../route-capabilities"
import type { ServerCapabilities } from "@/services/tldw/server-capabilities"

const makeCapabilities = (
  overrides: Partial<ServerCapabilities> = {}
): ServerCapabilities =>
  ({
    hasGuardian: false,
    hasWebClipper: false,
    hasPersonalization: false,
    hasSelfMonitoring: false,
    hasPersona: false,
    ...overrides
  } as ServerCapabilities)

describe("route capability gating", () => {
  it("exposes the canonical family wizard route path", () => {
    expect(FAMILY_WIZARD_SETTINGS_PATH).toBe("/settings/family-guardrails")
  })

  it("enables family wizard route when guardian capability exists without self-monitoring", () => {
    const caps = makeCapabilities({
      hasGuardian: true,
      hasSelfMonitoring: false
    })
    expect(isRouteEnabledForCapabilities(FAMILY_WIZARD_SETTINGS_PATH, caps)).toBe(true)
  })

  it("hides family wizard route when guardian capability is missing", () => {
    const caps = makeCapabilities({
      hasGuardian: false,
      hasSelfMonitoring: true
    })
    expect(isRouteEnabledForCapabilities(FAMILY_WIZARD_SETTINGS_PATH, caps)).toBe(false)
  })

  it("hides guardian route when capabilities are missing", () => {
    expect(isRouteEnabledForCapabilities(GUARDIAN_SETTINGS_PATH, null)).toBe(false)
  })

  it("hides guardian route when guardian capability is missing", () => {
    const caps = makeCapabilities({
      hasGuardian: false,
      hasSelfMonitoring: true
    })
    expect(isRouteEnabledForCapabilities(GUARDIAN_SETTINGS_PATH, caps)).toBe(false)
  })

  it("hides guardian route when self-monitoring capability is missing", () => {
    const caps = makeCapabilities({
      hasGuardian: true,
      hasSelfMonitoring: false
    })
    expect(isRouteEnabledForCapabilities(GUARDIAN_SETTINGS_PATH, caps)).toBe(false)
  })

  it("enables guardian route only when both capabilities are present", () => {
    const caps = makeCapabilities({
      hasGuardian: true,
      hasSelfMonitoring: true
    })
    expect(isGuardianSettingsAvailable(caps)).toBe(true)
    expect(isRouteEnabledForCapabilities(GUARDIAN_SETTINGS_PATH, caps)).toBe(true)
  })

  it("does not gate unrelated routes", () => {
    const caps = makeCapabilities({
      hasGuardian: false,
      hasSelfMonitoring: false
    })
    expect(isRouteEnabledForCapabilities("/settings/chat", caps)).toBe(true)
  })

  it("hides persona dock route when persona capability is missing", () => {
    const caps = makeCapabilities({
      hasPersona: false
    })
    expect(isRouteEnabledForCapabilities(PERSONA_DOCK_PATH, caps)).toBe(false)
  })

  it("enables persona dock route when persona capability is present", () => {
    const caps = makeCapabilities({
      hasPersona: true
    })
    expect(isPersonaDockAvailable(caps)).toBe(true)
    expect(isRouteEnabledForCapabilities(PERSONA_DOCK_PATH, caps)).toBe(true)
  })

  it.each([false, true])(
    "keeps the companion workspace available when hasPersonalization=%s",
    (hasPersonalization) => {
      const caps = makeCapabilities({
        hasPersonalization
      })
      expect(isCompanionAvailable(caps)).toBe(true)
      expect(isRouteEnabledForCapabilities(COMPANION_PATH, caps)).toBe(true)
    }
  )

  it("hides the companion conversation when either persona or personalization is missing", () => {
    expect(
      isRouteEnabledForCapabilities(
        COMPANION_CONVERSATION_PATH,
        makeCapabilities({
          hasPersona: true,
          hasPersonalization: false
        })
      )
    ).toBe(false)
    expect(
      isRouteEnabledForCapabilities(
        COMPANION_CONVERSATION_PATH,
        makeCapabilities({
          hasPersona: false,
          hasPersonalization: true
        })
      )
    ).toBe(false)
  })

  it("enables the companion conversation only when both persona and personalization are present", () => {
    const caps = makeCapabilities({
      hasPersona: true,
      hasPersonalization: true
    })
    expect(isRouteEnabledForCapabilities(COMPANION_CONVERSATION_PATH, caps)).toBe(true)
  })

  it("hides the clipper route when web clipper capability is missing", () => {
    const caps = makeCapabilities({
      hasWebClipper: false
    })
    expect(isClipperAvailable(caps)).toBe(false)
    expect(isRouteEnabledForCapabilities(CLIPPER_PATH, caps)).toBe(false)
  })

  it("enables the clipper route when web clipper capability is present", () => {
    const caps = makeCapabilities({
      hasWebClipper: true
    })
    expect(isClipperAvailable(caps)).toBe(true)
    expect(isRouteEnabledForCapabilities(CLIPPER_PATH, caps)).toBe(true)
  })
})
