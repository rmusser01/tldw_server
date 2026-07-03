import { describe, expect, it } from "vitest"
import type { FirstRunMetadata, FirstRunState } from "@/types/setup-onboarding"
import {
  isBlockedSetupState,
  isMutableWebUiSetupState,
  resolveApiSetupUrl,
  shouldShowSetupEntryChoice,
} from "../setup-entry-choice-utils"

const firstRunState = (
  status: FirstRunState["status"]
): FirstRunState => ({
  status,
  completed_steps: [],
  skipped_steps: [],
  step_data: {},
  first_chat: { completed: false },
  acknowledged_steps: [],
})

const firstRunMetadata = (
  overrides: Partial<FirstRunMetadata> = {}
): FirstRunMetadata => {
  const { connection, ...rest } = overrides

  return {
    auth_mode: "single_user",
    bundled_single_user_auth_available: true,
    manual_auth_required: false,
    setup_required: true,
    setup_completed: false,
    remote_setup_enabled: false,
    connection: {
      frontend_origin: null,
      api_origin: null,
      browser_access: "local",
      ...connection,
    },
    setup_paths: [],
    multi_user_exit: {
      guide_path: "/docs/multi-user",
    },
    ...rest,
  }
}

describe("setup entry choice trigger helpers", () => {
  it("shows the setup entry choice for incomplete first-run states", () => {
    for (const status of [
      "not_started",
      "in_progress",
      "first_chat_complete",
      "blocked",
    ] as const) {
      expect(shouldShowSetupEntryChoice(firstRunState(status), firstRunMetadata()))
        .toBe(true)
    }
  })

  it("does not show the setup entry choice for completed or unavailable setup state", () => {
    expect(
      shouldShowSetupEntryChoice(firstRunState("completed"), firstRunMetadata())
    ).toBe(false)
    expect(
      shouldShowSetupEntryChoice(firstRunState("skipped"), firstRunMetadata())
    ).toBe(false)
    expect(
      shouldShowSetupEntryChoice(
        firstRunState("unknown" as FirstRunState["status"]),
        firstRunMetadata()
      )
    ).toBe(false)
    expect(shouldShowSetupEntryChoice(null, firstRunMetadata())).toBe(false)
    expect(
      shouldShowSetupEntryChoice(firstRunState("not_started"), null)
    ).toBe(false)
    expect(
      shouldShowSetupEntryChoice(
        firstRunState("not_started"),
        firstRunMetadata({ setup_required: false })
      )
    ).toBe(false)
    expect(
      shouldShowSetupEntryChoice(
        firstRunState("not_started"),
        firstRunMetadata({ setup_completed: true })
      )
    ).toBe(false)
  })

  it("keeps blocked separate from mutable WebUI setup states", () => {
    expect(isBlockedSetupState(firstRunState("blocked"))).toBe(true)
    expect(isBlockedSetupState(firstRunState("in_progress"))).toBe(false)
    expect(isBlockedSetupState(null)).toBe(false)

    expect(isMutableWebUiSetupState(firstRunState("not_started"))).toBe(true)
    expect(isMutableWebUiSetupState(firstRunState("in_progress"))).toBe(true)
    expect(isMutableWebUiSetupState(firstRunState("first_chat_complete"))).toBe(
      true
    )
    expect(isMutableWebUiSetupState(firstRunState("blocked"))).toBe(false)
    expect(isMutableWebUiSetupState(null)).toBe(false)
  })
})

describe("resolveApiSetupUrl", () => {
  it("uses a loopback metadata API origin with the /setup path", () => {
    expect(
      resolveApiSetupUrl({
        metadata: firstRunMetadata({
          connection: { api_origin: "http://127.0.0.1:8000" },
        }),
        configuredServerUrl: null,
        currentOrigin: "http://127.0.0.1:8080",
      })
    ).toEqual({
      href: "http://127.0.0.1:8000/setup",
      source: "metadata",
    })
  })

  it("rejects single-label non-loopback metadata hosts", () => {
    expect(
      resolveApiSetupUrl({
        metadata: firstRunMetadata({
          connection: { api_origin: "http://app:8000" },
        }),
        configuredServerUrl: null,
        currentOrigin: "http://127.0.0.1:8080",
      })
    ).toBeNull()
  })

  it("accepts a same single-label hostname with a different port", () => {
    expect(
      resolveApiSetupUrl({
        metadata: firstRunMetadata({
          connection: { api_origin: "http://server:8000" },
        }),
        configuredServerUrl: null,
        currentOrigin: "http://server:8080",
      })?.href
    ).toBe("http://server:8000/setup")
  })

  it("rejects a candidate with the same origin as the current WebUI origin", () => {
    expect(
      resolveApiSetupUrl({
        metadata: firstRunMetadata({
          connection: { api_origin: "http://127.0.0.1:8080" },
        }),
        configuredServerUrl: null,
        currentOrigin: "http://127.0.0.1:8080",
      })
    ).toBeNull()
  })

  it("rejects a candidate matching metadata frontend origin", () => {
    expect(
      resolveApiSetupUrl({
        metadata: firstRunMetadata({
          connection: {
            api_origin: "http://web.example.test:8000",
            frontend_origin: "http://web.example.test:8000",
          },
        }),
        configuredServerUrl: null,
        currentOrigin: "http://127.0.0.1:8080",
      })
    ).toBeNull()
  })

  it("accepts a private LAN API origin", () => {
    expect(
      resolveApiSetupUrl({
        metadata: firstRunMetadata({
          connection: { api_origin: "http://192.168.1.20:8000" },
        }),
        configuredServerUrl: null,
        currentOrigin: "http://192.168.1.20:8080",
      })?.href
    ).toBe("http://192.168.1.20:8000/setup")
  })

  it("falls back to configured server URL when metadata origin is not browser-openable", () => {
    expect(
      resolveApiSetupUrl({
        metadata: firstRunMetadata({
          connection: { api_origin: "http://app:8000" },
        }),
        configuredServerUrl: "http://127.0.0.1:8000",
        currentOrigin: "http://127.0.0.1:8080",
      })
    ).toEqual({
      href: "http://127.0.0.1:8000/setup",
      source: "configured_server",
    })
  })

  it("rejects invalid URLs", () => {
    expect(
      resolveApiSetupUrl({
        metadata: firstRunMetadata({
          connection: { api_origin: "not a url" },
        }),
        configuredServerUrl: null,
        currentOrigin: "http://127.0.0.1:8080",
      })
    ).toBeNull()
  })

  it("preserves an existing /setup path without appending /setup/setup", () => {
    expect(
      resolveApiSetupUrl({
        metadata: firstRunMetadata({
          connection: {
            api_origin: "http://127.0.0.1:8000/setup?token=abc#section",
          },
        }),
        configuredServerUrl: null,
        currentOrigin: "http://127.0.0.1:8080",
      })?.href
    ).toBe("http://127.0.0.1:8000/setup")
  })
})
