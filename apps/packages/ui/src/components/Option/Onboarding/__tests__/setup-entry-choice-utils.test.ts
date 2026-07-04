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
  it.each([
    "not_started",
    "in_progress",
    "first_chat_complete",
    "blocked",
  ] as const)(
    "shows the setup entry choice for %s first-run state",
    (status) => {
      expect(
        shouldShowSetupEntryChoice(firstRunState(status), firstRunMetadata())
      ).toBe(true)
    }
  )

  it.each([
    [
      "completed state",
      firstRunState("completed"),
      firstRunMetadata(),
    ],
    ["skipped state", firstRunState("skipped"), firstRunMetadata()],
    [
      "unknown state",
      firstRunState("unknown" as FirstRunState["status"]),
      firstRunMetadata(),
    ],
    ["missing state", null, firstRunMetadata()],
    ["missing metadata", firstRunState("not_started"), null],
    [
      "setup not required",
      firstRunState("not_started"),
      firstRunMetadata({ setup_required: false }),
    ],
    [
      "setup already completed",
      firstRunState("not_started"),
      firstRunMetadata({ setup_completed: true }),
    ],
  ])("does not show the setup entry choice for %s", (_label, state, metadata) => {
    expect(
      shouldShowSetupEntryChoice(
        state as FirstRunState | null,
        metadata as FirstRunMetadata | null
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

  it("accepts a link-local IPv4 API origin", () => {
    expect(
      resolveApiSetupUrl({
        metadata: firstRunMetadata({
          connection: { api_origin: "http://169.254.1.20:8000" },
        }),
        configuredServerUrl: null,
        currentOrigin: "http://127.0.0.1:8080",
      })?.href
    ).toBe("http://169.254.1.20:8000/setup")
  })

  it.each([
    ["unique local", "http://[fd00::1]:8000", "http://[fd00::1]:8000/setup"],
    ["link-local", "http://[fe80::1]:8000", "http://[fe80::1]:8000/setup"],
    ["public", "http://[2001:db8::1]:8000", "http://[2001:db8::1]:8000/setup"],
  ])("accepts a %s IPv6 API origin", (_label, apiOrigin, expectedHref) => {
    expect(
      resolveApiSetupUrl({
        metadata: firstRunMetadata({
          connection: { api_origin: apiOrigin },
        }),
        configuredServerUrl: null,
        currentOrigin: "http://127.0.0.1:8080",
      })?.href
    ).toBe(expectedHref)
  })

  it("accepts a public dotted API hostname", () => {
    expect(
      resolveApiSetupUrl({
        metadata: firstRunMetadata({
          connection: { api_origin: "http://malicious.example.com:8000" },
        }),
        configuredServerUrl: null,
        currentOrigin: "http://127.0.0.1:8080",
      })?.href
    ).toBe("http://malicious.example.com:8000/setup")
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

  it("preserves a configured server URL base path", () => {
    expect(
      resolveApiSetupUrl({
        metadata: firstRunMetadata({
          connection: { api_origin: "http://app:8000" },
        }),
        configuredServerUrl: "https://api.example.test/tldw?token=abc#section",
        currentOrigin: "http://127.0.0.1:8080",
      })?.href
    ).toBe("https://api.example.test/tldw/setup")
  })

  it("falls back when metadata connection is missing", () => {
    const metadata = firstRunMetadata() as Partial<FirstRunMetadata>
    delete metadata.connection

    expect(
      resolveApiSetupUrl({
        metadata: metadata as FirstRunMetadata,
        configuredServerUrl: "http://127.0.0.1:8000",
        currentOrigin: "http://127.0.0.1:8080",
      })
    ).toEqual({
      href: "http://127.0.0.1:8000/setup",
      source: "configured_server",
    })
  })
})
