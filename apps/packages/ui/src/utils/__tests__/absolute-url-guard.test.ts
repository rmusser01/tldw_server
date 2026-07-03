import { describe, expect, it } from "vitest"
import {
  ABSOLUTE_URL_BLOCK_ERROR,
  absoluteOriginAllowlistFromConfig,
  evaluateAbsoluteUrlAccess,
  isAbsoluteHttpUrl,
  isAbsoluteUrlAllowlisted,
  isSameOriginAbsoluteUrlForConfiguredServer
} from "@/utils/absolute-url-guard"

const serverCfg = {
  serverUrl: "https://server.example.test",
  authMode: "single-user",
  apiKey: "secret-key"
}

describe("absolute-url-guard", () => {
  it("treats only http(s) paths as absolute", () => {
    expect(isAbsoluteHttpUrl("https://a.example/x")).toBe(true)
    expect(isAbsoluteHttpUrl("http://a.example/x")).toBe(true)
    expect(isAbsoluteHttpUrl("/api/v1/media")).toBe(false)
    expect(isAbsoluteHttpUrl("ftp://a.example")).toBe(false)
    expect(isAbsoluteHttpUrl(undefined)).toBe(false)
  })

  it("allowlist always contains the configured server origin", () => {
    const allow = absoluteOriginAllowlistFromConfig(serverCfg)
    expect(allow.has("https://server.example.test")).toBe(true)
  })

  it("merges explicit absoluteUrlAllowlist entries (string or array)", () => {
    const arrayCfg = {
      ...serverCfg,
      absoluteUrlAllowlist: ["https://cdn.example.test", "not-a-url"]
    }
    expect(isAbsoluteUrlAllowlisted("https://cdn.example.test/f", arrayCfg)).toBe(
      true
    )

    const stringCfg = {
      ...serverCfg,
      absoluteUrlAllowlist: "https://cdn.example.test, https://two.example.test"
    }
    expect(isAbsoluteUrlAllowlisted("https://two.example.test/x", stringCfg)).toBe(
      true
    )
  })

  it("same-origin check matches the configured server origin only", () => {
    expect(
      isSameOriginAbsoluteUrlForConfiguredServer(
        "https://server.example.test/api/v1/media/ingest/jobs",
        serverCfg
      )
    ).toBe(true)
    expect(
      isSameOriginAbsoluteUrlForConfiguredServer(
        "https://attacker.example/x",
        serverCfg
      )
    ).toBe(false)
  })

  describe("evaluateAbsoluteUrlAccess", () => {
    it("blocks a non-allowlisted cross-origin absolute URL (attacker)", () => {
      const decision = evaluateAbsoluteUrlAccess(
        "https://attacker.example/steal",
        serverCfg
      )
      expect(decision).toEqual({
        isAbsolute: true,
        blocked: true,
        skipAuth: true
      })
    })

    it("attaches auth for a same-origin absolute URL to the configured server", () => {
      const decision = evaluateAbsoluteUrlAccess(
        "https://server.example.test/api/v1/media/ingest/jobs",
        serverCfg
      )
      expect(decision).toEqual({
        isAbsolute: true,
        blocked: false,
        skipAuth: false
      })
    })

    it("permits but strips auth for an allowlisted cross-origin absolute URL", () => {
      const cfg = {
        ...serverCfg,
        absoluteUrlAllowlist: ["https://cdn.example.test"]
      }
      const decision = evaluateAbsoluteUrlAccess(
        "https://cdn.example.test/file",
        cfg
      )
      expect(decision).toEqual({
        isAbsolute: true,
        blocked: false,
        skipAuth: true
      })
    })

    it("leaves relative paths untouched (auth attached, not blocked)", () => {
      const decision = evaluateAbsoluteUrlAccess(
        "/api/v1/media/ingest/jobs",
        serverCfg
      )
      expect(decision).toEqual({
        isAbsolute: false,
        blocked: false,
        skipAuth: false
      })
    })
  })

  it("exposes a stable block error message", () => {
    expect(ABSOLUTE_URL_BLOCK_ERROR).toContain("allowlisted")
  })
})
