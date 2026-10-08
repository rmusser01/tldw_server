import { describe, expect, it } from "vitest"

import {
  NativeHistoryContextRefusal,
  NativeHistorySettingsReadError,
  nativeRegenerationFeedback
} from "../native-character-preflight"

const t = (_key: string, fallback: string) => fallback

describe("regeneration recovery guidance", () => {
  it.each([
    "native_history_persistence_unacknowledged",
    "invalid_history_admission",
    "invalid_history_settlement",
    "native_history_stream_interrupted",
    "unknown",
    "secret-token=do-not-display"
  ])("does not claim rollback or expose raw details for %s", (code) => {
    expect(nativeRegenerationFeedback(new Error(code), t)).toBe(
      "We could not confirm the new reply. Your original exchange and draft are kept. Reload this chat and review recovery before regenerating again."
    )
  })
  it.each(["message_changed", "stale_snapshot", "stale_selection"])(
    "explains a retired %s without claiming that a send was rolled back",
    (code) => {
      expect(nativeRegenerationFeedback(new Error(code), t)).toBe(
        "The saved message or conversation changed. Your original exchange and draft are kept. Reload the intended chat before regenerating again."
      )
    }
  )
  it("recognizes an authenticated backend version conflict without showing response content", () => {
    expect(
      nativeRegenerationFeedback(
        {
          status: 409,
          details: {
            detail: { code: "version_conflict", message: "sensitive response" }
          }
        },
        t
      )
    ).toContain("saved message or conversation changed")
  })
  it.each([
    "unsupported_history_regeneration",
    "native_history_explicit_context_unsupported",
    "native_history_client_tools_unsupported",
    "native_history_setting_unsupported:numCtx"
  ])("guides unsupported %s setup without discarding the original", (code) => {
    expect(nativeRegenerationFeedback(new Error(code), t)).toBe(
      "This reply or its active setup does not support regeneration yet. Your original exchange and draft are kept. Review conversation and response settings or start a compatible saved chat."
    )
  })
  it("certifies no new request only from the settings preflight boundary", () => {
    expect(
      nativeRegenerationFeedback(
        new NativeHistorySettingsReadError(new Error("private"), () => true),
        t
      )
    ).toBe(
      "Conversation settings could not be verified. No new reply was requested. Reload this saved chat or try again; your original exchange and draft are kept."
    )
  })
  it("recognizes an explicit server context refusal", () => {
    expect(
      nativeRegenerationFeedback(
        new NativeHistoryContextRefusal(new Error("private"), () => true),
        t
      )
    ).toBe(
      "The server refused regeneration because this saved chat uses unsupported context. Review the saved setup or start a compatible chat; your original exchange and draft are kept."
    )
  })
})
