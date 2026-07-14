import { describe, expect, it } from "vitest"

import { parseNotificationRuntimeConfig } from "../notification-runtime-scope"

const baseConfig = {
  serverUrl: "https://api.example.test",
  authMode: "multi-user",
  accessToken: "token-a"
}

describe("parseNotificationRuntimeConfig", () => {
  it.each([
    ["serverUrl", true],
    ["authMode", { toString: () => "multi-user" }],
    ["accessToken", 123]
  ])("rejects non-string multi-user %s values", (key, value) => {
    expect(parseNotificationRuntimeConfig({ ...baseConfig, [key]: value })).toBeNull()
  })

  it("rejects a non-string selected single-user API key", () => {
    expect(
      parseNotificationRuntimeConfig({
        serverUrl: "https://api.example.test",
        authMode: "single-user",
        apiKey: 123
      })
    ).toBeNull()
  })

  it.each([
    [
      "multi-user",
      { accessToken: "token-a", apiKey: 123 }
    ],
    [
      "single-user",
      { accessToken: 123, apiKey: "key-a" }
    ],
    [
      "multi-user",
      { accessToken: "token-a", apiKey: null }
    ],
    [
      "single-user",
      { accessToken: undefined, apiKey: "key-a" }
    ]
  ])("rejects a non-string unselected %s credential", (authMode, credentials) => {
    expect(
      parseNotificationRuntimeConfig({
        serverUrl: "https://api.example.test",
        authMode,
        ...credentials
      })
    ).toBeNull()
  })

  it.each([
    ["multi-user", { accessToken: "token-a" }],
    ["single-user", { apiKey: "key-a" }]
  ])("allows the unselected %s credential to be absent", (authMode, credential) => {
    expect(
      parseNotificationRuntimeConfig({
        serverUrl: "https://api.example.test",
        authMode,
        ...credential
      })
    ).toMatchObject({ authMode })
  })

  it.each([
    ["multi-user", { accessToken: "   " }],
    ["single-user", { apiKey: "   " }]
  ])("rejects an empty selected %s credential", (authMode, credential) => {
    expect(
      parseNotificationRuntimeConfig({
        serverUrl: "https://api.example.test",
        authMode,
        ...credential
      })
    ).toBeNull()
  })
})
