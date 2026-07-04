import { afterEach, describe, expect, it } from "vitest"
import {
  buildACPAuthHeaders,
  buildACPAuthParams
} from "@/services/acp/connection"
import {
  clearRuntimeAuthOverride,
  setRuntimeSingleUserApiKeyOverride
} from "@/services/tldw/runtime-auth-override"

describe("ACP connection auth", () => {
  afterEach(() => {
    clearRuntimeAuthOverride()
  })

  it("uses runtime single-user auth when tldwConfig has been scrubbed", () => {
    setRuntimeSingleUserApiKeyOverride("runtime-single-user-key")

    expect(
      buildACPAuthHeaders({
        authMode: "single-user",
        serverUrl: "http://127.0.0.1:8000"
      })
    ).toMatchObject({
      "X-API-KEY": "runtime-single-user-key"
    })

    expect(
      buildACPAuthParams({
        authMode: "single-user",
        serverUrl: "http://127.0.0.1:8000"
      })
    ).toMatchObject({
      api_key: "runtime-single-user-key"
    })
  })
})
