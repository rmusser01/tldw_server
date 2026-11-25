import { afterEach, expect, test, vi } from "vitest"

import { chromiumFixtureOptions } from "../outbound-observer"

vi.mock("@playwright/test", () => ({
  chromium: { executablePath: () => "/fixture/chromium" }
}))

afterEach(() => vi.unstubAllEnvs())

test("the WebSocket calibration uses Chromium directly with the fixture network restrictions", () => {
  vi.stubEnv("OUTBOUND_CHROMIUM_WRAPPER", "/fixture/trace-chromium")

  const application = chromiumFixtureOptions()
  const calibration = chromiumFixtureOptions({ trace: false })

  expect(application.executablePath).toBe("/fixture/trace-chromium")
  expect(application.env?.OUTBOUND_CHROMIUM_BINARY).toBe("/fixture/chromium")
  expect(calibration.executablePath).toBe("/fixture/chromium")
  expect(calibration.args).toEqual(application.args)
  expect(calibration.args).toContain("--no-proxy-server")
  expect(calibration.args).toContain("--host-resolver-rules=EXCLUDE 127.0.0.1, MAP * ~NOTFOUND")
})
