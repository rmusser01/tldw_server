import { describe, expect, it } from "vitest"
import { isServicePromptRequestPath } from "../service-prompt-scope-error"

describe("captured document annotation routes", () => {
  it.each([
    ["GET", "/api/v1/media/12/progress"],
    ["PUT", "/api/v1/media/12/progress"],
    ["DELETE", "/api/v1/media/12/progress"],
    ["GET", "/api/v1/media/12/annotations"],
    ["POST", "/api/v1/media/12/annotations"],
    ["POST", "/api/v1/media/12/annotations/sync"],
    ["PUT", "/api/v1/media/12/annotations/ann_123"],
    ["DELETE", "/api/v1/media/12/annotations/ann_123"]
  ])("admits document operation %s %s", (method, path) =>
    expect(isServicePromptRequestPath(path, method)).toBe(true)
  )
  it.each([
    ["POST", "/api/v1/media/12/progress"],
    ["PUT", "/api/v1/media/0/progress"],
    ["DELETE", "/api/v1/media/12/annotations"],
    ["PATCH", "/api/v1/media/12/annotations/ann_123"],
    ["GET", "/api/v1/media/12/annotations/sync"],
    ["POST", "/api/v1/media/12/annotations/ann_123"],
    ["POST", "/api/v1/media/0/annotations"],
    ["GET", "/api/v1/media/12/annotations/../admin"],
    ["DELETE", "/api/v1/media/12/annotations/ann%2felsewhere"]
  ])("refuses unrelated operation %s %s", (method, path) =>
    expect(isServicePromptRequestPath(path, method)).toBe(false)
  )
})
