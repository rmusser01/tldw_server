import { describe, expect, it } from "vitest"
import { formatModelsLastRefreshedTime } from "../modelsDisplayUtils"

describe("models display utilities", () => {
  it.each([
    [new Date(2026, 1, 18, 9, 5).getTime(), "09:05"],
    [new Date(2026, 1, 18, 0, 7).getTime(), "00:07"],
  ])("formats %s as a dayjs-compatible HH:mm time", (value, expected) => {
    expect(formatModelsLastRefreshedTime(value)).toBe(expected)
  })
})
