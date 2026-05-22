import { describe, expect, it } from "vitest"
import {
  isWatchlistRunActive,
  isWatchlistRunSuccessful,
  isWatchlistRunTerminal,
  normalizeWatchlistRunStatus
} from "../runStatus"

describe("watchlist run status helpers", () => {
  it("normalizes backend success aliases to completed", () => {
    expect(normalizeWatchlistRunStatus("completed")).toBe("completed")
    expect(normalizeWatchlistRunStatus("succeeded")).toBe("completed")
    expect(normalizeWatchlistRunStatus("success")).toBe("completed")
    expect(normalizeWatchlistRunStatus(" Succeeded ")).toBe("completed")
  })

  it("classifies active, successful, and terminal statuses", () => {
    expect(isWatchlistRunActive("queued")).toBe(true)
    expect(isWatchlistRunActive("succeeded")).toBe(false)
    expect(isWatchlistRunSuccessful("succeeded")).toBe(true)
    expect(isWatchlistRunTerminal("succeeded")).toBe(true)
    expect(isWatchlistRunTerminal("failed")).toBe(true)
    expect(isWatchlistRunTerminal("cancelled")).toBe(true)
    expect(isWatchlistRunTerminal("running")).toBe(false)
  })
})
