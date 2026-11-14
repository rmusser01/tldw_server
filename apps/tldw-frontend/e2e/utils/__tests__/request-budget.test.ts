import { describe, expect, it } from "vitest"
import {
  analyseRequests,
  normaliseRequestPath,
  requestKey,
  type RequestBudgetSettings,
  type TrackedRequest,
} from "../request-budget"

const settings: RequestBudgetSettings = {
  windowMs: 15_000,
  duplicateWindowMs: 2_000,
  idleFromMs: 5_000,
  pollingMinHits: 2,
}

const request = (at: number, url: string, method = "GET"): TrackedRequest => {
  const parsed = new URL(url, "http://127.0.0.1:8000")
  return {
    method,
    url: parsed.toString(),
    key: requestKey(method, parsed.toString()),
    exactKey: `${method} ${parsed.pathname}${parsed.search}`,
    at,
  }
}

describe("normaliseRequestPath", () => {
  it("replaces numeric, UUID and long hex segments with :id", () => {
    expect(normaliseRequestPath("/api/v1/notes/42")).toBe("/api/v1/notes/:id")
    expect(normaliseRequestPath("/api/v1/notes/0b9a2c1e-8a3f-4c55-9d2e-1f0e5a6b7c8d/links")).toBe(
      "/api/v1/notes/:id/links"
    )
    expect(normaliseRequestPath("/api/v1/chats/0123456789abcdef0123")).toBe("/api/v1/chats/:id")
    expect(normaliseRequestPath("/api/v1/llm/models/metadata")).toBe("/api/v1/llm/models/metadata")
  })
})

describe("analyseRequests", () => {
  it("counts totals and per-key requests inside the window only", () => {
    const report = analyseRequests(
      [request(100, "/api/v1/notes/?limit=20"), request(200, "/api/v1/notes/7"), request(16_000, "/api/v1/notes/8")],
      settings
    )
    expect(report.total).toBe(2)
    expect(report.perKey).toEqual({ "GET /api/v1/notes/": 1, "GET /api/v1/notes/:id": 1 })
  })

  it("flags identical GETs within the duplicate window, but not other queries or methods", () => {
    const report = analyseRequests(
      [
        request(100, "/api/v1/persona/profiles"),
        request(900, "/api/v1/persona/profiles"),
        request(1_500, "/api/v1/persona/profiles"),
        request(4_000, "/api/v1/persona/profiles"),
        request(100, "/api/v1/notes/?offset=0"),
        request(200, "/api/v1/notes/?offset=20"),
        request(100, "/api/v1/notes/", "POST"),
        request(200, "/api/v1/notes/", "POST"),
      ],
      settings
    )
    expect([...report.duplicateGets.keys()]).toEqual(["GET /api/v1/persona/profiles"])
    expect(report.duplicateGets.get("GET /api/v1/persona/profiles")?.count).toBe(2)
  })

  it("reports endpoints that keep firing after the idle threshold as polling", () => {
    const report = analyseRequests(
      [
        request(1_000, "/api/v1/buddies"),
        request(6_000, "/api/v1/buddies"),
        request(11_000, "/api/v1/buddies"),
        request(7_000, "/api/v1/health"),
      ],
      settings
    )
    expect([...report.polling.keys()]).toEqual(["GET /api/v1/buddies"])
    expect(report.polling.get("GET /api/v1/buddies")?.count).toBe(2)
  })
})
