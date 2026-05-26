import { describe, expect, it } from "vitest"

import {
  buildCapabilityState,
  getCapabilityErrorStatus,
  getCapabilityRawMessage
} from "../capability-state"

const endpoint = "/api/v1/ingestion-sources"

const errorWithStatus = (message: string, status: number) =>
  Object.assign(new Error(message), { status })

const renderedText = (value: unknown): string => {
  if (value == null) return ""
  if (typeof value === "string") return value
  if (typeof value === "number" || typeof value === "boolean") return String(value)
  return JSON.stringify(value)
}

describe("capability-state", () => {
  it("maps missing endpoint errors to user-language unavailable states with endpoint diagnostics", () => {
    const state = buildCapabilityState({
      featureName: "Sources",
      capabilityName: "ingestion source management",
      endpoint,
      method: "GET",
      error: errorWithStatus(`Request failed: 404 (GET ${endpoint})`, 404)
    })

    expect(state.state).toBe("unavailable")
    expect(state.title).toBe("Sources are unavailable on this server")
    expect(state.message).toBe(
      "The connected server does not advertise ingestion source management."
    )
    expect(`${state.title} ${state.message}`).not.toContain(endpoint)
    expect(state.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ label: "Request path", value: endpoint, code: true }),
        expect.objectContaining({ label: "Status", value: "404" }),
        expect.objectContaining({
          label: "Raw message",
          value: `Request failed: 404 (GET ${endpoint})`
        })
      ])
    )
  })

  it("maps auth and permission statuses to explicit recovery states", () => {
    expect(
      buildCapabilityState({
        featureName: "Scheduled tasks",
        capabilityName: "scheduled task management",
        endpoint: "/api/v1/scheduled-tasks",
        error: errorWithStatus("HTTP 401", 401)
      })
    ).toMatchObject({
      state: "auth_required",
      title: "Sign in before using scheduled tasks"
    })

    expect(
      buildCapabilityState({
        featureName: "Personal integrations",
        capabilityName: "personal integration management",
        endpoint: "/api/v1/integrations/personal",
        error: errorWithStatus("HTTP 403", 403)
      })
    ).toMatchObject({
      state: "permission_denied",
      title: "You do not have access to personal integrations"
    })
  })

  it("maps setup gaps, network failures, and partial failures to stable state keys", () => {
    expect(
      buildCapabilityState({
        featureName: "Telegram bot",
        capabilityName: "Telegram bot settings",
        endpoint: "/api/v1/integrations/workspace/telegram/bot",
        reason: "missing_config"
      })
    ).toMatchObject({
      state: "setup_required",
      title: "Telegram bot needs setup"
    })

    expect(
      buildCapabilityState({
        featureName: "Sources",
        capabilityName: "ingestion source management",
        endpoint,
        error: new Error("Failed to fetch")
      })
    ).toMatchObject({
      state: "unavailable",
      title: "Cannot reach Sources"
    })

    const partial = buildCapabilityState({
      featureName: "Scheduled tasks",
      capabilityName: "scheduled task management",
      endpoint: "/api/v1/scheduled-tasks",
      reason: "partial",
      partialErrors: ["Watchlist jobs failed at /api/v1/watchlists/jobs"]
    })

    expect(partial.state).toBe("degraded")
    expect(partial.message).not.toContain("/api/v1/watchlists/jobs")
    expect(partial.diagnostics?.map((diagnostic) => renderedText(diagnostic.value)).join(" ")).toContain(
      "/api/v1/watchlists/jobs"
    )
  })

  it("extracts status and raw messages from common query error shapes", () => {
    expect(getCapabilityErrorStatus({ response: { status: "403" } })).toBe(403)
    expect(getCapabilityErrorStatus(new Error("Request failed: 404"))).toBe(404)
    expect(getCapabilityRawMessage({ message: "Network error" })).toBe("Network error")
    expect(getCapabilityRawMessage("load failed")).toBe("load failed")
  })
})
