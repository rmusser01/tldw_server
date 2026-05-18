// @vitest-environment jsdom

import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useSetupReadiness } from "../useSetupReadiness"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args)
}))

vi.mock("@/services/tldw/path-utils", () => ({
  toAllowedPath: (path: string) => path
}))

const createResponse = (data: unknown, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  data,
  error: status >= 400 ? `Request failed: ${status}` : undefined
})

const profilesPayload = {
  setup_access: {
    mode: "first_run",
    needs_setup: true,
    setup_completed: false,
    remote_access_active: false
  },
  lane_ids: ["chat", "embeddings_rag", "speech"],
  supported_statuses: ["not_configured", "ready_with_warnings"],
  supported_overlays: ["restart_required"],
  active_overlays: [],
  overlays: [],
  lanes: [
    { lane_id: "chat", status: "not_configured", label: "Chat" },
    { lane_id: "embeddings_rag", status: "ready_with_warnings", label: "Embeddings/RAG" },
    { lane_id: "speech", status: "not_configured", label: "Speech" }
  ],
  profiles: [
    {
      profile_id: "local_balanced",
      label: "Local Balanced",
      lanes: {}
    }
  ],
  recommended_profile_id: "local_balanced"
}

const statusPayload = {
  ...profilesPayload,
  readiness_status: "not_started",
  operation_status: null
}

describe("useSetupReadiness", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.useRealTimers()
  })

  it("loads readiness profiles and status without provisioning", async () => {
    const requests: string[] = []
    mocks.bgRequest.mockImplementation((init: { path: string }) => {
      requests.push(String(init.path))
      if (String(init.path).endsWith("/profiles")) {
        return Promise.resolve(createResponse(profilesPayload))
      }
      if (String(init.path).endsWith("/status")) {
        return Promise.resolve(createResponse(statusPayload))
      }
      throw new Error(`Unexpected request: ${init.path}`)
    })

    const { result } = renderHook(() => useSetupReadiness())

    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.profiles?.recommended_profile_id).toBe("local_balanced")
    expect(requests).toContain("/api/v1/setup/readiness/profiles")
    expect(requests).toContain("/api/v1/setup/readiness/status")
    expect(requests).not.toContain("/api/v1/setup/readiness/provision")
  })

  it("maps first-run setup guard failures to the backend setup fallback", async () => {
    mocks.bgRequest.mockResolvedValue(createResponse({ detail: "Setup access is restricted to local requests." }, 403))

    const { result } = renderHook(() => useSetupReadiness())

    await waitFor(() => expect(result.current.guard).toBe("remote_setup_blocked"))
    expect(result.current.fallbackUrl).toBe("/setup")
    expect(result.current.error).toContain("Setup access is restricted")
  })

  it("uses admin setup readiness endpoints in admin mode", async () => {
    const requests: string[] = []
    mocks.bgRequest.mockImplementation((init: { path: string }) => {
      requests.push(String(init.path))
      if (String(init.path).endsWith("/profiles")) {
        return Promise.resolve(
          createResponse({
            ...profilesPayload,
            setup_access: { ...profilesPayload.setup_access, mode: "admin", needs_setup: false }
          })
        )
      }
      if (String(init.path).endsWith("/status")) {
        return Promise.resolve(
          createResponse({
            ...statusPayload,
            setup_access: { ...statusPayload.setup_access, mode: "admin", needs_setup: false }
          })
        )
      }
      throw new Error(`Unexpected request: ${init.path}`)
    })

    const { result } = renderHook(() => useSetupReadiness({ mode: "admin" }))

    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(requests).toContain("/api/v1/setup/admin/readiness/profiles")
    expect(requests).toContain("/api/v1/setup/admin/readiness/status")
    expect(result.current.profiles?.setup_access.mode).toBe("admin")
  })

  it("keeps preview and provisioning as separate explicit actions", async () => {
    const requests: Array<{ path: string; body?: string }> = []
    mocks.bgRequest.mockImplementation((init: { path: string; body?: string }) => {
      requests.push({ path: String(init.path), body: init.body })
      if (String(init.path).endsWith("/profiles")) {
        return Promise.resolve(createResponse(profilesPayload))
      }
      if (String(init.path).endsWith("/status")) {
        return Promise.resolve(createResponse(statusPayload))
      }
      if (String(init.path).endsWith("/preview")) {
        return Promise.resolve(
          createResponse({
            preview_id: "preview-1",
            profile_id: "local_balanced",
            lane_ids: ["chat", "embeddings_rag", "speech"],
            lanes: {},
            overlays: [],
            config_updates: {},
            secret_fields: [],
            install_plan: {},
            operation_required: true
          })
        )
      }
      if (String(init.path).endsWith("/provision")) {
        return Promise.resolve(
          createResponse({
            operation_id: "operation-1",
            operation_status: "queued",
            status_url: "/api/v1/setup/readiness/status",
            status: "provisioning",
            lanes: [],
            overlays: [],
            install_plan_submitted: true,
            config_updates_applied: true
          })
        )
      }
      throw new Error(`Unexpected request: ${init.path}`)
    })

    const { result } = renderHook(() => useSetupReadiness())

    await waitFor(() => expect(result.current.loading).toBe(false))
    await act(async () => {
      await result.current.previewSelection({ profile_id: "local_balanced" })
    })

    expect(requests.some((request) => request.path === "/api/v1/setup/readiness/preview")).toBe(true)
    expect(requests.some((request) => request.path === "/api/v1/setup/readiness/provision")).toBe(false)

    await act(async () => {
      await result.current.provision({ preview_id: "preview-1" })
    })

    const provisionRequest = requests.find(
      (request) => request.path === "/api/v1/setup/readiness/provision"
    )
    expect(provisionRequest).toBeTruthy()
    expect(JSON.parse(provisionRequest?.body || "{}")).toEqual({
      preview_id: "preview-1",
      confirmed: true
    })
  })
})
