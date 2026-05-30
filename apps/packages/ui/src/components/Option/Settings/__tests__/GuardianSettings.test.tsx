import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"

import { GuardianSettings } from "../GuardianSettings"

const expectDesignSystemAlert = (text: string | RegExp) => {
  const node =
    typeof text === "string"
      ? screen.getByText(text, { exact: false })
      : screen.getByText(text)

  expect(node.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
}

const {
  useQueryMock,
  useMutationMock,
  useQueryClientMock,
  invalidateQueriesMock,
  useCanonicalConnectionConfigMock,
  serverCapabilitiesState,
  connectionState
} = vi.hoisted(() => ({
  useQueryMock: vi.fn(),
  useMutationMock: vi.fn(),
  useQueryClientMock: vi.fn(),
  invalidateQueriesMock: vi.fn(),
  useCanonicalConnectionConfigMock: vi.fn(),
  serverCapabilitiesState: {
    capabilities: {
      hasGuardian: true,
      hasSelfMonitoring: true
    },
    loading: false
  } as {
    capabilities: { hasGuardian: boolean; hasSelfMonitoring: boolean } | null
    loading: boolean
  },
  connectionState: {
    uxState: "connected_ok" as
      | "connected_ok"
      | "testing"
      | "configuring_url"
      | "configuring_auth"
      | "error_auth"
      | "error_unreachable"
      | "unconfigured",
    navigate: vi.fn()
  }
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: useQueryMock,
  useMutation: useMutationMock,
  useQueryClient: useQueryClientMock
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  )
  return {
    ...actual,
    useNavigate: () => connectionState.navigate
  }
})

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionUxState: () => ({
    uxState: connectionState.uxState,
    hasCompletedFirstRun: true
  })
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => serverCapabilitiesState
}))

vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: (...args: unknown[]) =>
    useCanonicalConnectionConfigMock(...args)
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

const fetchMock = vi.fn()
vi.stubGlobal("fetch", fetchMock)

describe("GuardianSettings", () => {
  const originalMatchMedia = window.matchMedia
  const originalResizeObserver = globalThis.ResizeObserver

  const makeQueryResult = (overrides: Record<string, unknown> = {}) =>
    ({
      data: undefined,
      isLoading: false,
      isFetching: false,
      isRefetching: false,
      refetch: vi.fn(),
      ...overrides
    }) as any

  const baseRelationship = {
    id: "rel-12345678",
    guardian_user_id: "guardian-user",
    dependent_user_id: "dependent-user",
    relationship_type: "parent",
    status: "pending",
    consent_given_by_dependent: false,
    consent_given_at: null,
    dependent_visible: true,
    dissolution_reason: null,
    dissolved_at: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z"
  }

  let guardianRelationships = [baseRelationship]
  let dependentRelationships = [baseRelationship]
  let governancePolicies = [
    {
      id: "gov-1",
      owner_user_id: "guardian-user",
      name: "Baseline Policy",
      description: "",
      policy_mode: "self",
      scope_chat_types: "all",
      enabled: true,
      schedule_start: null,
      schedule_end: null,
      schedule_days: null,
      schedule_timezone: "UTC",
      transparent: false,
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z"
    }
  ]
  let crisisResources = {
    disclaimer: "If someone is in immediate danger, contact local emergency services.",
    resources: []
  }

  const setQueryMock = () => {
    vi.mocked(useQuery).mockImplementation((options: any) => {
      const queryKey = Array.isArray(options?.queryKey) ? options.queryKey : []
      if (queryKey[0] === "guardian" && queryKey[1] === "relationships") {
        const role = queryKey[2] === "dependent" ? "dependent" : "guardian"
        const items = role === "dependent" ? dependentRelationships : guardianRelationships
        return makeQueryResult({
          data: { items, total: items.length }
        })
      }
      if (queryKey[0] === "guardian" && queryKey[1] === "policies") {
        return makeQueryResult({
          data: { items: [], total: 0 }
        })
      }
      if (queryKey[0] === "guardian" && queryKey[1] === "audit") {
        return makeQueryResult({
          data: { items: [], total: 0 }
        })
      }
      if (queryKey[0] === "guardian" && queryKey[1] === "governance") {
        return makeQueryResult({
          data: { items: governancePolicies, total: governancePolicies.length }
        })
      }
      if (queryKey[0] === "guardian" && queryKey[1] === "crisis") {
        return makeQueryResult({
          data: crisisResources
        })
      }
      return makeQueryResult()
    })
  }

  beforeAll(() => {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: vi.fn().mockImplementation((query: string) => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn()
      }))
    })
    class ResizeObserverMock {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
    Object.defineProperty(globalThis, "ResizeObserver", {
      writable: true,
      value: ResizeObserverMock
    })
  })

  afterAll(() => {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: originalMatchMedia
    })
    Object.defineProperty(globalThis, "ResizeObserver", {
      writable: true,
      value: originalResizeObserver
    })
  })

  beforeEach(() => {
    fetchMock.mockReset()
    serverCapabilitiesState.loading = false
    serverCapabilitiesState.capabilities = {
      hasGuardian: true,
      hasSelfMonitoring: true
    }
    connectionState.uxState = "connected_ok"
    connectionState.navigate.mockReset()
    useCanonicalConnectionConfigMock.mockReset()
    useCanonicalConnectionConfigMock.mockReturnValue({
      config: {
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user",
        apiKey: "test-key"
      },
      loading: false
    })
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        paths: {
          "/api/v1/self-monitoring/rules": {},
          "/api/v1/self-monitoring/alerts": {},
          "/api/v1/self-monitoring/alerts/unread-count": {},
          "/api/v1/self-monitoring/governance-policies": {}
        }
      })
    })

    invalidateQueriesMock.mockReset()
    useQueryClientMock.mockReset()
    useQueryClientMock.mockReturnValue({
      invalidateQueries: invalidateQueriesMock
    })
    useMutationMock.mockReset()
    useMutationMock.mockReturnValue({
      mutate: vi.fn(),
      isPending: false
    })

    guardianRelationships = [baseRelationship]
    dependentRelationships = [baseRelationship]
    governancePolicies = [
      {
        id: "gov-1",
        owner_user_id: "guardian-user",
        name: "Baseline Policy",
        description: "",
        policy_mode: "self",
        scope_chat_types: "all",
        enabled: true,
        schedule_start: null,
        schedule_end: null,
        schedule_days: null,
        schedule_timezone: "UTC",
        transparent: false,
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z"
      }
    ]
    crisisResources = {
      disclaimer: "If someone is in immediate danger, contact local emergency services.",
      resources: []
    }

    vi.mocked(useQueryClient).mockReturnValue({
      invalidateQueries: invalidateQueriesMock
    } as any)
    vi.mocked(useMutation).mockReturnValue({
      mutate: vi.fn(),
      isPending: false
    } as any)
    setQueryMock()
  })

  it("shows unavailable state when guardian capabilities are missing", () => {
    serverCapabilitiesState.capabilities = {
      hasGuardian: false,
      hasSelfMonitoring: false
    }
    render(<GuardianSettings />)

    expect(screen.getByText("Guardian settings unavailable")).toBeInTheDocument()
    expectDesignSystemAlert("Guardian settings unavailable")
    expectDesignSystemAlert(
      "Your current server does not expose Guardian or Self-Monitoring endpoints."
    )
    expect(screen.queryByRole("tab", { name: /Self-Monitoring/i })).not.toBeInTheDocument()
  })

  it("shows unavailable state when capabilities fail to resolve", () => {
    serverCapabilitiesState.capabilities = null
    render(<GuardianSettings />)

    expect(screen.getByText("Guardian settings unavailable")).toBeInTheDocument()
    expect(screen.queryByRole("tab", { name: /Self-Monitoring/i })).not.toBeInTheDocument()
  })

  it("shows self-monitoring fallback guidance when endpoints are missing", async () => {
    const notFoundError = Object.assign(new Error("Request failed: 404"), {
      status: 404
    })
    vi.mocked(useQuery).mockImplementation((options: any) => {
      const queryKey = Array.isArray(options?.queryKey) ? options.queryKey : []
      if (queryKey[0] === "guardian" && queryKey[1] === "rules") {
        return makeQueryResult({
          data: undefined,
          error: notFoundError,
          isError: true
        })
      }
      return makeQueryResult()
    })

    render(<GuardianSettings />)

    expect(
      await screen.findByText("Self-Monitoring endpoints unavailable")
    ).toBeInTheDocument()
    expectDesignSystemAlert("Self-Monitoring endpoints unavailable")
    expectDesignSystemAlert(
      "Your connected server does not expose self-monitoring endpoints yet. Upgrade the server to use Guardian monitoring rules."
    )
  })

  it("shows self-monitoring fallback guidance without probing missing endpoints when OpenAPI lacks them", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        paths: {}
      })
    })

    render(<GuardianSettings />)

    expect(
      await screen.findByText("Self-Monitoring endpoints unavailable")
    ).toBeInTheDocument()

    const rulesQueryCall = vi
      .mocked(useQuery)
      .mock.calls.find(
        ([options]) =>
          Array.isArray((options as any)?.queryKey) &&
          (options as any).queryKey[0] === "guardian" &&
          (options as any).queryKey[1] === "rules"
      )

    expect((rulesQueryCall?.[0] as any)?.enabled).toBe(false)
  })

  it("shows guardian controls fallback guidance through the design-system alert", async () => {
    const notFoundError = Object.assign(new Error("Request failed: 404"), {
      status: 404
    })
    vi.mocked(useQuery).mockImplementation((options: any) => {
      const queryKey = Array.isArray(options?.queryKey) ? options.queryKey : []
      if (queryKey[0] === "guardian" && queryKey[1] === "relationships") {
        return makeQueryResult({
          data: undefined,
          error: notFoundError,
          isError: true
        })
      }
      return makeQueryResult({
        data: { items: [], total: 0 }
      })
    })

    render(<GuardianSettings />)

    fireEvent.click(screen.getByRole("tab", { name: /Guardian Controls/i }))

    expect(
      await screen.findByText("Guardian controls endpoints unavailable")
    ).toBeInTheDocument()
    expectDesignSystemAlert("Guardian controls endpoints unavailable")
    expectDesignSystemAlert(
      "This server does not expose guardian relationship APIs yet. Upgrade the server to manage guardian controls."
    )
  })

  it("shows crisis disclaimer guidance through the design-system alert", async () => {
    render(<GuardianSettings />)

    fireEvent.click(screen.getByRole("tab", { name: /Crisis Resources/i }))

    expect(await screen.findByText("Disclaimer")).toBeInTheDocument()
    expectDesignSystemAlert("Disclaimer")
    expectDesignSystemAlert(
      "If someone is in immediate danger, contact local emergency services."
    )
  })

  it("does not offer warn as a self-monitoring rule action", async () => {
    render(<GuardianSettings />)

    fireEvent.click(
      await screen.findByRole("button", { name: /Create Rule/i })
    )

    const dialog = await screen.findByRole("dialog")
    const actionLabel = within(dialog).getByText("Action")
    const actionFormItem = actionLabel.closest(".ant-form-item")
    const actionSelector = actionFormItem?.querySelector('[role="combobox"]')
    expect(actionSelector).not.toBeNull()
    fireEvent.mouseDown(actionSelector as Element)

    await waitFor(() => {
      expect(document.querySelector('.ant-select-item-option[title="Notify"]')).toBeInTheDocument()
      expect(document.querySelector('.ant-select-item-option[title="Redact"]')).toBeInTheDocument()
      expect(document.querySelector('.ant-select-item-option[title="Block"]')).toBeInTheDocument()
    })

    expect(document.querySelector('.ant-select-item-option[title="Warn"]')).toBeNull()
  }, 30000)

  it("shows Accept action only in dependent view", async () => {
    render(<GuardianSettings />)

    fireEvent.click(screen.getByRole("tab", { name: /Guardian Controls/i }))

    expect(screen.queryByRole("button", { name: /Accept/i })).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("radio", { name: /Dependent View/i }))

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Accept/i })).toBeInTheDocument()
    })
  }, 30000)

  it("shows Accept action for pending_consent relationships in dependent view", async () => {
    guardianRelationships = [
      {
        ...baseRelationship,
        status: "pending_consent",
        consent_given_by_dependent: false,
        consent_given_at: null
      }
    ]
    dependentRelationships = [...guardianRelationships]

    render(<GuardianSettings />)

    fireEvent.click(screen.getByRole("tab", { name: /Guardian Controls/i }))
    fireEvent.click(screen.getByRole("radio", { name: /Dependent View/i }))

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Accept/i })).toBeInTheDocument()
    })
  }, 15000)

  it("updates add-policy enablement when selected relationship status changes", async () => {
    guardianRelationships = [
      {
        ...baseRelationship,
        id: "rel-active-1",
        dependent_user_id: "dep-active",
        status: "active",
        consent_given_by_dependent: true,
        consent_given_at: "2026-01-01T00:05:00Z"
      }
    ]

    const view = render(<GuardianSettings />)

    fireEvent.click(screen.getByRole("tab", { name: /Guardian Controls/i }))
    const initialCell = await screen.findByText("dep-active")
    fireEvent.click(initialCell.closest("tr") ?? initialCell)

    const addPolicyButton = await screen.findByRole("button", { name: /Add Policy/i })
    await waitFor(() => {
      expect(addPolicyButton).toBeEnabled()
    })

    guardianRelationships = [
      {
        ...guardianRelationships[0],
        status: "suspended",
        updated_at: "2026-01-01T00:10:00Z"
      }
    ]

    view.rerender(<GuardianSettings />)
    const updatedCell = await screen.findByText("dep-active")
    fireEvent.click(updatedCell.closest("tr") ?? updatedCell)

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Add Policy/i })).toBeDisabled()
    })
  }, 30000)

  it("renders governance policies and updates when query data changes", async () => {
    const view = render(<GuardianSettings />)

    expect(await screen.findByText("Governance Policies")).toBeInTheDocument()
    expect(await screen.findByText("Baseline Policy")).toBeInTheDocument()

    governancePolicies = [
      ...governancePolicies,
      {
        id: "gov-2",
        owner_user_id: "guardian-user",
        name: "Evening Policy",
        description: "",
        policy_mode: "guardian",
        scope_chat_types: "all",
        enabled: true,
        schedule_start: null,
        schedule_end: null,
        schedule_days: null,
        schedule_timezone: "UTC",
        transparent: true,
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z"
      }
    ]

    view.rerender(<GuardianSettings />)
    await waitFor(() => {
      expect(screen.getByText("Evening Policy")).toBeInTheDocument()
    })

    governancePolicies = governancePolicies.filter((policy) => policy.id !== "gov-1")

    view.rerender(<GuardianSettings />)
    await waitFor(() => {
      expect(screen.queryByText("Baseline Policy")).not.toBeInTheDocument()
      expect(screen.getByText("Evening Policy")).toBeInTheDocument()
    })
  })
})
