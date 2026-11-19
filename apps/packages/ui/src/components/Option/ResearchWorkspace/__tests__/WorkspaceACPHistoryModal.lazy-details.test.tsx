import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter } from "react-router-dom"

import { WorkspaceACPHistoryModal } from "../WorkspaceACPHistoryModal"

const fetchMock = vi.hoisted(() => vi.fn())
// Stable reference: the modal's load effect depends on the config identity,
// so a fresh object per render would loop the effect.
const connectionConfig = vi.hoisted(() => ({ serverUrl: "http://svc.test" }))
// The modal reads history through the scoped workspace-acp-history service
// (bgRequest); reads must carry the service-prompt scope (WP-44 kept the
// dev scoped reads, only the detail layer became lazy).
const bgRequestMock = vi.hoisted(() => vi.fn())
const acpScope = vi.hoisted(() => {
  const controller = new AbortController()
  return {
    scope: {
      scopeKey: "acp-history-lazy-test",
      capability: "unchecked",
      definitions: {},
      requestScope: {
        config: { serverUrl: "http://svc.test", authMode: "single-user" },
        userId: 1
      },
      scopeSignal: controller.signal,
      scopeInvalidatedSignal: controller.signal,
      release: () => undefined
    }
  }
})

vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => ({
    config: connectionConfig,
    loading: false
  })
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => bgRequestMock(...args)
}))

vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: async () => acpScope.scope
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: unknown) => {
      if (typeof fallback === "string") return fallback
      if (fallback && typeof fallback === "object") {
        const options = fallback as { defaultValue?: string }
        return options.defaultValue ?? key
      }
      return key
    }
  })
}))

const WORKSPACE_ID = "ws-1"

const jsonResponse = (payload: unknown) =>
  Promise.resolve({
    ok: true,
    status: 200,
    json: async () => payload
  })

const routeFetch = (tasks: Record<number, { runs?: unknown[] }>) =>
  fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
    const url = String(input)
    if (url.startsWith("http://svc.test/api/v1/acp/sessions")) {
      return jsonResponse({ sessions: [] })
    }
    if (
      url.startsWith(
        "http://svc.test/api/v1/agent-orchestration/projects"
      ) &&
      !url.includes("/tasks")
    ) {
      return jsonResponse({
        projects: [
          {
            id: 1,
            name: "Research project",
            canonical_workspace: { canonical_workspace_id: WORKSPACE_ID }
          }
        ]
      })
    }
    if (url === "http://svc.test/api/v1/agent-orchestration/projects/1/tasks") {
      return jsonResponse({
        tasks: [
          {
            id: 11,
            project_id: 1,
            title: "First task",
            status: "completed",
            updated_at: "2026-10-02T10:00:00Z"
          },
          {
            id: 12,
            project_id: 1,
            title: "Second task",
            status: "failed",
            updated_at: "2026-10-01T10:00:00Z"
          }
        ]
      })
    }
    const detailMatch = url.match(
      /\/api\/v1\/agent-orchestration\/tasks\/(\d+)$/
    )
    if (detailMatch) {
      const taskId = Number(detailMatch[1])
      const stub = tasks[taskId] ?? { runs: [] }
      return jsonResponse({
        id: taskId,
        project_id: 1,
        title: taskId === 11 ? "First task" : "Second task",
        status: taskId === 11 ? "completed" : "failed",
        ...stub
      })
    }
    return jsonResponse({})
  })

const requestedPaths = () =>
  fetchMock.mock.calls.map(([input]) => String(input))

const renderModal = () =>
  render(
    <MemoryRouter>
      <WorkspaceACPHistoryModal
        open
        workspaceId={WORKSPACE_ID}
        workspaceName="Workspace"
        onCancel={vi.fn()}
        onOpenAgentTasks={vi.fn()}
      />
    </MemoryRouter>
  )

describe("WorkspaceACPHistoryModal lazy task details (WP-44)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubGlobal("fetch", fetchMock)
    bgRequestMock.mockImplementation(
      async (init: {
        path: string
        abortSignal?: AbortSignal
        headers?: Record<string, string>
      }) => {
        const response = await fetchMock(`http://svc.test${init.path}`, {
          headers: init.headers,
          signal: init.abortSignal
        })
        if (!response.ok) {
          const payload = await response.json().catch(() => null)
          const detail =
            payload && typeof payload === "object"
              ? (payload as { detail?: unknown }).detail
              : null
          const message =
            typeof detail === "string"
              ? detail
              : detail && typeof detail === "object"
                ? (detail as Record<string, unknown>).message ||
                  (detail as Record<string, unknown>).code ||
                  `HTTP ${response.status}`
                : `HTTP ${response.status}`
          throw Object.assign(new Error(message), {
            status: response.status,
            details: payload
          })
        }
        return response.json()
      }
    )
  })

  it("opens with the tasks layer only — no eager task-detail fan-out", async () => {
    const tasks: Record<number, { runs?: unknown[] }> = {
      11: {
        runs: [
          {
            id: 111,
            task_id: 11,
            status: "completed",
            result_summary: "Run done"
          }
        ]
      },
      12: { runs: [] }
    }
    routeFetch(tasks)

    renderModal()

    expect(await screen.findByText("First task")).toBeInTheDocument()
    expect(screen.getByText("Second task")).toBeInTheDocument()

    await waitFor(() => {
      expect(requestedPaths()).toEqual(
        expect.arrayContaining([
          expect.stringContaining("/api/v1/agent-orchestration/projects/1/tasks"),
          expect.stringContaining("/api/v1/acp/sessions")
        ])
      )
    })

    expect(requestedPaths()).toEqual(
      expect.arrayContaining([
        expect.stringContaining("canonical_workspace_id=ws-1")
      ])
    )
    expect(
      requestedPaths().filter((path) =>
        /\/agent-orchestration\/tasks\/\d+$/.test(path)
      )
    ).toEqual([])
  })

  it("fetches a task's runs only when its row is expanded, and caches the result", async () => {
    const tasks: Record<number, { runs?: unknown[] }> = {
      11: {
        runs: [
          {
            id: 111,
            task_id: 11,
            status: "completed",
            result_summary: "Run done"
          }
        ]
      },
      12: { runs: [] }
    }
    routeFetch(tasks)

    const user = userEvent.setup()
    renderModal()

    await screen.findByText("First task")

    await user.click(screen.getByText("First task"))

    expect(await screen.findByText("Run done")).toBeInTheDocument()
    expect(
      requestedPaths().filter((path) =>
        /\/agent-orchestration\/tasks\/\d+$/.test(path)
      )
    ).toEqual(["http://svc.test/api/v1/agent-orchestration/tasks/11"])

    // Collapse and expand again: the cached runs must not refetch.
    await user.click(screen.getByText("First task"))
    await user.click(screen.getByText("First task"))
    expect(await screen.findByText("Run done")).toBeInTheDocument()
    expect(
      requestedPaths().filter((path) =>
        /\/agent-orchestration\/tasks\/\d+$/.test(path)
      )
    ).toEqual(["http://svc.test/api/v1/agent-orchestration/tasks/11"])
  })
})
