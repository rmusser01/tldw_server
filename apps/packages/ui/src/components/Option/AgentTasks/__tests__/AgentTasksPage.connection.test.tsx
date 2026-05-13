import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import AgentTasksPage from "../index"

const storageMocks = vi.hoisted(() => ({
  useStorage: vi.fn()
}))

const configMocks = vi.hoisted(() => ({
  getConfig: vi.fn()
}))

const deploymentMocks = vi.hoisted(() => ({
  isHostedTldwDeployment: vi.fn(() => false)
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (...args: unknown[]) => storageMocks.useStorage(...args)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: (...args: unknown[]) => configMocks.getConfig(...args)
  }
}))

vi.mock("@/services/tldw/deployment-mode", () => ({
  isHostedTldwDeployment: () => deploymentMocks.isHostedTldwDeployment()
}))

vi.mock("antd", () => {
  const formApi = {
    resetFields: vi.fn(),
    getFieldValue: vi.fn()
  }
  const Form = Object.assign(
    ({ children }: { children?: React.ReactNode }) => <form>{children}</form>,
    {
      useForm: () => [formApi],
      Item: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>
    }
  )

  return {
  Alert: ({
      title,
      message,
      description
    }: {
      title?: React.ReactNode
      message?: React.ReactNode
      description?: React.ReactNode
    }) => (
      <div>
        <div>{message ?? title}</div>
        {description ? <div>{description}</div> : null}
      </div>
    ),
    Badge: ({ count }: { count?: React.ReactNode }) => <span>{count}</span>,
    Button: ({
      children,
      onClick,
      disabled
    }: {
      children?: React.ReactNode
      onClick?: (event?: React.MouseEvent<HTMLButtonElement>) => void
      disabled?: boolean
    }) => (
      <button type="button" disabled={disabled} onClick={onClick}>
        {children}
      </button>
    ),
    Card: ({
      title,
      extra,
      children
    }: {
      title?: React.ReactNode
      extra?: React.ReactNode
      children?: React.ReactNode
    }) => (
      <section>
        {title}
        {extra}
        {children}
      </section>
    ),
    Collapse: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
    Empty: ({
      description,
      children
    }: {
      description?: React.ReactNode
      children?: React.ReactNode
    }) => (
      <div>
        <div>{description}</div>
        {children}
      </div>
    ),
    Form,
    Input: Object.assign(() => <input />, {
      TextArea: () => <textarea />
    }),
    Modal: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
    Select: ({
      options = [],
      value,
      onChange,
      placeholder,
      "aria-label": ariaLabel
    }: {
      options?: Array<{ value: string | number; label: React.ReactNode }>
      value?: string | number
      onChange?: (value: string | number | undefined) => void
      placeholder?: React.ReactNode
      "aria-label"?: string
    }) => (
      <select
        aria-label={
          ariaLabel || (typeof placeholder === "string" ? placeholder : "select")
        }
        value={value ?? ""}
        onChange={(event) => onChange?.(event.target.value || undefined)}
      >
        {placeholder ? <option value="">{placeholder}</option> : null}
        {options.map((option) => (
          <option key={String(option.value)} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    ),
    Spin: () => <div>Loading...</div>,
    Tag: ({ children }: { children?: React.ReactNode }) => <span>{children}</span>,
    Tooltip: ({ children }: { children?: React.ReactNode }) => <>{children}</>
  }
})

describe("AgentTasksPage connection and payload normalization", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    deploymentMocks.isHostedTldwDeployment.mockReturnValue(false)
    window.location.hash = ""

    storageMocks.useStorage.mockImplementation((key: string, fallback: string) => {
      if (key === "serverUrl") return ["http://localhost:8000", vi.fn()]
      if (key === "authMode") return ["single-user", vi.fn()]
      if (key === "apiKey") return ["", vi.fn()]
      if (key === "accessToken") return ["", vi.fn()]
      return [fallback, vi.fn()]
    })

    configMocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "real-key",
      accessToken: ""
    })

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input)

        if (url === "http://127.0.0.1:8000/openapi.json") {
          return {
            ok: true,
            json: async () => ({
              paths: {
                "/api/v1/agent-orchestration/projects": {},
              }
            })
          }
        }

        if (url === "http://127.0.0.1:8000/api/v1/acp/health") {
          expect((init?.headers as Record<string, string>)?.["X-API-KEY"]).toBe("real-key")
          return {
            ok: true,
            json: async () => ({
              runner: "ok",
              agent: "ok",
              api_keys: "ok"
            })
          }
        }

        if (url === "http://127.0.0.1:8000/api/v1/agent-orchestration/projects") {
          expect((init?.headers as Record<string, string>)?.["X-API-KEY"]).toBe("real-key")
          return {
            ok: true,
            json: async () => [
              {
                id: 7,
                name: "Research Project",
                user_id: 1,
                created_at: "2026-03-20T19:00:00Z",
                task_summary: {
                  total_tasks: 1,
                  status_counts: {
                    todo: 1
                  }
                }
              }
            ]
          }
        }

        if (url === "http://127.0.0.1:8000/api/v1/agent-orchestration/projects/7/tasks") {
          expect((init?.headers as Record<string, string>)?.["X-API-KEY"]).toBe("real-key")
          return {
            ok: true,
            json: async () => [
              {
                id: 11,
                project_id: 7,
                title: "Draft spec",
                status: "todo",
                review_count: 0,
                max_review_attempts: 3,
                created_at: "2026-03-20T19:00:00Z",
                updated_at: "2026-03-20T19:00:00Z"
              }
            ]
          }
        }

        throw new Error(`unexpected fetch: ${url}`)
      })
    )
  })

  it("loads projects and tasks from canonical config-backed requests even when legacy storage is stale", async () => {
    render(<AgentTasksPage />)

    const projectButton = await screen.findByText("Research Project")
    expect(projectButton).toBeInTheDocument()

    fireEvent.click(projectButton)

    expect(await screen.findByText("Draft spec")).toBeInTheDocument()

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledTimes(4)
    })
  })

  it("shows an unsupported-state message instead of surfacing raw HTTP 404 when orchestration routes are absent", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: false,
        status: 404,
        json: async () => ({
          detail: "Not Found"
        })
      }))
    )

    render(<AgentTasksPage />)

    expect(await screen.findByText("Agent orchestration unavailable")).toBeInTheDocument()
    expect(
      screen.getByText("This server does not expose agent orchestration endpoints.")
    ).toBeInTheDocument()
    expect(screen.queryByText("HTTP 404")).toBeNull()
  })

  it("uses the OpenAPI spec to suppress project probes when orchestration routes are absent", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url === "http://127.0.0.1:8000/openapi.json") {
        return {
          ok: true,
          json: async () => ({
            paths: {
              "/api/v1/health": {}
            }
          })
        }
      }
      if (url === "http://127.0.0.1:8000/api/v1/acp/health") {
        return {
          ok: true,
          json: async () => ({
            runner: "ok",
            agent: "ok",
            api_keys: "ok"
          })
        }
      }
      throw new Error(`unexpected fetch: ${url}`)
    })
    vi.stubGlobal("fetch", fetchMock)

    render(<AgentTasksPage />)

    expect(await screen.findByText("Agent orchestration unavailable")).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/openapi.json"
    )
  })

  it("shows actionable ACP setup gaps from the shared health state", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url === "http://127.0.0.1:8000/openapi.json") {
        return {
          ok: true,
          json: async () => ({
            paths: {
              "/api/v1/agent-orchestration/projects": {},
            }
          })
        }
      }
      if (url === "http://127.0.0.1:8000/api/v1/acp/health") {
        expect((init?.headers as Record<string, string>)?.["X-API-KEY"]).toBe("real-key")
        return {
          ok: true,
          json: async () => ({
            runner: {
              status: "missing",
              source: "PATH"
            },
            agents: [
              {
                agent_type: "codex",
                status: "unavailable",
                api_key_set: false
              }
            ],
            overall: "unavailable",
            message: "ACP runner not found"
          })
        }
      }
      if (url === "http://127.0.0.1:8000/api/v1/agent-orchestration/projects") {
        return {
          ok: true,
          json: async () => []
        }
      }
      throw new Error(`unexpected fetch: ${url}`)
    })
    vi.stubGlobal("fetch", fetchMock)

    render(<AgentTasksPage />)

    expect(await screen.findByText("ACP setup needs attention")).toBeInTheDocument()
    expect(screen.getByText("Runner is missing")).toBeInTheDocument()
    expect(screen.getByText("API keys are missing")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /open agent registry/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /open acp playground/i })).toBeInTheDocument()
  })

  it("omits stored ACP credentials from hosted-mode proxy requests", async () => {
    deploymentMocks.isHostedTldwDeployment.mockReturnValue(true)
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      const headers = (init?.headers as Record<string, string> | undefined) ?? {}
      if (url === "/openapi.json") {
        return {
          ok: true,
          json: async () => ({
            paths: {
              "/api/v1/agent-orchestration/projects": {},
            }
          })
        }
      }
      if (url === "/api/proxy/acp/health") {
        expect(headers["X-API-KEY"]).toBeUndefined()
        expect(headers.Authorization).toBeUndefined()
        return {
          ok: true,
          json: async () => ({
            runner: "ok",
            agent: "ok",
            api_keys: "ok"
          })
        }
      }
      if (url === "/api/proxy/agent-orchestration/projects") {
        expect(headers["X-API-KEY"]).toBeUndefined()
        expect(headers.Authorization).toBeUndefined()
        return {
          ok: true,
          json: async () => []
        }
      }
      throw new Error(`unexpected fetch: ${url}`)
    })
    vi.stubGlobal("fetch", fetchMock)

    render(<AgentTasksPage />)

    expect(await screen.findByText("No projects yet")).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/proxy/agent-orchestration/projects",
      expect.objectContaining({
        headers: { "Content-Type": "application/json" }
      })
    )
  })

  it("opens task run diagnostics from enriched task detail without manual ID copying", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url === "http://127.0.0.1:8000/openapi.json") {
        return {
          ok: true,
          json: async () => ({
            paths: {
              "/api/v1/agent-orchestration/projects": {},
            }
          })
        }
      }
      if (url === "http://127.0.0.1:8000/api/v1/acp/health") {
        return {
          ok: true,
          json: async () => ({
            runner: "ok",
            agent: "ok",
            api_keys: "ok"
          })
        }
      }
      if (url === "http://127.0.0.1:8000/api/v1/agent-orchestration/projects") {
        return {
          ok: true,
          json: async () => [
            {
              id: 7,
              name: "Research Project",
              user_id: 1,
              created_at: "2026-03-20T19:00:00Z"
            }
          ]
        }
      }
      if (url === "http://127.0.0.1:8000/api/v1/agent-orchestration/projects/7/tasks") {
        return {
          ok: true,
          json: async () => [
            {
              id: 11,
              project_id: 7,
              title: "Draft spec",
              status: "triage",
              review_count: 1,
              max_review_attempts: 3,
              created_at: "2026-03-20T19:00:00Z",
              updated_at: "2026-03-20T19:00:00Z"
            }
          ]
        }
      }
      if (url === "http://127.0.0.1:8000/api/v1/agent-orchestration/tasks/11") {
        expect((init?.headers as Record<string, string>)?.["X-API-KEY"]).toBe("real-key")
        return {
          ok: true,
          json: async () => ({
            id: 11,
            project_id: 7,
            title: "Draft spec",
            status: "triage",
            review_count: 1,
            max_review_attempts: 3,
            created_at: "2026-03-20T19:00:00Z",
            updated_at: "2026-03-20T19:00:00Z",
            reviews: [
              {
                reviewer: "reviewer-agent",
                approved: false,
                feedback: "Needs citations",
                created_at: "2026-03-20T19:10:00Z"
              }
            ],
            runs: [
              {
                id: 51,
                task_id: 11,
                session_id: "sess-1",
                agent_type: "codex",
                status: "failed",
                error: "Workspace root not allowed",
                started_at: "2026-03-20T19:00:00Z",
                session: {
                  session_id: "sess-1",
                  available: true,
                  links: {
                    diagnostics: "/api/v1/acp/sessions/sess-1/diagnostics",
                    artifacts: "/api/v1/acp/sessions/sess-1/artifacts",
                    audit: "/api/v1/acp/sessions/sess-1/audit"
                  }
                },
                history: {
                  event_count: 3,
                  audit_event_count: 2,
                  artifact_count: 1,
                  diagnostic_count: 1,
                  tool_call_count: 4,
                  result: {
                    role: "assistant",
                    preview: "I could not access the workspace."
                  }
                },
                failure_context: {
                  reason_code: "workspace_root_not_allowed",
                  message: "Workspace root not allowed",
                  source: "session_diagnostic"
                },
                review_decision: {
                  available: true,
                  approved: false,
                  reviewer: "reviewer-agent",
                  feedback_preview: "Needs citations"
                }
              }
            ]
          })
        }
      }
      throw new Error(`unexpected fetch: ${url}`)
    })
    vi.stubGlobal("fetch", fetchMock)

    render(<AgentTasksPage />)

    const projectButton = await screen.findByText("Research Project")
    fireEvent.click(projectButton)
    expect(await screen.findByText("Draft spec")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: /inspect/i }))

    expect(await screen.findByText("Run #51")).toBeInTheDocument()
    expect(screen.getByText("sess-1")).toBeInTheDocument()
    expect(screen.getByText("workspace_root_not_allowed")).toBeInTheDocument()
    expect(screen.getByText("Workspace root not allowed")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /open diagnostics/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /open artifacts/i })).toBeInTheDocument()
    expect(screen.getByText("Needs citations")).toBeInTheDocument()
  })

  it("uses the route workspace filter to show only linked canonical workspace projects and tasks", async () => {
    window.location.hash = "#/agent-tasks?workspace=workspace-alpha"
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url === "http://127.0.0.1:8000/openapi.json") {
        return {
          ok: true,
          json: async () => ({
            paths: {
              "/api/v1/agent-orchestration/projects": {},
            }
          })
        }
      }
      if (url === "http://127.0.0.1:8000/api/v1/acp/health") {
        return {
          ok: true,
          json: async () => ({
            runner: "ok",
            agent: "ok",
            api_keys: "ok"
          })
        }
      }
      if (url === "http://127.0.0.1:8000/api/v1/agent-orchestration/projects") {
        return {
          ok: true,
          json: async () => [
            {
              id: 7,
              name: "Alpha Project",
              user_id: 1,
              created_at: "2026-03-20T19:00:00Z",
              canonical_workspace: {
                acp_workspace_id: 33,
                canonical_workspace_id: "workspace-alpha",
                canonical_workspace_source: "workspace_playground",
                link_status: "linked"
              },
              task_summary: {
                total_tasks: 1,
                status_counts: {
                  todo: 1
                }
              }
            },
            {
              id: 8,
              name: "Beta Project",
              user_id: 1,
              created_at: "2026-03-20T19:00:00Z",
              canonical_workspace: {
                acp_workspace_id: 34,
                canonical_workspace_id: "workspace-beta",
                canonical_workspace_source: "workspace_playground",
                link_status: "linked"
              },
              task_summary: {
                total_tasks: 1,
                status_counts: {
                  todo: 1
                }
              }
            }
          ]
        }
      }
      if (url === "http://127.0.0.1:8000/api/v1/agent-orchestration/projects/7/tasks") {
        return {
          ok: true,
          json: async () => [
            {
              id: 11,
              project_id: 7,
              title: "Alpha task",
              status: "todo",
              review_count: 0,
              max_review_attempts: 3,
              created_at: "2026-03-20T19:00:00Z",
              updated_at: "2026-03-20T19:00:00Z"
            }
          ]
        }
      }
      if (url === "http://127.0.0.1:8000/api/v1/agent-orchestration/projects/8/tasks") {
        throw new Error("filtered workspace should not request beta tasks")
      }
      throw new Error(`unexpected fetch: ${url}`)
    })
    vi.stubGlobal("fetch", fetchMock)

    render(<AgentTasksPage />)

    expect(await screen.findByText("Alpha Project")).toBeInTheDocument()
    expect(screen.queryByText("Beta Project")).toBeNull()
    expect(screen.getAllByText("Workspace: workspace-alpha").length).toBeGreaterThan(0)

    fireEvent.click(screen.getByText("Alpha Project"))

    expect(await screen.findByText("Alpha task")).toBeInTheDocument()
    expect(fetchMock).not.toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/v1/agent-orchestration/projects/8/tasks",
      expect.anything()
    )
  })

  it("surfaces workspace setup gaps when the selected canonical workspace has no linked ACP execution project", async () => {
    window.location.hash = "#/agent-tasks?workspace=workspace-missing"
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url === "http://127.0.0.1:8000/openapi.json") {
        return {
          ok: true,
          json: async () => ({
            paths: {
              "/api/v1/agent-orchestration/projects": {},
            }
          })
        }
      }
      if (url === "http://127.0.0.1:8000/api/v1/acp/health") {
        return {
          ok: true,
          json: async () => ({
            runner: "ok",
            agent: "ok",
            api_keys: "ok"
          })
        }
      }
      if (url === "http://127.0.0.1:8000/api/v1/agent-orchestration/projects") {
        return {
          ok: true,
          json: async () => []
        }
      }
      throw new Error(`unexpected fetch: ${url}`)
    })
    vi.stubGlobal("fetch", fetchMock)

    render(<AgentTasksPage />)

    expect(await screen.findByText("Workspace setup needs attention")).toBeInTheDocument()
    expect(
      screen.getByText("No ACP execution workspace is linked to workspace-missing")
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        "Create an agent task from WorkspacePlayground so the execution root, environment, and MCP readiness can be validated before dispatch."
      )
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /open workspaceplayground/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /open acp playground/i })).toBeInTheDocument()
  })
})
