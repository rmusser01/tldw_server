import React from "react"
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor
} from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { WorkspaceAgentTaskHandoffModal } from "../WorkspaceAgentTaskHandoffModal"

const connection = vi.hoisted(() => ({
  config: {
    serverUrl: "https://backend.example.test",
    authMode: "single-user" as "single-user" | "multi-user",
    authSource: "cookie-session" as "cookie-session" | "manual",
    apiKey: "stale-key",
    accessToken: "stale-token"
  }
}))
vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => ({
    config: connection.config,
    loading: false
  })
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, value: unknown) =>
      typeof value === "string"
        ? value
        : (value as { defaultValue?: string })?.defaultValue || _key
  })
}))
vi.mock("antd", () => ({
  Modal: ({
    open,
    children,
    footer
  }: {
    open: boolean
    children: React.ReactNode
    footer: React.ReactNode
  }) =>
    open ? (
      <div>
        {children}
        {footer}
      </div>
    ) : null,
  Button: ({
    children,
    onClick,
    disabled
  }: {
    children: React.ReactNode
    onClick?: () => void
    disabled?: boolean
  }) => (
    <button onClick={onClick} disabled={disabled}>
      {children}
    </button>
  ),
  Input: Object.assign(
    (props: React.InputHTMLAttributes<HTMLInputElement>) => (
      <input {...props} />
    ),
    {
      TextArea: ({
        autoSize: _autoSize,
        ...props
      }: React.TextareaHTMLAttributes<HTMLTextAreaElement> & {
        autoSize?: unknown
      }) => <textarea {...props} />
    }
  )
}))
vi.mock("@/components/ui", () => ({
  Alert: ({
    children,
    title
  }: {
    children: React.ReactNode
    title?: React.ReactNode
  }) => (
    <div>
      {title}
      {children}
    </div>
  )
}))

const csrf = "current-handoff-csrf"
const submit = () => {
  render(
    <WorkspaceAgentTaskHandoffModal
      open
      workspaceId="canonical-a"
      workspaceName="C04"
      onCancel={vi.fn()}
      onOpenAgentTasks={vi.fn()}
    />
  )
  fireEvent.change(screen.getByLabelText("Execution root path"), {
    target: { value: "/workspace/public-repo" }
  })
  fireEvent.change(screen.getByLabelText("Task title"), {
    target: { value: "Summarize README" }
  })
  fireEvent.change(screen.getByLabelText("Task description"), {
    target: { value: "Read README; do not write files." }
  })
  fireEvent.change(screen.getByLabelText("Agent type"), {
    target: { value: "opencode" }
  })
  fireEvent.click(screen.getByRole("button", { name: "Create task" }))
}
const response = (value: unknown, ok = true) => ({
  ok,
  status: ok ? 200 : 403,
  json: async () => value
})

describe("workspace handoff request authentication", () => {
  beforeEach(() => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    connection.config.authMode = "single-user"
    connection.config.authSource = "cookie-session"
    document.cookie = `csrf_token=${csrf}; Path=/`
  })
  afterEach(() => {
    cleanup()
    document.cookie = "csrf_token=; Max-Age=0; Path=/"
    vi.unstubAllEnvs()
    vi.unstubAllGlobals()
  })

  it("creates every handoff row with current single-user cookie CSRF", async () => {
    const requests: Array<[string, RequestInit]> = []
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init: RequestInit) => {
        requests.push([url, init])
        const headers = new Headers(init.headers)
        if (headers.get("X-CSRF-Token") !== csrf)
          return response({ detail: "CSRF token validation failed" }, false)
        if (url.endsWith("canonical-bridge")) return response({ id: 11 })
        if (url.endsWith("/projects")) return response({ id: 22 })
        return response({ id: 33 })
      })
    )
    submit()
    await waitFor(() =>
      expect(screen.getByText("Task #33")).toBeInTheDocument()
    )
    expect(requests).toHaveLength(3)
    for (const [url, init] of requests) {
      expect(url.startsWith("/api/v1/agent-orchestration/")).toBe(true)
      expect(init.credentials).toBe("same-origin")
      const headers = new Headers(init.headers)
      expect(headers.get("X-CSRF-Token")).toBe(csrf)
      expect(headers.get("Authorization")).toBeNull()
      expect(headers.get("X-API-KEY")).toBeNull()
    }
    expect(JSON.parse(String(requests[1][1].body)).workspace_id).toBe(11)
    expect(JSON.parse(String(requests[2][1].body)).agent_type).toBe("opencode")
  })

  it("uses current cookie CSRF for rollback DELETE after task creation fails", async () => {
    const requests: Array<[string, RequestInit]> = []
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init: RequestInit) => {
        requests.push([url, init])
        if (url.endsWith("canonical-bridge")) return response({ id: 11 })
        if (url.endsWith("/projects")) return response({ id: 22 })
        if (init.method === "DELETE") return response({})
        return response({ detail: "Task admission refused" }, false)
      })
    )
    submit()
    await waitFor(() =>
      expect(screen.getByText("Task admission refused")).toBeInTheDocument()
    )
    expect(requests).toHaveLength(4)
    const [url, init] = requests[3]
    expect(url).toBe("/api/v1/agent-orchestration/projects/22")
    expect(init.method).toBe("DELETE")
    expect(init.credentials).toBe("same-origin")
    expect(new Headers(init.headers).get("X-CSRF-Token")).toBe(csrf)
  })

  it.each(["single-user", "multi-user"] as const)(
    "preserves direct %s authentication",
    async (mode) => {
      vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "")
      connection.config.authMode = mode
      connection.config.authSource = "manual"
      const fetchMock = vi.fn(async (url: string) =>
        response({
          id: url.endsWith("canonical-bridge")
            ? 11
            : url.endsWith("/projects")
              ? 22
              : 33
        })
      )
      vi.stubGlobal("fetch", fetchMock)
      submit()
      await waitFor(() =>
        expect(screen.getByText("Task #33")).toBeInTheDocument()
      )
      const [url, init] = fetchMock.mock.calls[0] as unknown as [
        string,
        RequestInit
      ]
      expect(url).toBe(
        "https://backend.example.test/api/v1/agent-orchestration/workspaces/canonical-bridge"
      )
      const headers = new Headers(init.headers)
      expect(
        headers.get(mode === "single-user" ? "X-API-KEY" : "Authorization")
      ).toBe(mode === "single-user" ? "stale-key" : "Bearer stale-token")
      expect(headers.get("X-CSRF-Token")).toBeNull()
    }
  )

  it("preserves multi-user JWT authentication through the quickstart same-origin route", async () => {
    connection.config.authMode = "multi-user"
    connection.config.authSource = "manual"
    const fetchMock = vi.fn(async (url: string) =>
      response({
        id: url.endsWith("canonical-bridge")
          ? 11
          : url.endsWith("/projects")
            ? 22
            : 33
      })
    )
    vi.stubGlobal("fetch", fetchMock)
    submit()
    await waitFor(() =>
      expect(screen.getByText("Task #33")).toBeInTheDocument()
    )
    const [url, init] = fetchMock.mock.calls[0] as unknown as [
      string,
      RequestInit
    ]
    expect(url).toBe("/api/v1/agent-orchestration/workspaces/canonical-bridge")
    expect(new Headers(init.headers).get("Authorization")).toBe(
      "Bearer stale-token"
    )
    expect(new Headers(init.headers).get("X-API-KEY")).toBeNull()
    expect(new Headers(init.headers).get("X-CSRF-Token")).toBeNull()
  })

  it("keeps hosted mutations on the proxy without browser credentials", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "hosted")
    const fetchMock = vi.fn(async (url: string) =>
      response({
        id: url.endsWith("canonical-bridge")
          ? 11
          : url.endsWith("/projects")
            ? 22
            : 33
      })
    )
    vi.stubGlobal("fetch", fetchMock)
    submit()
    await waitFor(() =>
      expect(screen.getByText("Task #33")).toBeInTheDocument()
    )
    const [url, init] = fetchMock.mock.calls[0] as unknown as [
      string,
      RequestInit
    ]
    expect(url).toContain("/api/")
    expect(new Headers(init.headers).get("Authorization")).toBeNull()
    expect(new Headers(init.headers).get("X-API-KEY")).toBeNull()
    expect(new Headers(init.headers).get("X-CSRF-Token")).toBeNull()
  })
})
