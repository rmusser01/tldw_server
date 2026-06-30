import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"

import { PromptStudioPlaygroundPage } from "../PromptStudioPlaygroundPage"

const mocks = vi.hoisted(() => ({
  online: true,
  uxState: "connected_ok" as
    | "connected_ok"
    | "testing"
    | "configuring_url"
    | "configuring_auth"
    | "error_auth"
    | "error_unreachable"
    | "unconfigured",
  hasCompletedFirstRun: true,
  navigate: vi.fn()
}))

const serviceMocks = vi.hoisted(() => ({
  hasPromptStudio: vi.fn(),
  getPromptStudioDefaults: vi.fn()
}))

const i18nMocks = vi.hoisted(() => ({
  t: vi.fn((_key: string, fallback?: string) => fallback ?? _key)
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: i18nMocks.t
  })
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  )
  return {
    ...actual,
    useNavigate: () => mocks.navigate
  }
})

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => mocks.online
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionUxState: () => ({
    uxState: mocks.uxState,
    hasCompletedFirstRun: mocks.hasCompletedFirstRun
  })
}))

vi.mock("@/services/prompt-studio", () => ({
  hasPromptStudio: (...args: unknown[]) =>
    serviceMocks.hasPromptStudio(...args),
  listProjects: vi.fn(),
  createProject: vi.fn(),
  listPrompts: vi.fn(),
  createPrompt: vi.fn(),
  getPrompt: vi.fn(),
  updatePrompt: vi.fn(),
  getPromptHistory: vi.fn(),
  revertPrompt: vi.fn(),
  executePrompt: vi.fn(),
  listTestCases: vi.fn(),
  createTestCase: vi.fn(),
  createBulkTestCases: vi.fn(),
  listEvaluations: vi.fn(),
  createEvaluation: vi.fn(),
  getEvaluation: vi.fn()
}))

vi.mock("@/services/prompt-studio-settings", () => ({
  getPromptStudioDefaults: (...args: unknown[]) =>
    serviceMocks.getPromptStudioDefaults(...args)
}))

vi.mock("antd", () => {
  const formApi = {
    setFieldsValue: vi.fn(),
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
      type,
      title,
      message,
      description,
      action
    }: {
      type?: string
      title?: React.ReactNode
      message?: React.ReactNode
      description?: React.ReactNode
      action?: React.ReactNode
    }) => (
      <div data-testid={`alert-${type || "info"}`}>
        <div>{title ?? message}</div>
        {description ? <div>{description}</div> : null}
        {action}
      </div>
    ),
    Badge: ({ children, text }: { children?: React.ReactNode; text?: React.ReactNode }) => (
      <div>
        {children}
        {text}
      </div>
    ),
    Button: ({
      children,
      onClick
    }: {
      children?: React.ReactNode
      onClick?: () => void
    }) => (
      <button type="button" onClick={onClick}>
        {children}
      </button>
    ),
    Card: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
    Divider: () => <hr />,
    Empty: ({ description }: { description?: React.ReactNode }) => <div>{description}</div>,
    Form,
    Input: Object.assign(
      ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
      {
        TextArea: () => <textarea />
      }
    ),
    InputNumber: () => <input type="number" />,
    Modal: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
    Select: () => <select aria-label="select" />,
    Skeleton: () => <div>Loading...</div>,
    Space: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
    Switch: () => <input type="checkbox" />,
    Table: () => <div>table</div>,
    Tabs: () => <div>tabs</div>,
    Tag: ({ children }: { children?: React.ReactNode }) => <span>{children}</span>,
    Typography: {
      Title: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
      Paragraph: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
      Text: ({ children }: { children?: React.ReactNode }) => <span>{children}</span>
    }
  }
})

const renderPage = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false
      }
    }
  })

  return render(
    <MemoryRouter>
      <QueryClientProvider client={queryClient}>
        <PromptStudioPlaygroundPage />
      </QueryClientProvider>
    </MemoryRouter>
  )
}

const expectDesignSystemAlertForText = (text: string) => {
  const alertText = screen.getByText(text)
  expect(alertText.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
}

const expectTranslation = (key: string, fallback: string) => {
  expect(i18nMocks.t).toHaveBeenCalledWith(key, fallback)
}

describe("PromptStudioPlaygroundPage connection states", () => {
  beforeEach(() => {
    mocks.online = true
    mocks.uxState = "connected_ok"
    mocks.hasCompletedFirstRun = true
    mocks.navigate.mockReset()
    serviceMocks.hasPromptStudio.mockReset()
    serviceMocks.getPromptStudioDefaults.mockReset()
    i18nMocks.t.mockClear()
    serviceMocks.hasPromptStudio.mockResolvedValue(true)
    serviceMocks.getPromptStudioDefaults.mockResolvedValue({
      pageSize: 10,
      defaultProjectId: null
    })
  })

  it("shows auth guidance and opens server settings when credentials are missing", () => {
    mocks.online = false
    mocks.uxState = "error_auth"

    renderPage()

    expect(
      screen.getByText("Add your credentials to use Prompt Studio")
    ).toBeInTheDocument()
    expectDesignSystemAlertForText("Add your credentials to use Prompt Studio")
    expectTranslation("option:promptStudio.openSettings", "Open Settings")
    expectTranslation(
      "option:promptStudio.addCredentialsDesc",
      "Prompt Studio needs a reachable tldw server plus valid credentials before projects, prompts, and evaluations can load."
    )
    expect(
      screen.queryByText("Connect to your server to use Prompt Studio")
    ).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Open Settings" }))
    expect(mocks.navigate).toHaveBeenCalledWith("/settings/tldw")
  })

  it("shows setup guidance and routes first-run users to setup", () => {
    mocks.online = false
    mocks.uxState = "unconfigured"
    mocks.hasCompletedFirstRun = false

    renderPage()

    expect(
      screen.getByText("Finish setup to use Prompt Studio")
    ).toBeInTheDocument()
    expectDesignSystemAlertForText("Finish setup to use Prompt Studio")
    expectTranslation("option:promptStudio.finishSetupAction", "Finish Setup")
    expectTranslation(
      "option:promptStudio.finishSetupDesc",
      "Prompt Studio depends on a configured tldw server before projects, prompts, and evaluations can load."
    )

    fireEvent.click(screen.getByRole("button", { name: "Finish Setup" }))
    expect(mocks.navigate).toHaveBeenCalledWith("/")
  })

  it("shows diagnostics guidance when the server is unreachable", () => {
    mocks.online = false
    mocks.uxState = "error_unreachable"

    renderPage()

    expect(
      screen.getByText("Can't reach your tldw server right now")
    ).toBeInTheDocument()
    expectDesignSystemAlertForText("Can't reach your tldw server right now")
    expectTranslation(
      "option:promptStudio.healthDiagnostics",
      "Health & diagnostics"
    )
    expectTranslation("option:promptStudio.openSettings", "Open Settings")
    expectTranslation(
      "option:promptStudio.unreachableDesc",
      "Prompt Studio depends on a reachable tldw server. Review your server status and URL before trying again."
    )

    fireEvent.click(screen.getByRole("button", { name: "Health & diagnostics" }))
    expect(mocks.navigate).toHaveBeenCalledWith("/settings/health")
  })

  it("keeps testing states on the loading path instead of showing offline guidance", () => {
    mocks.online = false
    mocks.uxState = "testing"
    serviceMocks.hasPromptStudio.mockImplementation(
      () => new Promise(() => undefined)
    )
    serviceMocks.getPromptStudioDefaults.mockImplementation(
      () => new Promise(() => undefined)
    )

    renderPage()

    expect(screen.getByText("Loading...")).toBeInTheDocument()
    expect(
      screen.queryByText("Connect to your server to use Prompt Studio")
    ).not.toBeInTheDocument()
  })
})
