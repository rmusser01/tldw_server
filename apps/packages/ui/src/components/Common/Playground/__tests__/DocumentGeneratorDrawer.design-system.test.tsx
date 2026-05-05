// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { DocumentGeneratorDrawer } from "../DocumentGeneratorDrawer"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue ?? _key
  })
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: () => ({ data: [] })
}))

vi.mock("@/services/tldw-server", () => ({
  fetchChatModels: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn()
  }
}))

vi.mock("@/store/ui-mode", () => ({
  useUiModeStore: (selector: (state: { mode: string }) => string) =>
    selector({ mode: "standard" })
}))

const capabilitiesState = vi.hoisted(() => ({
  hasChatDocuments: false
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: capabilitiesState
  })
}))

vi.mock("antd", () => {
  const FormComponent: React.FC<{ children?: React.ReactNode }> & {
    useForm: () => Array<{
      setFieldsValue: ReturnType<typeof vi.fn>
      validateFields: ReturnType<typeof vi.fn>
    }>
    Item: React.FC<{ children?: React.ReactNode }>
  } = ({ children }) => (
    <form>{children}</form>
  )
  FormComponent.useForm = () => [
    {
      setFieldsValue: vi.fn(),
      validateFields: vi.fn()
    }
  ]
  FormComponent.Item = ({ children }) => <div>{children}</div>

  const InputComponent: React.FC<React.InputHTMLAttributes<HTMLInputElement>> & {
    TextArea: React.FC<React.TextareaHTMLAttributes<HTMLTextAreaElement>>
  } = (props) => <input {...props} />
  InputComponent.TextArea = (props) => <textarea {...props} />

  return {
    Alert: ({ title }: { title?: React.ReactNode }) => <div>{title}</div>,
    Button: ({
      children,
      loading: _loading,
      ...props
    }: React.ButtonHTMLAttributes<HTMLButtonElement> & { loading?: boolean }) => (
      <button type="button" {...props}>
        {children}
      </button>
    ),
    Drawer: ({
      children,
      open
    }: {
      children?: React.ReactNode
      open?: boolean
    }) => (open ? <div>{children}</div> : null),
    Form: FormComponent,
    Input: InputComponent,
    InputNumber: (props: React.InputHTMLAttributes<HTMLInputElement>) => (
      <input {...props} />
    ),
    Modal: Object.assign(
      ({
        children,
        open
      }: {
        children?: React.ReactNode
        open?: boolean
      }) => (open ? <div>{children}</div> : null),
      { confirm: vi.fn() }
    ),
    Select: () => <select />,
    Switch: () => <input type="checkbox" />,
    Tag: ({ children }: { children?: React.ReactNode }) => <span>{children}</span>,
    Tooltip: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
    message: {
      error: vi.fn(),
      success: vi.fn()
    }
  }
})

vi.mock("../Markdown", () => ({
  default: () => null
}))

describe("DocumentGeneratorDrawer design-system recovery alerts", () => {
  it("renders availability recovery and drawer actions through shared primitives", () => {
    capabilitiesState.hasChatDocuments = false

    render(
      <DocumentGeneratorDrawer
        open
        onClose={() => undefined}
        conversationId={null}
      />
    )

    expect(screen.getByTestId("document-generator-capability-alert")).toHaveAttribute(
      "data-ds-component",
      "Alert"
    )
    expect(screen.getByTestId("document-generator-capability-alert")).toHaveAttribute(
      "role",
      "status"
    )
    expect(screen.getByTestId("document-generator-capability-alert")).toHaveAttribute(
      "aria-live",
      "polite"
    )
    expect(
      screen.getByText("Document generation is not available on this server.")
    ).toBeInTheDocument()
    expect(screen.getByTestId("document-generator-conversation-alert")).toHaveAttribute(
      "data-ds-component",
      "Alert"
    )
    expect(screen.getByTestId("document-generator-conversation-alert")).toHaveAttribute(
      "role",
      "status"
    )
    expect(screen.getByTestId("document-generator-conversation-alert")).toHaveAttribute(
      "aria-live",
      "polite"
    )
    expect(
      screen.getByText("Start a server-backed chat to generate documents.")
    ).toBeInTheDocument()

    const footer = screen.getByTestId("document-generator-drawer-footer")
    expect(footer).toHaveAttribute("data-ds-component", "ModalFooter")
    expect(screen.getByRole("button", { name: "Generate" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Refresh" })).toBeDisabled()
  })
})
