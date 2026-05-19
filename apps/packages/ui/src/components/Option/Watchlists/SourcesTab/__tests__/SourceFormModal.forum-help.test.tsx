import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { SourceFormModal } from "../SourceFormModal"

const formApi = {
  setFieldsValue: vi.fn(),
  resetFields: vi.fn(),
  validateFields: vi.fn()
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: unknown) =>
      typeof defaultValue === "string" ? defaultValue : _key
  })
}))

vi.mock("antd", () => {
  const FormComponent = ({ children }: any) => <form>{children}</form>
  FormComponent.Item = ({ label, extra, children }: any) => (
    <div>
      {label ? <label>{label}</label> : null}
      {extra ? <div>{extra}</div> : null}
      {children}
    </div>
  )
  FormComponent.useForm = () => [formApi]

  return {
    Form: FormComponent,
    Input: ({ placeholder }: any) => <input placeholder={placeholder} />,
    Modal: ({ open, title, children }: any) => (open ? <div><h2>{title}</h2>{children}</div> : null),
    Alert: ({ title, message, description }: any) => (
      <div>
        {title ?? message ? <span>{title ?? message}</span> : null}
        {description ? <span>{description}</span> : null}
      </div>
    ),
    Button: ({ children }: any) => <button type="button">{children}</button>,
    message: {
      info: vi.fn(),
      success: vi.fn(),
      warning: vi.fn(),
      error: vi.fn()
    },
    Select: ({ options = [] }: any) => (
      <div>
        {options.map((option: any) => (
          <span key={String(option.value)}>{String(option.label)}</span>
        ))}
      </div>
    )
  }
})

vi.mock("@/services/watchlists", () => ({
  testWatchlistSource: vi.fn(),
  testWatchlistSourceDraft: vi.fn()
}))

describe("SourceFormModal forum type guidance", () => {
  it("shows explicit coming-soon explanation for disabled forum type", () => {
    render(
      <SourceFormModal
        open
        onClose={vi.fn()}
        onSubmit={vi.fn()}
        existingTags={[]}
      />
    )

    expect(
      screen.getByText("Forum monitoring is coming soon. Use RSS Feed or Website for now.")
    ).toBeInTheDocument()
    expect(screen.getByText("Forum (coming soon)")).toBeInTheDocument()
  })

  it("enables forum source option when the backend capability is enabled", () => {
    render(
      <SourceFormModal
        open
        forumsEnabled
        onClose={vi.fn()}
        onSubmit={vi.fn()}
        existingTags={[]}
      />
    )

    expect(
      screen.queryByText("Forum monitoring is coming soon. Use RSS Feed or Website for now.")
    ).not.toBeInTheDocument()
    expect(screen.queryByText("Forum (coming soon)")).not.toBeInTheDocument()
    expect(screen.getByText("Forum")).toBeInTheDocument()
  })
})
