import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { TemplateEditor } from "../TemplateEditor"

const serviceMocks = vi.hoisted(() => ({
  createWatchlistTemplate: vi.fn(),
  fetchWatchlistRuns: vi.fn(),
  getWatchlistTemplate: vi.fn(),
  getWatchlistTemplateVersions: vi.fn(),
  validateWatchlistTemplate: vi.fn()
}))

const i18nMocks = vi.hoisted(() => ({
  t: (_key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
    if (typeof defaultValue !== "string") return _key
    if (!options) return defaultValue
    return defaultValue.replace(/\{\{(\w+)\}\}/g, (_, token) => String(options[token] ?? ""))
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: i18nMocks.t
  })
}))

vi.mock("antd", () => {
  const createFormState = () => {
    let fields: Record<string, unknown> = {}
    return {
      setFieldsValue: (values: Record<string, unknown>) => {
        fields = { ...fields, ...values }
      },
      getFieldValue: (name: string) => fields[name],
      resetFields: () => {
        fields = {}
      },
      validateFields: async () => fields,
      isFieldTouched: () => false,
      setFields: () => {}
    }
  }

  const Form = ({ children }: { children: React.ReactNode }) => <form>{children}</form>
  Form.Item = ({ children }: { children: React.ReactNode }) => <div>{children}</div>
  Form.useForm = () => {
    const ref = React.useRef<any>(null)
    if (!ref.current) {
      ref.current = createFormState()
    }
    return [ref.current]
  }
  Form.useWatch = (name: string, form: any) => form?.getFieldValue?.(name)

  const RadioButton = ({
    value,
    children,
    groupValue,
    onGroupChange,
    ...rest
  }: any) => (
    <button
      type="button"
      role="radio"
      aria-checked={groupValue === value}
      onClick={() => onGroupChange?.({ target: { value } })}
      {...rest}
    >
      {children}
    </button>
  )

  const Radio: any = ({ children }: { children: React.ReactNode }) => <span>{children}</span>
  Radio.Group = ({ value, onChange, children }: any) => (
    <div role="radiogroup">
      {React.Children.map(children, (child: any) =>
        React.cloneElement(child, {
          groupValue: value,
          onGroupChange: onChange
        })
      )}
    </div>
  )
  Radio.Button = RadioButton

  const Input: any = ({ value, onChange, ...rest }: any) => (
    <input
      value={(value as string) || ""}
      onChange={onChange}
      readOnly={typeof onChange !== "function"}
      {...rest}
    />
  )
  Input.TextArea = ({ value, onChange, ...rest }: any) => (
    <textarea
      value={(value as string) || ""}
      onChange={onChange}
      readOnly={typeof onChange !== "function"}
      {...rest}
    />
  )

  const Select = ({ value, onChange, options = [], allowClear: _allowClear, ...rest }: any) => (
    <select
      value={value == null ? "" : String(value)}
      onChange={(event) => onChange?.(event.currentTarget.value || undefined)}
      {...rest}
    >
      <option value="" />
      {options.map((option: any) => (
        <option key={String(option.value)} value={String(option.value)}>
          {String(option.label)}
        </option>
      ))}
    </select>
  )

  const Tabs = ({ items = [], activeKey, onChange }: any) => {
    const resolvedActiveKey = activeKey || items[0]?.key
    const activeItem = items.find((item: any) => item.key === resolvedActiveKey) || items[0]
    return (
      <div>
        <div>
          {items.map((item: any) => (
            <button
              key={item.key}
              type="button"
              role="tab"
              onClick={() => onChange?.(item.key)}
            >
              {item.label}
            </button>
          ))}
        </div>
        <div>{activeItem?.children}</div>
      </div>
    )
  }

  return {
    Alert: ({ title, description }: any) => (
      <div>
        {title}
        {description}
      </div>
    ),
    Button: ({ children, onClick, loading, danger: _danger, ...rest }: any) => (
      <button type="button" disabled={Boolean(loading)} onClick={() => onClick?.()} {...rest}>
        {children}
      </button>
    ),
    Checkbox: ({ checked, onChange, children }: any) => (
      <label>
        <input
          type="checkbox"
          checked={Boolean(checked)}
          onChange={(event) => onChange?.({ target: { checked: event.currentTarget.checked } })}
        />
        {children}
      </label>
    ),
    Divider: () => <hr />,
    Form,
    Input,
    Modal: ({ open, children, title }: any) => (open ? <div>{title}{children}</div> : null),
    Radio,
    Select,
    Space: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
    Spin: () => <span>loading</span>,
    Tabs,
    message: {
      success: vi.fn(),
      error: vi.fn(),
      info: vi.fn()
    }
  }
})

vi.mock("@/services/watchlists", () => ({
  createWatchlistTemplate: (...args: unknown[]) => serviceMocks.createWatchlistTemplate(...args),
  fetchWatchlistRuns: (...args: unknown[]) => serviceMocks.fetchWatchlistRuns(...args),
  getWatchlistTemplate: (...args: unknown[]) => serviceMocks.getWatchlistTemplate(...args),
  getWatchlistTemplateVersions: (...args: unknown[]) => serviceMocks.getWatchlistTemplateVersions(...args),
  validateWatchlistTemplate: (...args: unknown[]) => serviceMocks.validateWatchlistTemplate(...args)
}))

vi.mock("../TemplateCodeEditor", () => ({
  TemplateCodeEditor: React.forwardRef<HTMLTextAreaElement, { value: string; onChange: (value: string) => void }>(
    ({ value, onChange }, ref) => (
      <textarea
        ref={ref}
        data-testid="template-code-editor"
        value={value}
        onChange={(event) => onChange(event.currentTarget.value)}
      />
    )
  )
}))

vi.mock("../../shared", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../shared")>()
  return {
    ...actual,
    WatchlistsHelpTooltip: () => null,
    useWatchlistsViewport: () => ({ breakpoint: 768, isConstrained: false })
  }
})

vi.mock("@/utils/watchlists-prevention-telemetry", () => ({
  trackWatchlistsPreventionTelemetry: vi.fn()
}))

describe("TemplateEditor visual mode", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    serviceMocks.fetchWatchlistRuns.mockResolvedValue({ items: [] })
    serviceMocks.validateWatchlistTemplate.mockResolvedValue({ valid: true, errors: [] })
    serviceMocks.createWatchlistTemplate.mockResolvedValue({})
  })

  it("renders Visual mode and core block controls", async () => {
    render(<TemplateEditor open template={null} onClose={vi.fn()} />)
    expect(await screen.findByText("Visual")).toBeInTheDocument()
    expect(screen.getByText("Add Header")).toBeInTheDocument()
    expect(screen.getByText("Add Item Loop")).toBeInTheDocument()
  })
})
