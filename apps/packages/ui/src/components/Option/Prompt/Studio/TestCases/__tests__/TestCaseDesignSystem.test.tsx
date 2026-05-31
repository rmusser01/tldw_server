import React from "react"
import { fireEvent, render, screen, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { TestCaseBulkPanel } from "../TestCaseBulkPanel"
import { TestCaseGenerateModal } from "../TestCaseGenerateModal"
import { TestCaseRunModal } from "../TestCaseRunModal"

const testRunResults = vi.hoisted(() => [
  {
    test_case_id: 88,
    passed: true,
    output: "Grounded answer with citations.",
    execution_time: 1.23
  },
  {
    test_case_id: 89,
    passed: false,
    output: "Ungrounded answer.",
    execution_time: 0.82
  },
  {
    test_case_id: 90,
    passed: null,
    error: "Provider timeout",
    execution_time: 0.45
  }
])

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string; [key: string]: unknown }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions?.defaultValue) {
        return Object.entries(fallbackOrOptions).reduce(
          (value, [name, replacement]) =>
            name === "defaultValue"
              ? value
              : value.replace(
                  new RegExp(`{{${name}}}`, "g"),
                  String(replacement)
                ),
          fallbackOrOptions.defaultValue
        )
      }
      return key
    }
  })
}))

vi.mock("@/components/Common/Button", () => ({
  Button: ({
    children,
    disabled,
    htmlType,
    onClick
  }: {
    children?: React.ReactNode
    disabled?: boolean
    htmlType?: "button" | "submit" | "reset"
    onClick?: () => void
  }) => (
    <button type={htmlType ?? "button"} disabled={disabled} onClick={onClick}>
      {children}
    </button>
  )
}))

vi.mock("@/services/prompt-studio", () => ({
  createBulkTestCases: vi.fn(),
  exportTestCases: vi.fn(),
  generateTestCases: vi.fn(),
  importTestCases: vi.fn(),
  listPrompts: vi.fn(),
  runTestCases: vi.fn()
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: ({ queryKey }: { queryKey: unknown[] }) => {
    if (queryKey.includes("prompts")) {
      return {
        data: {
          data: {
            data: [
              {
                id: 55,
                project_id: 7,
                name: "Research synthesis",
                version_number: 2
              }
            ]
          }
        }
      }
    }

    return { data: undefined, isLoading: false }
  },
  useMutation: (options: any = {}) => ({
    isPending: false,
    mutate: vi.fn((args) => {
      options.onSuccess?.({ data: { data: testRunResults } }, args)
    })
  }),
  useQueryClient: () => ({ invalidateQueries: vi.fn() })
}))

vi.mock("antd", () => {
  const passthrough = ({ children }: { children?: React.ReactNode }) => (
    <div>{children}</div>
  )
  const Form = Object.assign(
    ({
      children,
      onFinish
    }: {
      children?: React.ReactNode
      onFinish?: (values: Record<string, unknown>) => void
    }) => (
      <form
        onSubmit={(event) => {
          event.preventDefault()
          onFinish?.({})
        }}
      >
        {children}
      </form>
    ),
    {
      Item: ({
        children,
        label
      }: {
        children?: React.ReactNode
        label?: React.ReactNode
      }) => (
        <label>
          {label}
          {children}
        </label>
      ),
      useForm: () => [
        {
          resetFields: vi.fn()
        }
      ]
    }
  )

  const Select = ({
    onChange,
    options = [],
    placeholder,
    value
  }: {
    onChange?: (value: unknown) => void
    options?: Array<{ label: React.ReactNode; value: unknown }>
    placeholder?: string
    value?: unknown
  }) => (
    <select
      aria-label={placeholder ?? "Select"}
      value={value === null || value === undefined ? "" : String(value)}
      onChange={(event) => {
        const selected = options.find(
          (option) => String(option.value) === event.currentTarget.value
        )
        onChange?.(selected?.value ?? event.currentTarget.value)
      }}
    >
      <option value="">{placeholder ?? "Select"}</option>
      {options.map((option) => (
        <option key={String(option.value)} value={String(option.value)}>
          {option.label}
        </option>
      ))}
    </select>
  )

  const Upload = Object.assign(passthrough, {
    Dragger: passthrough
  })

  return {
    Alert: ({ message, title, description }: any) => (
      <div data-antd-component="Alert">
        {title ?? message}
        {description}
      </div>
    ),
    Drawer: ({ open, children, title }: any) =>
      open ? (
        <section>
          <h2>{title}</h2>
          {children}
        </section>
      ) : null,
    Form,
    InputNumber: (props: any) => <input type="number" {...props} />,
    Modal: ({ open, title, children, footer }: any) =>
      open ? (
        <section>
          <h2>{title}</h2>
          {children}
          {footer}
        </section>
      ) : null,
    Select,
    Spin: () => <div>Loading</div>,
    Table: ({ dataSource = [], columns = [], rowKey }: any) => (
      <table>
        <tbody>
          {dataSource.map((record: any, index: number) => (
            <tr key={record[rowKey] ?? index}>
              {columns.map((column: any) => (
                <td key={column.key ?? column.dataIndex}>
                  {column.render
                    ? column.render(record[column.dataIndex], record, index)
                    : record[column.dataIndex]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    ),
    Tabs: ({ items = [] }: any) => (
      <div>
        {items.map((item: any) => (
          <section key={item.key}>
            <h3>{item.label}</h3>
            {item.children}
          </section>
        ))}
      </div>
    ),
    Tag: ({ children }: { children?: React.ReactNode }) => (
      <span data-antd-component="Tag">{children}</span>
    ),
    Upload,
    notification: {
      error: vi.fn(),
      success: vi.fn(),
      warning: vi.fn()
    }
  }
})

const expectDesignSystemAlert = (text: string) => {
  const nodes = screen.getAllByText(text, { exact: false })

  expect(
    nodes.some((node) => node.closest('[data-ds-component="Alert"]'))
  ).toBe(true)
}

const expectDesignSystemBadge = (
  text: string | RegExp,
  scope: { getByText: typeof screen.getByText } = screen
) => {
  const node = scope.getByText(text)

  expect(node.closest('[data-ds-component="Badge"]')).toBeTruthy()
}

describe("Prompt Studio test-case design-system states", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("renders import and export guidance through design-system Alerts", () => {
    render(<TestCaseBulkPanel open projectId={7} onClose={vi.fn()} />)

    expectDesignSystemAlert("Export all test cases from this project")
    expectDesignSystemAlert("Import test cases from a JSON or CSV file")
  })

  it("renders generation and run guidance through design-system Alerts", () => {
    render(<TestCaseGenerateModal open projectId={7} onClose={vi.fn()} />)
    expectDesignSystemAlert("Use AI to automatically generate test cases")

    render(
      <TestCaseRunModal
        open
        projectId={7}
        testCaseIds={[88, 89, 90]}
        onClose={vi.fn()}
      />
    )
    expectDesignSystemAlert("Run 3 test cases against a prompt")
  })

  it("renders test run summary and row statuses through design-system Badges", () => {
    render(
      <TestCaseRunModal
        open
        projectId={7}
        testCaseIds={[88, 89, 90]}
        onClose={vi.fn()}
      />
    )

    fireEvent.change(screen.getByLabelText("Select a prompt..."), {
      target: { value: "55" }
    })
    fireEvent.click(screen.getByText("Run Tests"))

    expectDesignSystemBadge("1 passed")
    expectDesignSystemBadge("1 failed")

    const resultsTable = within(screen.getByRole("table"))
    expectDesignSystemBadge("Pass", resultsTable)
    expectDesignSystemBadge("Fail", resultsTable)
    expectDesignSystemBadge("Error", resultsTable)
  })
})
