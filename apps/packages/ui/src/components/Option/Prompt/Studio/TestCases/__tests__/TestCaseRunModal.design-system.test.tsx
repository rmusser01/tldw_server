import React from "react"
import { fireEvent, render, screen, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { TestCaseRunModal } from "../TestCaseRunModal"

const runResults = vi.hoisted(() => ({
  results: [
    {
      test_case_id: 101,
      passed: true,
      output: "The answer cited the required source.",
      execution_time: 0.42
    },
    {
      test_case_id: 102,
      passed: false,
      output: "The answer omitted the required evidence.",
      execution_time: 0.67
    },
    {
      test_case_id: 103,
      passed: null,
      output: "",
      error: "Provider timed out",
      execution_time: 1.25
    }
  ]
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
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
      return _key
    }
  })
}))

vi.mock("@/services/prompt-studio", () => ({
  listPrompts: vi.fn(),
  runTestCases: vi.fn()
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: () => ({
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
  }),
  useMutation: (config: { onSuccess?: (response: unknown) => void }) => ({
    isPending: false,
    mutate: vi.fn(() =>
      config.onSuccess?.({
        data: {
          data: runResults.results
        }
      })
    )
  })
}))

vi.mock("antd", () => ({
  Alert: ({
    message,
    title,
    description
  }: {
    message?: React.ReactNode
    title?: React.ReactNode
    description?: React.ReactNode
  }) => (
    <div data-antd-component="Alert">
      {title ?? message}
      {description}
    </div>
  ),
  Modal: ({
    open,
    title,
    children
  }: {
    open?: boolean
    title?: React.ReactNode
    children?: React.ReactNode
  }) =>
    open ? (
      <section role="dialog">
        <h2>{title}</h2>
        {children}
      </section>
    ) : null,
  notification: {
    error: vi.fn(),
    success: vi.fn(),
    warning: vi.fn()
  },
  Select: ({
    onChange,
    options,
    placeholder,
    value
  }: {
    onChange?: (value: number) => void
    options?: Array<{ label: string; value: number }>
    placeholder?: string
    value?: number | null
  }) => (
    <select
      aria-label="Select Prompt to Run Against"
      value={value ?? ""}
      onChange={(event) => onChange?.(Number(event.currentTarget.value))}
    >
      <option value="">{placeholder}</option>
      {options?.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  ),
  Spin: () => <div data-testid="loading-spinner" />,
  Table: ({
    columns,
    dataSource
  }: {
    columns: Array<{
      key?: string
      dataIndex?: string
      render?: (value: unknown, record: (typeof runResults.results)[number]) => React.ReactNode
    }>
    dataSource: typeof runResults.results
  }) => (
    <table>
      <tbody>
        {dataSource.map((record) => (
          <tr key={record.test_case_id}>
            {columns.map((column, index) => {
              const value = column.dataIndex
                ? record[column.dataIndex as keyof typeof record]
                : undefined
              return (
                <td key={column.key ?? column.dataIndex ?? index}>
                  {column.render ? column.render(value, record) : String(value ?? "")}
                </td>
              )
            })}
          </tr>
        ))}
      </tbody>
    </table>
  ),
  Tag: ({ children }: { children?: React.ReactNode }) => (
    <span data-antd-component="Tag">{children}</span>
  )
}))

function renderModal() {
  return render(
    <TestCaseRunModal
      open
      projectId={7}
      testCaseIds={[101, 102, 103]}
      onClose={vi.fn()}
    />
  )
}

function expectClosestDsComponent(text: string | RegExp, component: string) {
  const node = screen.getByText(text)
  const element = node.closest(`[data-ds-component="${component}"]`)
  expect(element).not.toBeNull()
  return element as HTMLElement
}

describe("TestCaseRunModal design-system product states", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("renders the run guidance through the design-system Alert primitive", () => {
    renderModal()

    const alert = expectClosestDsComponent(
      "Run 3 test cases against a prompt to see the outputs.",
      "Alert"
    )

    expect(alert).toHaveAttribute("role", "status")
  })

  it("renders run result summary and status labels through design-system Badge", async () => {
    renderModal()

    fireEvent.change(screen.getByRole("combobox"), {
      target: { value: "55" }
    })
    fireEvent.click(screen.getByRole("button", { name: /Run Tests/i }))

    const summary = await screen.findByText("Results")
    const summaryRegion = summary.closest("div")
    expect(summaryRegion).not.toBeNull()

    expect(
      within(summaryRegion as HTMLElement)
        .getByText("1 passed")
        .closest('[data-ds-component="Badge"]')
    ).not.toBeNull()
    expect(
      within(summaryRegion as HTMLElement)
        .getByText("1 failed")
        .closest('[data-ds-component="Badge"]')
    ).not.toBeNull()

    expectClosestDsComponent("Pass", "Badge")
    expectClosestDsComponent("Fail", "Badge")
    expectClosestDsComponent("Error", "Badge")
  })
})
