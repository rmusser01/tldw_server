import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { CompareStrategiesModal } from "../CompareStrategiesModal"
import { CreateOptimizationWizard } from "../CreateOptimizationWizard"
import { OptimizationProgressPanel } from "../OptimizationProgressPanel"

const queryState = vi.hoisted(() => ({
  optimization: {
    id: 101,
    project_id: 7,
    prompt_id: 55,
    name: "Demo optimization",
    description: "Improves a reusable research prompt.",
    status: "completed",
    config: { strategy: "iterative", max_iterations: 3 },
    best_prompt_id: 56,
    best_score: 0.91,
    current_iteration: 3,
    total_iterations: 3,
    error_message: "Provider returned an invalid score.",
    cancel_reason: "Stopped by reviewer.",
    started_at: "2026-05-30T10:00:00Z",
    completed_at: "2026-05-30T10:03:00Z"
  }
}))

const promptStudioStore = vi.hoisted(() => ({
  wizardStep: "selectPrompt",
  setWizardStep: vi.fn((step: string) => {
    promptStudioStore.wizardStep = step
  }),
  resetWizard: vi.fn(() => {
    promptStudioStore.wizardStep = "selectPrompt"
  })
}))

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

vi.mock("@/store/prompt-studio", () => ({
  usePromptStudioStore: (selector: (state: typeof promptStudioStore) => unknown) =>
    selector(promptStudioStore)
}))

vi.mock("@/services/prompt-studio", () => ({
  createOptimization: vi.fn(),
  getOptimization: vi.fn(),
  getOptimizationIterations: vi.fn(),
  getOptimizationStrategies: vi.fn(),
  listPrompts: vi.fn(),
  listTestCases: vi.fn()
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: ({ queryKey }: { queryKey: unknown[] }) => {
    if (queryKey.includes("optimization-iterations")) {
      return {
        data: {
          data: {
            data: [
              {
                iteration: 3,
                prompt_id: 56,
                score: 0.91,
                changes: "Tightened evidence instructions.",
                timestamp: "2026-05-30T10:03:00Z"
              }
            ]
          }
        },
        isLoading: false
      }
    }

    if (queryKey.includes("optimization") && queryKey.includes(101)) {
      return {
        data: { data: { data: queryState.optimization } },
        isLoading: false
      }
    }

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

    if (queryKey.includes("test-cases")) {
      return {
        data: {
          data: {
            data: [
              {
                id: 88,
                project_id: 7,
                name: "Citation check",
                inputs: {},
                is_golden: true
              }
            ]
          }
        }
      }
    }

    return { data: undefined, isLoading: false }
  },
  useMutation: () => ({ mutate: vi.fn(), isPending: false }),
  useQueryClient: () => ({ invalidateQueries: vi.fn() })
}))

vi.mock("antd", () => {
  const passthrough = ({ children }: { children?: React.ReactNode }) => (
    <div>{children}</div>
  )
  const Form = Object.assign(passthrough, {
    Item: ({ children, label }: { children?: React.ReactNode; label?: React.ReactNode }) => (
      <label>
        {label}
        {children}
      </label>
    ),
    useForm: () => [
      {
        getFieldsValue: () => ({}),
        getFieldValue: () => undefined,
        resetFields: vi.fn()
      }
    ]
  })
  const Descriptions = Object.assign(passthrough, {
    Item: ({ children, label }: { children?: React.ReactNode; label?: React.ReactNode }) => (
      <div>
        <span>{label}</span>
        {children}
      </div>
    )
  })

  return {
    Alert: ({ message, title, description }: any) => (
      <div data-antd-component="Alert">
        {title ?? message}
        {description}
      </div>
    ),
    Card: passthrough,
    Checkbox: ({
      children,
      checked,
      onChange
    }: {
      children?: React.ReactNode
      checked?: boolean
      onChange?: (event: { target: { checked: boolean } }) => void
    }) => (
      <label>
        <input
          type="checkbox"
          checked={checked}
          onChange={(event) =>
            onChange?.({ target: { checked: event.currentTarget.checked } })
          }
        />
        {children}
      </label>
    ),
    Descriptions,
    Drawer: ({ open, children, title }: any) =>
      open ? (
        <section>
          <h2>{title}</h2>
          {children}
        </section>
      ) : null,
    Form,
    Input: (props: any) => <input {...props} />,
    InputNumber: (props: any) => <input type="number" {...props} />,
    Modal: ({ open, title, children, footer }: any) =>
      open ? (
        <section>
          <h2>{title}</h2>
          {children}
          {footer}
        </section>
      ) : null,
    Progress: ({ percent }: { percent?: number }) => (
      <div role="progressbar" aria-valuenow={percent} />
    ),
    Radio: passthrough,
    Select: () => <select />,
    Skeleton: () => <div>Loading</div>,
    Spin: () => <div>Loading</div>,
    Statistic: ({ title, value, suffix }: any) => (
      <div>
        {title}
        <span>
          {value}
          {suffix}
        </span>
      </div>
    ),
    Steps: () => <nav aria-label="Wizard steps" />,
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
    Tag: ({ children }: { children?: React.ReactNode }) => (
      <span data-antd-component="Tag">{children}</span>
    ),
    Timeline: ({ items = [] }: any) => (
      <div>{items.map((item: any, index: number) => <div key={index}>{item.children}</div>)}</div>
    ),
    Tooltip: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
    notification: {
      error: vi.fn(),
      success: vi.fn()
    }
  }
})

const expectDesignSystemAlert = (text: string | RegExp) => {
  const nodes =
    typeof text === "string"
      ? screen.getAllByText(text, { exact: false })
      : screen.getAllByText(text)

  expect(
    nodes.some((node) => node.closest('[data-ds-component="Alert"]'))
  ).toBe(true)
}

const expectDesignSystemBadge = (text: string | RegExp) => {
  const nodes =
    typeof text === "string"
      ? screen.getAllByText(text, { exact: false })
      : screen.getAllByText(text)

  expect(
    nodes.some((node) => node.closest('[data-ds-component="Badge"]'))
  ).toBe(true)
}

describe("Prompt Studio optimization design-system states", () => {
  beforeEach(() => {
    promptStudioStore.wizardStep = "selectPrompt"
    vi.clearAllMocks()
  })

  it("renders optimization wizard guidance through the design-system Alert", () => {
    render(
      <CreateOptimizationWizard open projectId={7} onClose={vi.fn()} />
    )

    expectDesignSystemAlert("Select the prompt you want to optimize")
  })

  it("renders strategy comparison guidance and parameter chips through design-system primitives", () => {
    render(
      <CompareStrategiesModal
        open
        selectedStrategy="iterative"
        onClose={vi.fn()}
        onSelectStrategy={vi.fn()}
      />
    )

    expectDesignSystemAlert("Compare different optimization strategies")
    expectDesignSystemBadge("max_iterations")
  })

  it("renders progress status and diagnostics through design-system primitives", () => {
    render(
      <OptimizationProgressPanel
        open
        optimizationId={101}
        onClose={vi.fn()}
      />
    )

    expectDesignSystemBadge("Completed")
    expectDesignSystemBadge("Iterative")
    expectDesignSystemAlert("Error")
    expectDesignSystemAlert("Cancelled")
    expectDesignSystemAlert("Best prompt found")
  })
})
