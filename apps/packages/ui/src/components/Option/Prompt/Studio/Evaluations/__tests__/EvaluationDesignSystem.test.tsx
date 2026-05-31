import React from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { CreateEvaluationWizard } from "../CreateEvaluationWizard"
import { EvaluationDetailPanel } from "../EvaluationDetailPanel"

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
  createEvaluation: vi.fn(),
  getEvaluation: vi.fn(),
  listPrompts: vi.fn(),
  listTestCases: vi.fn()
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: ({ queryKey }: { queryKey: unknown[] }) => {
    if (queryKey.includes("evaluation")) {
      return {
        data: {
          data: {
            id: 401,
            project_id: 7,
            prompt_id: 55,
            test_case_ids: [88],
            name: "Citation quality check",
            description: "Checks grounded answer quality.",
            status: "completed",
            aggregate_metrics: {
              accuracy: 0.91,
              pass_rate: 0.88,
              f1: 0.84
            },
            config: {
              model_name: "gpt-4o-mini",
              temperature: 0.3,
              max_tokens: 2048
            },
            created_at: "2026-05-30T10:00:00Z",
            completed_at: "2026-05-30T10:02:00Z"
          }
        },
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
        getFieldsValue: () => ({}),
        getFieldValue: () => undefined,
        resetFields: vi.fn()
      }
    ]
  })
  const Descriptions = Object.assign(passthrough, {
    Item: ({
      children,
      label
    }: {
      children?: React.ReactNode
      label?: React.ReactNode
    }) => (
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
    Select: () => <select />,
    Skeleton: () => <div>Loading</div>,
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
    notification: {
      error: vi.fn(),
      success: vi.fn()
    }
  }
})

const expectDesignSystemAlert = (text: string) => {
  const nodes = screen.getAllByText(text, { exact: false })

  expect(
    nodes.some((node) => node.closest('[data-ds-component="Alert"]'))
  ).toBe(true)
}

const expectDesignSystemBadge = (text: string) => {
  const nodes = screen.getAllByText(text, { exact: false })

  expect(
    nodes.some((node) => node.closest('[data-ds-component="Badge"]'))
  ).toBe(true)
}

describe("Prompt Studio evaluation design-system states", () => {
  beforeEach(() => {
    promptStudioStore.wizardStep = "selectPrompt"
    vi.clearAllMocks()
    cleanup()
  })

  it("renders evaluation wizard guidance through the design-system Alert", () => {
    const expectedGuidanceByStep = [
      {
        step: "selectPrompt",
        text: "Select the prompt you want to evaluate"
      },
      {
        step: "selectTestCases",
        text: "Select which test cases to include"
      },
      {
        step: "configureModel",
        text: "Configure the model settings"
      },
      {
        step: "review",
        text: "Review your evaluation settings"
      }
    ]

    for (const { step, text } of expectedGuidanceByStep) {
      promptStudioStore.wizardStep = step

      const { unmount } = render(
        <CreateEvaluationWizard open projectId={7} onClose={vi.fn()} />
      )

      expectDesignSystemAlert(text)
      unmount()
    }
  })

  it("renders evaluation detail status through the design-system Badge", () => {
    render(
      <EvaluationDetailPanel open evaluationId={401} onClose={vi.fn()} />
    )

    expectDesignSystemBadge("Completed")
  })
})
