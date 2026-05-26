import { describe, it, expect, vi, afterEach } from "vitest"
import { render, screen, fireEvent, cleanup } from "@testing-library/react"
import NodeConfigPanel from "../NodeConfigPanel"
import { useWorkflowEditorStore } from "@/store/workflow-editor"
import { buildStepRegistry } from "../step-registry"
import type { WorkflowStepSchema } from "@/types/workflow-editor"
import { useWorkflowDynamicOptions } from "../dynamic-options"

vi.mock("../dynamic-options", () => ({
  useWorkflowDynamicOptions: vi.fn()
}))

vi.mock("antd", async () => {
  const actual = await vi.importActual("antd")

  const FormItem = ({ label, children }: any) => (
    <div>
      {label ? <label>{label}</label> : null}
      {children}
    </div>
  )
  const Form = ({ children, ...rest }: any) => <div {...rest}>{children}</div>
  Form.Item = FormItem

  return {
    ...actual,
    Form,
    Select: ({ value, options = [], onChange, loading, notFoundContent, placeholder, ...rest }: any) => (
      <div data-testid="mock-select">
        <div
          data-testid="mock-select-trigger"
          role="combobox"
          onMouseDown={() => {}}
        >
          {loading && (!options || options.length === 0) ? notFoundContent : null}
        </div>
        <div data-testid="mock-select-options">
          {(options || []).map((opt: any) => (
            <div key={opt.value} role="option" onClick={() => onChange?.(opt.value)}>
              {opt.label}
            </div>
          ))}
        </div>
      </div>
    )
  }
})

const setupStore = (
  stepType: string,
  schema: WorkflowStepSchema,
  config: Record<string, unknown> = {}
) => {
  useWorkflowEditorStore.setState({
    nodes: [
      {
        id: "node-1",
        type: stepType,
        position: { x: 0, y: 0 },
        data: {
          label: stepType,
          stepType,
          config
        }
      }
    ],
    edges: [],
    selectedNodeIds: ["node-1"],
    stepRegistry: buildStepRegistry([]),
    stepSchemas: { [stepType]: schema }
  })
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  useWorkflowEditorStore.setState({
    nodes: [],
    edges: [],
    selectedNodeIds: [],
    stepSchemas: {},
    stepRegistry: buildStepRegistry([])
  })
})

describe("NodeConfigPanel selectors", () => {
  it("shows loading state for dynamic select options", async () => {
    const schema: WorkflowStepSchema = {
      type: "object",
      properties: {
        model: { type: "string" }
      }
    }
    setupStore("llm", schema)

    vi.mocked(useWorkflowDynamicOptions).mockReturnValue({
      optionsByKey: {},
      loadingByKey: { model: true }
    })

    render(<NodeConfigPanel />)
    const trigger = screen.getByTestId("mock-select-trigger")
    expect(trigger).not.toBeNull()
    fireEvent.mouseDown(trigger)

    expect(await screen.findByText("Loading options...")).toBeInTheDocument()
  })

  it("renders provided options in the select dropdown", async () => {
    const schema: WorkflowStepSchema = {
      type: "object",
      properties: {
        model: { type: "string" }
      }
    }
    setupStore("llm", schema)

    vi.mocked(useWorkflowDynamicOptions).mockReturnValue({
      optionsByKey: {
        model: [
          { value: "gpt-4", label: "GPT-4" },
          { value: "gpt-4o-mini", label: "GPT-4o Mini" }
        ]
      },
      loadingByKey: {}
    })

    render(<NodeConfigPanel />)
    const trigger = screen.getByTestId("mock-select-trigger")
    expect(trigger).not.toBeNull()
    fireEvent.mouseDown(trigger)

    expect(await screen.findByText("GPT-4")).toBeInTheDocument()
    expect(await screen.findByText("GPT-4o Mini")).toBeInTheDocument()
  })

  it("exposes aria labels for icon-only node actions", () => {
    const schema: WorkflowStepSchema = {
      type: "object",
      properties: {
        model: { type: "string" }
      }
    }
    setupStore("llm", schema)

    vi.mocked(useWorkflowDynamicOptions).mockReturnValue({
      optionsByKey: {},
      loadingByKey: {}
    })

    render(<NodeConfigPanel />)

    expect(
      screen.getByRole("button", { name: "Duplicate node" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Delete node" })
    ).toBeInTheDocument()
  })

  it("merges deep research metadata with server schema fields", () => {
    const schema: WorkflowStepSchema = {
      type: "object",
      properties: {
        query: { type: "string", description: "Templated research query resolved from workflow context" },
        source_policy: {
          type: "string",
          enum: ["balanced", "local_first", "external_first", "local_only", "external_only"]
        },
        autonomy_mode: {
          type: "string",
          enum: ["checkpointed", "autonomous"]
        },
        save_artifact: { type: "boolean", default: true }
      },
      required: ["query"]
    }
    setupStore("deep_research", schema, {
      query: "{{ inputs.topic }}",
      source_policy: "balanced",
      autonomy_mode: "checkpointed",
      save_artifact: true
    })

    vi.mocked(useWorkflowDynamicOptions).mockReturnValue({
      optionsByKey: {},
      loadingByKey: {}
    })

    render(<NodeConfigPanel />)

    expect(
      screen.getByText("Use {{variable}} for template placeholders")
    ).toBeInTheDocument()
    expect(screen.getByText("Balanced")).toBeInTheDocument()
    expect(screen.getByText("Checkpointed")).toBeInTheDocument()
    expect(screen.getByText("Save Artifact")).toBeInTheDocument()
  })

  it("merges deep research wait metadata with server schema fields", () => {
    const schema: WorkflowStepSchema = {
      type: "object",
      properties: {
        run_id: { type: "string", description: "Templated research run ID" },
        run: { type: "object", description: "Optional launch output object" },
        include_bundle: { type: "boolean", default: true },
        fail_on_cancelled: { type: "boolean", default: true }
      },
      required: ["run_id"]
    }
    setupStore("deep_research_wait", schema, {
      run_id: "{{ deep_research.run_id }}",
      include_bundle: true,
      fail_on_cancelled: true
    })

    vi.mocked(useWorkflowDynamicOptions).mockReturnValue({
      optionsByKey: {},
      loadingByKey: {}
    })

    render(<NodeConfigPanel />)

    expect(
      screen.getByText("Use {{variable}} for template placeholders")
    ).toBeInTheDocument()
    expect(screen.getByText("Include Bundle")).toBeInTheDocument()
    expect(screen.getByText("Fail on Cancelled")).toBeInTheDocument()
    expect(screen.getAllByText("Run").length).toBeGreaterThan(0)
  })

  it("merges deep research load-bundle metadata with server schema fields", () => {
    const schema: WorkflowStepSchema = {
      type: "object",
      properties: {
        run_id: {
          type: "string",
          description: "Templated research run ID, typically {{ deep_research_wait.run_id }}"
        },
        run: { type: "object", description: "Optional prior step output object containing run_id" },
        save_artifact: { type: "boolean", default: true }
      }
    }
    setupStore("deep_research_load_bundle", schema, {
      run_id: "{{ deep_research_wait.run_id }}",
      save_artifact: true
    })

    vi.mocked(useWorkflowDynamicOptions).mockReturnValue({
      optionsByKey: {},
      loadingByKey: {}
    })

    render(<NodeConfigPanel />)

    expect(
      screen.getByText("Use {{variable}} for template placeholders")
    ).toBeInTheDocument()
    expect(screen.getByText("Save Artifact")).toBeInTheDocument()
    expect(screen.getByText("Run ID")).toBeInTheDocument()
    expect(
      screen.getByText(
        "Loads references from a completed deep research run without returning the full bundle"
      )
    ).toBeInTheDocument()
  })

  it("merges deep research select-bundle-fields metadata with server schema fields", () => {
    const schema: WorkflowStepSchema = {
      type: "object",
      properties: {
        run_id: {
          type: "string",
          description: "Templated research run ID, typically {{ deep_research_wait.run_id }}"
        },
        run: { type: "object", description: "Optional prior step output object containing run_id" },
        fields: {
          type: "array",
          items: {
            type: "string",
            enum: ["question", "verification_summary", "unsupported_claims"]
          },
          description: "Canonical top-level bundle fields to load inline"
        },
        save_artifact: { type: "boolean", default: true }
      },
      required: ["fields"]
    }
    setupStore("deep_research_select_bundle_fields", schema, {
      run_id: "{{ deep_research_wait.run_id }}",
      fields: ["question", "verification_summary"],
      save_artifact: true
    })

    vi.mocked(useWorkflowDynamicOptions).mockReturnValue({
      optionsByKey: {},
      loadingByKey: {}
    })

    render(<NodeConfigPanel />)

    expect(
      screen.getByText("Use {{variable}} for template placeholders")
    ).toBeInTheDocument()
    expect(screen.getByText("Fields")).toBeInTheDocument()
    expect(screen.getByText("Question")).toBeInTheDocument()
    expect(screen.getByText("Verification Summary")).toBeInTheDocument()
    expect(screen.getByText("Save Artifact")).toBeInTheDocument()
    expect(
      screen.getByText(
        "Loads selected canonical bundle fields from a completed deep research run and returns null for missing allowed fields"
      )
    ).toBeInTheDocument()
  })
})
