// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { PlaygroundContextWindowModal } from "../PlaygroundContextWindowModal"
import {
  PlaygroundImageGenModal,
  type PlaygroundImageGenModalProps
} from "../PlaygroundImageGenModal"
import { PlaygroundMcpSettingsModal } from "../PlaygroundMcpSettingsModal"
import { PlaygroundRawRequestModal } from "../PlaygroundRawRequestModal"
import {
  PlaygroundStartupTemplateModal,
  type PlaygroundStartupTemplateModalProps
} from "../PlaygroundStartupTemplateModal"

type MockInputProps = React.InputHTMLAttributes<HTMLInputElement> & {
  "data-testid"?: string
}

type MockTextAreaProps = React.TextareaHTMLAttributes<HTMLTextAreaElement> & {
  "data-testid"?: string
}

type MockSelectOption = {
  value: string | number
  label: React.ReactNode
}

type MockSelectProps = {
  value?: string | number
  options?: MockSelectOption[]
  onChange?: (value: string | number | undefined) => void
  disabled?: boolean
  loading?: boolean
  "data-testid"?: string
  placeholder?: string
  allowClear?: boolean
  children?: React.ReactNode
}

type MockButtonProps = Omit<
  React.ButtonHTMLAttributes<HTMLButtonElement>,
  "type"
> & {
  variant?: string
  type?: string
  htmlType?: "button" | "submit" | "reset"
  icon?: React.ReactNode
  loading?: boolean
  "data-testid"?: string
}

const t = (
  key: string,
  fallback?: string | { defaultValue?: string; count?: number },
  options?: { count?: number }
) => {
  const value =
    typeof fallback === "string" ? fallback : (fallback?.defaultValue ?? key)
  const count =
    options?.count ??
    (typeof fallback === "object" ? fallback.count : undefined)
  return typeof count === "number"
    ? value.replace("{{count}}", String(count))
    : value
}

vi.mock("antd", () => {
  const InputComponent = ({
    value,
    onChange,
    placeholder,
    disabled,
    readOnly,
    className,
    "data-testid": dataTestId
  }: MockInputProps) => (
    <input
      value={String(value ?? "")}
      onChange={(event) => onChange?.(event)}
      placeholder={placeholder}
      disabled={disabled}
      readOnly={readOnly}
      className={className}
      data-testid={dataTestId}
    />
  )
  InputComponent.TextArea = ({
    value,
    onChange,
    placeholder,
    readOnly,
    disabled,
    className,
    "data-testid": dataTestId
  }: MockTextAreaProps) => (
    <textarea
      value={String(value ?? "")}
      onChange={(event) => onChange?.(event)}
      placeholder={placeholder}
      readOnly={readOnly}
      disabled={disabled}
      className={className}
      data-testid={dataTestId}
    />
  )

  const SelectComponent = ({
    value,
    options = [],
    onChange,
    disabled,
    loading,
    "data-testid": dataTestId,
    placeholder,
    allowClear,
    children
  }: MockSelectProps) => (
    <select
      aria-label={placeholder}
      data-testid={dataTestId}
      value={String(value ?? "")}
      onChange={(event) =>
        onChange?.(
          event.target.value === ""
            ? undefined
            : Number.isNaN(Number(event.target.value))
              ? event.target.value
              : Number(event.target.value)
        )
      }
      disabled={disabled || loading}
    >
      {allowClear ? <option value="" /> : null}
      {options.map((option) => (
        <option key={String(option.value)} value={String(option.value)}>
          {String(option.label)}
        </option>
      ))}
      {children}
    </select>
  )
  SelectComponent.Option = ({
    value,
    children
  }: {
    value: string | number
    children?: React.ReactNode
  }) => <option value={value}>{children}</option>
  SelectComponent.OptGroup = ({ children }: { children?: React.ReactNode }) => (
    <>{children}</>
  )

  return {
    Button: ({
      children,
      onClick,
      disabled,
      loading,
      variant,
      type,
      htmlType,
      icon,
      className,
      title,
      "aria-label": ariaLabel,
      "data-testid": dataTestId
    }: MockButtonProps) => (
      <button
        type={htmlType || "button"}
        onClick={onClick}
        disabled={disabled || loading}
        className={className}
        title={title}
        aria-label={ariaLabel}
        data-testid={dataTestId}
        data-variant={variant}
        data-antd-type={type}
        aria-busy={loading || undefined}
      >
        {icon}
        {children}
      </button>
    ),
    Input: InputComponent,
    InputNumber: ({
      value,
      onChange,
      disabled,
      placeholder
    }: {
      value?: number
      onChange?: (value: number) => void
      disabled?: boolean
      placeholder?: string
    }) => (
      <input
        type="number"
        value={value ?? ""}
        onChange={(event) => onChange?.(Number(event.target.value))}
        disabled={disabled}
        placeholder={placeholder}
      />
    ),
    Modal: ({
      open,
      children,
      footer,
      title
    }: {
      open?: boolean
      children?: React.ReactNode
      footer?: React.ReactNode
      title?: React.ReactNode
    }) =>
      open ? (
        <section role="dialog" aria-label={String(title ?? "Modal")}>
          {children}
          {footer !== null && footer !== undefined ? (
            <div data-testid="antd-modal-footer">{footer}</div>
          ) : null}
        </section>
      ) : null,
    Radio: {
      Group: ({ children }: { children?: React.ReactNode }) => (
        <div>{children}</div>
      ),
      Button: ({ children }: { children?: React.ReactNode }) => (
        <button type="button">{children}</button>
      )
    },
    Select: SelectComponent,
    Switch: ({
      checked,
      onChange
    }: {
      checked?: boolean
      onChange?: (checked: boolean) => void
    }) => (
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange?.(event.target.checked)}
      />
    )
  }
})

vi.mock("lucide-react", () => ({
  Loader2: () => <span data-testid="loader-icon" />,
  WandSparkles: () => <span data-testid="wand-icon" />
}))

vi.mock("@/utils/provider-registry", () => ({
  getProviderDisplayName: (provider: string) => provider
}))

vi.mock("../ContextFootprintPanel", () => ({
  ContextFootprintPanel: () => <div data-testid="context-footprint-panel" />
}))

vi.mock("../SessionInsightsPanel", () => ({
  SessionInsightsPanel: () => <div data-testid="session-insights-panel" />
}))

vi.mock("../hooks", () => ({
  CONTEXT_FOOTPRINT_THRESHOLD_PERCENT: 80,
  toText: (value: unknown) => String(value ?? "")
}))

const baseImageGenProps: PlaygroundImageGenModalProps = {
  open: true,
  onClose: vi.fn(),
  busy: false,
  backend: "comfyui",
  backendOptions: [{ value: "comfyui", label: "ComfyUI" }],
  onBackendChange: vi.fn(),
  onHydrateSettings: vi.fn(),
  promptMode: "scene",
  onPromptModeChange: vi.fn(),
  promptStrategies: [{ id: "scene", label: "Scene" }],
  syncPolicy: "inherit",
  onSyncPolicyChange: vi.fn(),
  syncChatMode: "off",
  onSyncChatModeChange: vi.fn(),
  syncGlobalDefault: "off",
  onSyncGlobalDefaultChange: vi.fn(),
  resolvedSyncMode: "off",
  prompt: "A calm landscape",
  onPromptChange: vi.fn(),
  contextBreakdown: [],
  onClearRefineState: vi.fn(),
  refineSubmitting: false,
  refineBaseline: "",
  refineCandidate: null,
  refineModel: null,
  refineLatencyMs: null,
  refineDiff: null,
  onCreateDraft: vi.fn(),
  onRefine: vi.fn(),
  onApplyRefined: vi.fn(),
  onRejectRefined: vi.fn(),
  format: "png",
  onFormatChange: vi.fn(),
  width: undefined,
  onWidthChange: vi.fn(),
  height: undefined,
  onHeightChange: vi.fn(),
  steps: undefined,
  onStepsChange: vi.fn(),
  cfgScale: undefined,
  onCfgScaleChange: vi.fn(),
  seed: undefined,
  onSeedChange: vi.fn(),
  sampler: "",
  onSamplerChange: vi.fn(),
  model: "",
  onModelChange: vi.fn(),
  negativePrompt: "",
  onNegativePromptChange: vi.fn(),
  extraParams: "",
  onExtraParamsChange: vi.fn(),
  referenceFileId: undefined,
  onReferenceFileIdChange: vi.fn(),
  referenceImageCandidates: [],
  referenceImageCandidatesLoading: false,
  submitting: false,
  onSubmit: vi.fn(),
  t
}

describe("Playground modal footers design-system migration", () => {
  it("renders startup template destructive, cancel, and apply actions through ModalFooter", () => {
    const onDelete = vi.fn()
    const onClose = vi.fn()
    const onApply = vi.fn()

    render(
      <PlaygroundStartupTemplateModal
        preview={
          {
            id: "template-1",
            selectedModel: "llama",
            character: { name: "Guide" },
            ragPinnedResults: []
          } as NonNullable<PlaygroundStartupTemplateModalProps["preview"]>
        }
        onClose={onClose}
        onDelete={onDelete}
        onApply={onApply}
        promptDescription="Prompt"
        promptResolution={null}
        preset={undefined}
        t={t}
      />
    )

    const footer = screen.getByTestId("startup-template-preview-modal-footer")
    expect(footer).toHaveAttribute("data-ds-component", "ModalFooter")

    fireEvent.click(
      within(footer).getByRole("button", { name: "Delete template" })
    )
    expect(onDelete).toHaveBeenCalledWith("template-1")

    fireEvent.click(within(footer).getByRole("button", { name: "Cancel" }))
    expect(onClose).toHaveBeenCalledTimes(1)

    fireEvent.click(
      within(footer).getByRole("button", { name: "Apply template" })
    )
    expect(onApply).toHaveBeenCalledTimes(1)
  })

  it("preserves ordered raw request footer actions and disabled copy state", () => {
    const onRefresh = vi.fn()
    const onCopy = vi.fn()
    const onClose = vi.fn()
    const onExtra = vi.fn()

    render(
      <PlaygroundRawRequestModal
        open
        onClose={onClose}
        snapshot={null}
        json=""
        onRefresh={onRefresh}
        onCopy={onCopy}
        extraFooter={<button onClick={onExtra}>Run research</button>}
        t={t}
      />
    )

    const footer = screen.getByTestId("raw-chat-request-modal-footer")
    expect(footer).toHaveAttribute("data-ds-component", "ModalFooter")
    expect(
      within(footer)
        .getAllByRole("button")
        .map((button) => button.textContent)
    ).toEqual(["Refresh", "Run research", "Copy", "Close"])

    fireEvent.click(within(footer).getByRole("button", { name: "Refresh" }))
    expect(onRefresh).toHaveBeenCalledTimes(1)

    fireEvent.click(
      within(footer).getByRole("button", { name: "Run research" })
    )
    expect(onExtra).toHaveBeenCalledTimes(1)

    expect(within(footer).getByRole("button", { name: "Copy" })).toBeDisabled()

    fireEvent.click(within(footer).getByRole("button", { name: "Close" }))
    expect(onClose).toHaveBeenCalledTimes(1)
    expect(onCopy).not.toHaveBeenCalled()
  })

  it("renders context-window and session-insights actions through ModalFooter", () => {
    const onCloseContextWindow = vi.fn()
    const onSaveContextWindow = vi.fn()
    const onResetContextWindow = vi.fn()
    const onCloseSessionInsights = vi.fn()

    render(
      <PlaygroundContextWindowModal
        contextWindowModalOpen
        onCloseContextWindow={onCloseContextWindow}
        onSaveContextWindow={onSaveContextWindow}
        onResetContextWindow={onResetContextWindow}
        contextWindowDraftValue={4096}
        onContextWindowDraftChange={vi.fn()}
        resolvedMaxContext={4096}
        requestedContextWindowOverride={4096}
        modelContextLength={8192}
        isContextWindowOverrideActive
        isContextWindowOverrideClamped={false}
        nonMessageContextPercent={20}
        showNonMessageContextWarning={false}
        tokenBudgetRiskLabel="Low"
        tokenBudgetRisk={{ overflowTokens: 0 }}
        contextFootprintRows={[]}
        formatContextWindowValue={(value) => String(value ?? "default")}
        onClearPromptContext={vi.fn()}
        onClearPinnedSourceContext={vi.fn()}
        onClearHistoryContext={vi.fn()}
        onCreateSummaryCheckpoint={vi.fn()}
        onReviewCharacterContext={vi.fn()}
        onTrimLargestContextContributor={vi.fn()}
        sessionInsightsOpen
        onCloseSessionInsights={onCloseSessionInsights}
        sessionInsights={{}}
        t={t}
      />
    )

    const contextFooter = screen.getByTestId("context-window-modal-footer")
    expect(contextFooter).toHaveAttribute("data-ds-component", "ModalFooter")

    fireEvent.click(
      within(contextFooter).getByRole("button", { name: "Use model default" })
    )
    expect(onResetContextWindow).toHaveBeenCalledTimes(1)

    fireEvent.click(within(contextFooter).getByRole("button", { name: "Save" }))
    expect(onSaveContextWindow).toHaveBeenCalledTimes(1)

    const insightsFooter = screen.getByTestId("session-insights-modal-footer")
    expect(insightsFooter).toHaveAttribute("data-ds-component", "ModalFooter")
    fireEvent.click(
      within(insightsFooter).getByRole("button", { name: "Close" })
    )
    expect(onCloseSessionInsights).toHaveBeenCalledTimes(1)
  })

  it("preserves image-generation footer disabled and loading states", () => {
    const onRefine = vi.fn()
    const onSubmit = vi.fn()

    render(
      <PlaygroundImageGenModal
        {...baseImageGenProps}
        busy={true}
        refineSubmitting={false}
        submitting={false}
        onRefine={onRefine}
        onSubmit={onSubmit}
      />
    )

    const footer = screen.getByTestId("image-generation-modal-footer")
    expect(footer).toHaveAttribute("data-ds-component", "ModalFooter")
    expect(
      within(footer).getByRole("button", { name: "Create prompt" })
    ).toBeDisabled()
    expect(
      within(footer).getByRole("button", { name: "Cancel" })
    ).toBeDisabled()
    expect(
      within(footer).getByRole("button", { name: "Refine with LLM" })
    ).toBeDisabled()
    expect(
      within(footer).getByRole("button", { name: "Refine with LLM" })
    ).toHaveAttribute("aria-busy", "true")
    expect(
      within(footer).getByRole("button", { name: "Generate image" })
    ).toBeDisabled()

    fireEvent.click(
      within(footer).getByRole("button", { name: "Refine with LLM" })
    )
    fireEvent.click(
      within(footer).getByRole("button", { name: "Generate image" })
    )
    expect(onRefine).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it("adds an explicit MCP settings close action through ModalFooter", () => {
    const onClose = vi.fn()

    render(
      <PlaygroundMcpSettingsModal
        open
        onClose={onClose}
        hasMcp={false}
        mcpStatusLabel="Unavailable"
        catalogsLoading={false}
        catalogGroups={{ team: [], org: [], global: [] }}
        catalogDraft=""
        onCatalogDraftChange={vi.fn()}
        onCatalogCommit={vi.fn()}
        onCatalogSelect={vi.fn()}
        toolCatalogId={null}
        onToolCatalogIdChange={vi.fn()}
        toolCatalogStrict={false}
        onToolCatalogStrictChange={vi.fn()}
        moduleOptions={[]}
        moduleOptionsLoading={false}
        toolModules={[]}
        onModuleSelect={vi.fn()}
        discoveredTools={[]}
        toolCounts={{
          discovered: 0,
          executable: 0,
          disabled: 0,
          colliding: 0,
          chatEnabled: 0
        }}
        toolsLoading={false}
        mcpHealthState="unavailable"
        onToolEnabledChange={vi.fn()}
        onResetToolFilter={vi.fn()}
        isSmallModel={false}
        t={t}
      />
    )

    const footer = screen.getByTestId("mcp-settings-modal-footer")
    expect(footer).toHaveAttribute("data-ds-component", "ModalFooter")
    fireEvent.click(within(footer).getByRole("button", { name: "Close" }))
    expect(onClose).toHaveBeenCalledTimes(1)
  })
})
