// @vitest-environment jsdom
import React from "react"
import {
  QueryClient,
  QueryClientProvider
} from "@tanstack/react-query"
import { act, render, renderHook, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { PlaygroundImageGenModal } from "../PlaygroundImageGenModal"
import { usePlaygroundImageGen } from "../hooks/usePlaygroundImageGen"

const referenceImageCandidates = [
  {
    file_id: 17,
    title: "Reference portrait",
    mime_type: "image/jpeg",
    width: 768,
    height: 1024,
    created_at: "2026-03-27T10:00:00.000Z"
  }
]

const listReferenceImageCandidatesMock = vi.hoisted(() => vi.fn())

vi.mock("antd", () => {
  const InputComponent = ({
    value,
    onChange,
    placeholder,
    disabled,
    className,
    "data-testid": dataTestId
  }: any) => (
    <input
      value={value ?? ""}
      onChange={(event) => onChange?.(event)}
      placeholder={placeholder}
      disabled={disabled}
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
  }: any) => (
    <textarea
      value={value ?? ""}
      onChange={(event) => onChange?.(event)}
      placeholder={placeholder}
      readOnly={readOnly}
      disabled={disabled}
      className={className}
      data-testid={dataTestId}
    />
  )

  return {
    Button: ({
      children,
      onClick,
      disabled,
      loading,
      htmlType,
      className,
      title,
      "aria-label": ariaLabel
    }: any) => (
      <button
        type={htmlType || "button"}
        onClick={onClick}
        disabled={disabled || loading}
        className={className}
        title={title}
        aria-label={ariaLabel}
      >
        {children}
      </button>
    ),
    Input: InputComponent,
    InputNumber: ({ value, onChange, disabled }: any) => (
      <input
        type="number"
        value={value ?? ""}
        onChange={(event) => onChange?.(Number(event.target.value))}
        disabled={disabled}
      />
    ),
    Modal: ({ open, children }: { open?: boolean; children: React.ReactNode }) =>
      open ? <div data-testid="modal">{children}</div> : null,
    Radio: {
      Group: ({ children }: any) => <div>{children}</div>,
      Button: ({ children }: any) => <button type="button">{children}</button>
    },
    Select: ({
      value,
      options = [],
      onChange,
      disabled,
      loading,
      "data-testid": dataTestId,
      placeholder,
      allowClear
    }: any) => (
      <select
        aria-label={placeholder}
        data-testid={dataTestId}
        value={value ?? ""}
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
        {options.map((option: any) => (
          <option key={String(option.value)} value={String(option.value)}>
            {String(option.label)}
          </option>
        ))}
      </select>
    )
  }
})

vi.mock("lucide-react", () => ({
  WandSparkles: () => null
}))

vi.mock("@/utils/provider-registry", () => ({
  getProviderDisplayName: (provider: string) => provider
}))

vi.mock("../hooks", () => ({
  toText: (value: unknown) => String(value ?? "")
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    listReferenceImageCandidates: listReferenceImageCandidatesMock
  }
}))

describe("PlaygroundImageGenModal reference image picker", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("renders reference image candidates and updates the selected file id", async () => {
    const user = userEvent.setup()
    const onReferenceFileIdChange = vi.fn()

    render(
      <PlaygroundImageGenModal
        {...({
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
          submitting: false,
          onSubmit: vi.fn(),
          t: (_key: string, fallback?: string) => fallback || _key,
          referenceImageCandidates,
          referenceImageCandidatesLoading: false,
          referenceFileId: undefined,
          onReferenceFileIdChange
        } as any)}
      />
    )

    const picker = screen.getByTestId("image-generate-reference-image-select")
    expect(picker).toBeInTheDocument()
    expect(screen.getByRole("option", { name: /Reference portrait/i })).toBeInTheDocument()

    await user.selectOptions(picker, "17")
    expect(onReferenceFileIdChange).toHaveBeenCalledWith(17)
  })

  it("submits the selected reference file id and clears it on later opens", async () => {
    listReferenceImageCandidatesMock.mockResolvedValue({
      items: referenceImageCandidates
    })

    const queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false
        }
      }
    })
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    )

    const sendMessage = vi.fn(async () => ({ status: "submitted" as const }))
    const notificationApi = {
      error: vi.fn(),
      success: vi.fn()
    }

    const { result } = renderHook(
      () =>
        usePlaygroundImageGen({
          imageBackendDefaultTrimmed: "",
          imageBackendOptions: [],
          imageEventSyncChatMode: "off",
          imageEventSyncGlobalDefault: "off",
          updateChatSettings: vi.fn(),
          setImageEventSyncGlobalDefault: vi.fn(),
          messages: [],
          selectedCharacterName: null,
          selectedModel: "deepseek-chat",
          currentApiProvider: "openai",
          formMessage: "",
          sendMessage,
          textAreaFocus: vi.fn(),
          notificationApi,
          t: (_key: string, fallback?: string) => fallback || _key,
          setToolsPopoverOpen: vi.fn()
        }),
      { wrapper }
    )

    act(() => {
      result.current.openImageGenerateModal()
    })

    await waitFor(() =>
      expect(listReferenceImageCandidatesMock).toHaveBeenCalledTimes(1)
    )

    act(() => {
      result.current.setImageGenerateBackend("comfyui")
      result.current.setImageGeneratePrompt("A calm landscape")
      result.current.setImageGenerateReferenceFileId(17)
    })

    await act(async () => {
      await result.current.submitImageGenerateModal()
    })

    expect(sendMessage).toHaveBeenCalledTimes(1)
    expect(sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        imageGenerationRequest: expect.objectContaining({
          backend: "comfyui",
          prompt: "A calm landscape",
          referenceFileId: 17
        })
      })
    )
    expect(result.current.imageGenerateReferenceFileId).toBeUndefined()

    act(() => {
      result.current.setImageGenerateReferenceFileId(17)
      result.current.closeImageGenerateModal()
      result.current.openImageGenerateModal()
    })

    expect(result.current.imageGenerateReferenceFileId).toBeUndefined()
  })
})
