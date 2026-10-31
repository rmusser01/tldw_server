import { useStoreChatModelSettings } from "@/store/model"
import { fireEvent, render, screen } from "@testing-library/react"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ControlRow } from "../ControlRow"

const mocks = vi.hoisted(() => ({
  setSelectedAssistant: vi.fn(),
  setStorage: vi.fn(),
  setToolCatalog: vi.fn(),
  setToolCatalogId: vi.fn(),
  setToolModules: vi.fn(),
  setToolCatalogStrict: vi.fn(),
  setMcpToolEnabled: vi.fn(),
  resetMcpToolFilter: vi.fn(),
  openModelSelect: vi.fn()
}))
const chatModelsState = vi.hoisted(() => ({
  data: undefined as Array<Record<string, unknown>> | undefined,
  isLoading: true
}))

const buildChatModel = (
  overrides: Record<string, unknown> = {}
): Record<string, unknown> => ({
  id: "gpt-4o",
  name: "tldw:gpt-4o",
  model: "tldw:gpt-4o",
  provider: "openai",
  nickname: "GPT-4o",
  context_length: 128_000,
  avatar: undefined,
  modified_at: "2026-08-02T00:00:00.000Z",
  size: 0,
  digest: "",
  is_configured: true,
  provider_is_configured: true,
  catalog_only: false,
  provider_enabled: true,
  availability: "available",
  readiness_reason_code: undefined,
  readiness_message: undefined,
  chat_provider: "openai",
  details: {
    provider: "openai",
    capabilities: ["chat"],
    type: "chat",
    modalities: ["text"],
    is_configured: true,
    provider_is_configured: true,
    catalog_only: false,
    provider_enabled: true,
    availability: "available",
    readiness_reason_code: undefined,
    readiness_message: undefined,
    chat_provider: "openai"
  },
  ...overrides
})

vi.mock("wxt/browser", () => ({
  browser: { runtime: {}, tabs: {} }
}))

vi.mock("antd", () => ({
  Input: (props: React.InputHTMLAttributes<HTMLInputElement>) => (
    <input {...props} />
  ),
  InputNumber: (props: React.InputHTMLAttributes<HTMLInputElement>) => (
    <input type="number" {...props} />
  ),
  Popover: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
  Radio: {
    Group: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
    Button: ({ children }: { children?: React.ReactNode }) => <>{children}</>
  },
  Select: Object.assign(
    ({ children }: { children?: React.ReactNode }) => (
      <select>{children}</select>
    ),
    {
      Option: ({ children }: { children?: React.ReactNode }) => (
        <option>{children}</option>
      ),
      OptGroup: ({ children }: { children?: React.ReactNode }) => (
        <optgroup>{children}</optgroup>
      )
    }
  ),
  Switch: () => <button type="button" role="switch" />,
  Tooltip: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
  Upload: ({ children }: { children?: React.ReactNode }) => <>{children}</>
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: () => chatModelsState
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, initialValue: unknown) =>
    [
      key === "selectedModel" ? "gpt-4o" : initialValue,
      mocks.setStorage
    ] as const
}))

vi.mock("@/components/Common/ModelSelect", () => ({
  ModelSelect: React.forwardRef((_props, ref) => {
    const triggerRef = React.useRef<HTMLButtonElement>(null)
    React.useImperativeHandle(ref, () => ({
      openAndFocus: () => {
        mocks.openModelSelect()
        triggerRef.current?.focus()
      }
    }))
    return (
      <button ref={triggerRef} type="button" data-testid="chat-model-select">
        Model
      </button>
    )
  })
}))

vi.mock("@/components/Common/PromptSelect", () => ({
  PromptSelect: ({
    systemPrompt,
    setSystemPrompt,
    selectedModel,
    currentProvider,
    promptAssistContextKey,
    onSelectModel
  }: {
    systemPrompt?: string
    setSystemPrompt: (value: string) => void
    selectedModel?: string | null
    currentProvider?: string | null
    promptAssistContextKey?: string
    onSelectModel?: () => void
  }) => (
    <div>
      <output data-testid="owned-system-prompt">
        {systemPrompt === undefined ? "undefined" : systemPrompt}
      </output>
      <button
        type="button"
        onClick={() =>
          setSystemPrompt(`Conversation override for ${currentProvider}`)
        }>
        Edit system override
      </button>
      <output data-testid="prompt-assist-route">
        {selectedModel ?? "none"}:{currentProvider ?? "none"}
      </output>
      <output data-testid="prompt-assist-context">
        {promptAssistContextKey ?? "none"}
      </output>
      <button type="button" onClick={onSelectModel}>
        Recover model selection
      </button>
    </div>
  )
}))

vi.mock("@/components/Common/FeatureHint", () => ({
  FeatureHint: ({ children }: { children?: React.ReactNode }) => (
    <>{children}</>
  ),
  useFeatureHintSeen: () => true
}))

vi.mock("@/components/Common/McpToolSelector", () => ({
  McpToolSelector: () => null
}))

vi.mock("../ConversationContextPopover", () => ({
  ConversationContextPopover: () => null
}))

vi.mock("@/hooks/useChatMoodBadgePreference", () => ({
  useChatMoodBadgePreference: () => [false, vi.fn()] as const
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({ capabilities: {} })
}))

vi.mock("@/hooks/useMcpTools", () => ({
  useMcpTools: () => ({
    hasMcp: false,
    healthState: "unavailable",
    discoveredTools: [],
    chatTools: [],
    toolCounts: { enabled: 0, total: 0 },
    toolsLoading: false,
    catalogs: [],
    catalogsLoading: false,
    toolCatalog: "",
    toolCatalogId: null,
    toolModules: [],
    moduleOptions: [],
    moduleOptionsLoading: false,
    toolCatalogStrict: false,
    setToolCatalog: mocks.setToolCatalog,
    setToolCatalogId: mocks.setToolCatalogId,
    setToolModules: mocks.setToolModules,
    setToolCatalogStrict: mocks.setToolCatalogStrict,
    setToolEnabled: mocks.setMcpToolEnabled,
    resetToolFilter: mocks.resetMcpToolFilter
  })
}))

vi.mock("@/services/tldw-server", () => ({ fetchChatModels: vi.fn() }))
vi.mock("@/services/sidepanel-chat-handoff", () => ({
  buildSidepanelChatHandoffRoute: vi.fn(),
  consumeSidepanelChatHandoff: vi.fn(),
  createSidepanelChatHandoff: vi.fn()
}))
vi.mock("@/utils/quick-ingest-open", () => ({
  buildQuickIngestOpenDetailFromUrl: vi.fn(),
  requestQuickIngestOpen: vi.fn()
}))
vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [null, mocks.setSelectedAssistant] as const
}))

const defaultProps = () => ({
  selectedModel: "gpt-4o",
  currentProvider: "openai",
  promptAssistContextKey: "local:history-7",
  selectedSystemPrompt: "prompt-1",
  setSelectedSystemPrompt: vi.fn(),
  setSelectedQuickPrompt: vi.fn(),
  selectedCharacterId: null,
  setSelectedCharacterId: vi.fn(),
  webSearch: false,
  setWebSearch: vi.fn(),
  chatMode: "normal" as const,
  setChatMode: vi.fn(),
  toolChoice: "auto" as const,
  setToolChoice: vi.fn(),
  onImageUpload: vi.fn(),
  onToggleRag: vi.fn(),
  isConnected: true
})

describe("ControlRow system prompt override ownership", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useStoreChatModelSettings.getState().reset()
    chatModelsState.data = undefined
    chatModelsState.isLoading = true
  })

  it("hands prompt recovery to the existing sidepanel model selector", () => {
    render(<ControlRow {...defaultProps()} />)

    fireEvent.click(
      screen.getByRole("button", { name: "Recover model selection" })
    )

    expect(mocks.openModelSelect).toHaveBeenCalledTimes(1)
    expect(screen.getByTestId("chat-model-select")).toHaveFocus()
  })

  it("synchronizes the selected route and isolates edits from global and stale scopes", () => {
    const { rerender } = render(<ControlRow {...defaultProps()} />)

    expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe(
      "openai:gpt-4o"
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Edit system override" })
    )

    expect(useStoreChatModelSettings.getState().systemPrompt).toBe(
      "Conversation override for openai"
    )
    expect(
      useStoreChatModelSettings.getState().getEffectiveSettings("openai:gpt-4o")
        .systemPrompt
    ).toBe("Conversation override for openai")
    expect(
      useStoreChatModelSettings.getState().globalSettings.systemPrompt
    ).toBeUndefined()
    expect(
      useStoreChatModelSettings.getState().getEffectiveSettings("azure:gpt-4o")
        .systemPrompt
    ).toBeUndefined()
    expect(screen.getByTestId("prompt-assist-route")).toHaveTextContent(
      "gpt-4o:openai"
    )
    expect(screen.getByTestId("prompt-assist-context")).toHaveTextContent(
      "local:history-7"
    )

    rerender(
      <ControlRow
        {...defaultProps()}
        currentProvider="anthropic"
        selectedModel="claude-3-5-sonnet"
      />
    )

    expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe(
      "anthropic:claude-3-5-sonnet"
    )
    expect(screen.getByTestId("owned-system-prompt")).toHaveTextContent(
      "undefined"
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Edit system override" })
    )

    expect(
      useStoreChatModelSettings
        .getState()
        .getEffectiveSettings("anthropic:claude-3-5-sonnet").systemPrompt
    ).toBe("Conversation override for anthropic")
    expect(
      useStoreChatModelSettings.getState().getEffectiveSettings("openai:gpt-4o")
        .systemPrompt
    ).toBe("Conversation override for openai")
  })

  it("clears the owned override when the selected library template changes", () => {
    const { rerender } = render(<ControlRow {...defaultProps()} />)

    fireEvent.click(
      screen.getByRole("button", { name: "Edit system override" })
    )

    expect(screen.getByTestId("owned-system-prompt")).toHaveTextContent(
      "Conversation override for openai"
    )

    rerender(<ControlRow {...defaultProps()} />)

    expect(
      useStoreChatModelSettings.getState().getEffectiveSettings("openai:gpt-4o")
        .systemPrompt
    ).toBe("Conversation override for openai")

    rerender(<ControlRow {...defaultProps()} selectedSystemPrompt="prompt-2" />)

    expect(useStoreChatModelSettings.getState().systemPrompt).toBeUndefined()
    expect(screen.getByTestId("owned-system-prompt")).toHaveTextContent(
      "undefined"
    )
    expect(
      useStoreChatModelSettings.getState().getEffectiveSettings("openai:gpt-4o")
        .systemPrompt
    ).toBeUndefined()
  })

  it("uses catalog provider metadata when the active provider is absent", () => {
    chatModelsState.data = [buildChatModel()]
    chatModelsState.isLoading = false
    useStoreChatModelSettings
      .getState()
      .updateScopedSetting(
        "azure:gpt-4o",
        "systemPrompt",
        "Stale Azure override"
      )

    const props = { ...defaultProps(), currentProvider: undefined }
    const { rerender } = render(<ControlRow {...props} />)

    expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe(
      "openai:gpt-4o"
    )
    expect(screen.getByTestId("owned-system-prompt")).toHaveTextContent(
      "undefined"
    )
    expect(screen.getByTestId("prompt-assist-route")).toHaveTextContent(
      "gpt-4o:openai"
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Edit system override" })
    )

    expect(
      useStoreChatModelSettings.getState().getEffectiveSettings("openai:gpt-4o")
        .systemPrompt
    ).toBe("Conversation override for openai")
    expect(
      useStoreChatModelSettings.getState().getEffectiveSettings("azure:gpt-4o")
        .systemPrompt
    ).toBe("Stale Azure override")
    expect(
      useStoreChatModelSettings.getState().globalSettings.systemPrompt
    ).toBeUndefined()

    rerender(<ControlRow {...props} selectedSystemPrompt="prompt-2" />)

    expect(
      useStoreChatModelSettings.getState().getEffectiveSettings("openai:gpt-4o")
        .systemPrompt
    ).toBeUndefined()
    expect(
      useStoreChatModelSettings.getState().getEffectiveSettings("azure:gpt-4o")
        .systemPrompt
    ).toBe("Stale Azure override")
    expect(
      useStoreChatModelSettings.getState().globalSettings.systemPrompt
    ).toBeUndefined()
  })

  it("fails closed until provider metadata resolves and switches scope immediately", () => {
    const props = { ...defaultProps(), currentProvider: undefined }
    const rendered = render(<ControlRow {...props} />)

    fireEvent.click(
      screen.getByRole("button", { name: "Edit system override" })
    )
    expect(
      useStoreChatModelSettings.getState().scopedSettingsByModelKey
    ).toEqual({})
    expect(
      useStoreChatModelSettings.getState().globalSettings.systemPrompt
    ).toBeUndefined()

    chatModelsState.data = [
      buildChatModel({
        provider: "unknown",
        chat_provider: undefined,
        details: {
          provider: undefined,
          capabilities: ["chat"],
          type: "chat",
          modalities: ["text"]
        }
      })
    ]
    chatModelsState.isLoading = false
    rendered.rerender(
      <ControlRow {...props} promptAssistContextKey="local:history-missing" />
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Edit system override" })
    )
    expect(
      useStoreChatModelSettings.getState().scopedSettingsByModelKey
    ).toEqual({})
    expect(
      useStoreChatModelSettings.getState().globalSettings.systemPrompt
    ).toBeUndefined()

    chatModelsState.data = [buildChatModel()]
    rendered.rerender(
      <ControlRow {...props} promptAssistContextKey="local:history-resolved" />
    )

    expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe(
      "openai:gpt-4o"
    )
    fireEvent.click(
      screen.getByRole("button", { name: "Edit system override" })
    )
    expect(
      useStoreChatModelSettings.getState().getEffectiveSettings("openai:gpt-4o")
        .systemPrompt
    ).toBe("Conversation override for openai")

    chatModelsState.data = [
      buildChatModel({
        provider: "anthropic",
        chat_provider: "anthropic",
        details: {
          provider: "anthropic",
          capabilities: ["chat"],
          type: "chat",
          modalities: ["text"]
        }
      })
    ]
    rendered.rerender(
      <ControlRow {...props} promptAssistContextKey="local:history-anthropic" />
    )

    expect(screen.getByTestId("owned-system-prompt")).toHaveTextContent(
      "undefined"
    )
    expect(screen.getByTestId("prompt-assist-route")).toHaveTextContent(
      "gpt-4o:anthropic"
    )
    fireEvent.click(
      screen.getByRole("button", { name: "Edit system override" })
    )
    expect(
      useStoreChatModelSettings
        .getState()
        .getEffectiveSettings("anthropic:gpt-4o").systemPrompt
    ).toBe("Conversation override for anthropic")
    expect(
      useStoreChatModelSettings.getState().getEffectiveSettings("openai:gpt-4o")
        .systemPrompt
    ).toBe("Conversation override for openai")
  })

  it("uses a provider-qualified selection without guessing from its model name", () => {
    const props = {
      ...defaultProps(),
      currentProvider: undefined,
      selectedModel: "anthropic:claude-3-5-sonnet"
    }
    render(<ControlRow {...props} />)

    expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe(
      "anthropic:claude-3-5-sonnet"
    )
    expect(screen.getByTestId("prompt-assist-route")).toHaveTextContent(
      "anthropic:claude-3-5-sonnet:anthropic"
    )
    fireEvent.click(
      screen.getByRole("button", { name: "Edit system override" })
    )
    expect(
      useStoreChatModelSettings
        .getState()
        .getEffectiveSettings("anthropic:claude-3-5-sonnet").systemPrompt
    ).toBe("Conversation override for anthropic")
  })

  it("prefers a provider-qualified selection over a stale current provider", () => {
    const anthropicScope = "anthropic:claude-3-5-sonnet"
    const staleOpenAiScope = "openai:claude-3-5-sonnet"
    useStoreChatModelSettings
      .getState()
      .updateScopedSetting(
        anthropicScope,
        "systemPrompt",
        "Existing Anthropic override"
      )
    useStoreChatModelSettings
      .getState()
      .updateScopedSetting(
        staleOpenAiScope,
        "systemPrompt",
        "Stale OpenAI override"
      )

    const props = {
      ...defaultProps(),
      selectedModel: "anthropic:claude-3-5-sonnet",
      currentProvider: "openai"
    }
    const { rerender } = render(<ControlRow {...props} />)

    expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe(
      anthropicScope
    )
    expect(screen.getByTestId("owned-system-prompt")).toHaveTextContent(
      "Existing Anthropic override"
    )
    expect(screen.getByTestId("prompt-assist-route")).toHaveTextContent(
      "anthropic:claude-3-5-sonnet:anthropic"
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Edit system override" })
    )

    expect(
      useStoreChatModelSettings.getState().getEffectiveSettings(anthropicScope)
        .systemPrompt
    ).toBe("Conversation override for anthropic")
    expect(
      useStoreChatModelSettings
        .getState()
        .getEffectiveSettings(staleOpenAiScope).systemPrompt
    ).toBe("Stale OpenAI override")
    expect(
      useStoreChatModelSettings.getState().globalSettings.systemPrompt
    ).toBeUndefined()

    rerender(<ControlRow {...props} selectedSystemPrompt="prompt-2" />)

    expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe(
      anthropicScope
    )
    expect(
      useStoreChatModelSettings.getState().getEffectiveSettings(anthropicScope)
        .systemPrompt
    ).toBeUndefined()
    expect(
      useStoreChatModelSettings
        .getState()
        .getEffectiveSettings(staleOpenAiScope).systemPrompt
    ).toBe("Stale OpenAI override")
    expect(
      useStoreChatModelSettings.getState().globalSettings.systemPrompt
    ).toBeUndefined()
  })
})
