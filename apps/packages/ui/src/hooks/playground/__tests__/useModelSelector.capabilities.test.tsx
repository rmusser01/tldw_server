import React from "react"
import { act, render, renderHook, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useModelSelector } from "../useModelSelector"
import { normalizeTldwModels } from "@/services/tldw/model-normalization"

const chatModelSettingsState = vi.hoisted(() => ({
  apiProvider: null as string | null,
  numCtx: null as number | null
}))

const storageSeed = vi.hoisted(() => ({
  values: new Map<string, unknown>()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallbackOrOptions?: unknown, maybeOptions?: Record<string, unknown>) => {
      let template = key
      let options: Record<string, unknown> | undefined
      if (typeof fallbackOrOptions === "string") {
        template = fallbackOrOptions
        options = maybeOptions
      } else if (
        fallbackOrOptions &&
        typeof fallbackOrOptions === "object" &&
        "defaultValue" in (fallbackOrOptions as Record<string, unknown>)
      ) {
        template = String(
          (fallbackOrOptions as { defaultValue?: unknown }).defaultValue ?? key
        )
        options = fallbackOrOptions as Record<string, unknown>
      } else {
        options = maybeOptions
      }
      if (!options) return template
      return template.replace(/\{\{(\w+)\}\}/g, (_match, token) => {
        const value = options?.[token]
        return value == null ? "" : String(value)
      })
    }
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, defaultValue: unknown) => {
    const initialValue =
      storageSeed.values.has(key)
        ? storageSeed.values.get(key)
        : key === "favoriteChatModels"
          ? []
          : key === "modelSelectSortMode"
            ? "provider"
            : defaultValue
    const [value, setValue] = React.useState(initialValue)
    return [value, setValue, { isLoading: false }] as const
  }
}))

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock("@/components/Common/ProviderIcon", () => ({
  ProviderIcons: ({ provider }: { provider?: string }) => (
    <span data-testid={`provider-icon-${provider || "unknown"}`} />
  )
}))

vi.mock("@/services/tldw", () => ({
  tldwModels: {
    getProviderDisplayName: (provider: string) =>
      provider ? provider.toUpperCase() : "CUSTOM"
  }
}))

vi.mock("@/utils/provider-registry", () => ({
  getProviderDisplayName: (provider?: string) =>
    provider ? provider.toUpperCase() : "OTHER"
}))

vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: (
    selector: (state: { apiProvider: string | null; numCtx: number | null }) => unknown
  ) => selector(chatModelSettingsState)
}))

const unwrapFirstMenuItem = (items: any[]) => {
  if (!Array.isArray(items) || items.length === 0) return null
  const first = items[0]
  if (first?.type === "group" && Array.isArray(first.children)) {
    return first.children[0] ?? null
  }
  return first
}

describe("useModelSelector capability rendering", () => {
  beforeEach(() => {
    storageSeed.values.clear()
    chatModelSettingsState.apiProvider = null
    chatModelSettingsState.numCtx = null
  })

  it("includes vision/tools/streaming/context and price badges in dropdown items", () => {
    const { result } = renderHook(() =>
      useModelSelector({
        composerModels: [
          {
            model: "openai/gpt-4o-mini",
            nickname: "GPT-4o mini",
            provider: "openai",
            context_length: 8192,
            details: {
              capabilities: ["vision", "tools", "streaming"],
              price_hint: "$0.15/$0.60"
            }
          }
        ],
        selectedModel: "openai/gpt-4o-mini",
        setSelectedModel: vi.fn(),
        navigate: vi.fn()
      })
    )

    const firstItem = unwrapFirstMenuItem(result.current.modelDropdownMenuItems)
    expect(firstItem).not.toBeNull()

    render(<>{firstItem?.label}</>)

    expect(screen.getByText("Vision")).toBeInTheDocument()
    expect(screen.getByText("Tools")).toBeInTheDocument()
    expect(screen.getByText("Streaming")).toBeInTheDocument()
    expect(screen.getByText("8k ctx")).toBeInTheDocument()
    expect(screen.getByText("$0.15/$0.60")).toBeInTheDocument()
  })

  it("defaults to configured models and exposes catalog models only in catalog scope", () => {
    const { result } = renderHook(() =>
      useModelSelector({
        composerModels: [
          {
            model: "configured-model",
            nickname: "Configured Model",
            provider: "openai",
            is_configured: true
          },
          {
            model: "catalog-model",
            nickname: "Catalog Model",
            provider: "openrouter",
            catalog_only: true
          }
        ],
        selectedModel: "configured-model",
        setSelectedModel: vi.fn(),
        navigate: vi.fn()
      })
    )

    expect(result.current.modelListScope).toBe("configured")
    expect(result.current.filteredModels.map((model: any) => model.model)).toEqual([
      "configured-model"
    ])

    act(() => {
      result.current.setModelListScope("catalog")
    })

    expect(result.current.filteredModels.map((model: any) => model.model)).toEqual([
      "configured-model",
      "catalog-model"
    ])
  })

  it("uses provider-qualified menu keys and selected metadata when model ids collide", () => {
    chatModelSettingsState.apiProvider = "anthropic"
    const setSelectedModel = vi.fn()

    const { result } = renderHook(() =>
      useModelSelector({
        composerModels: [
          {
            model: "shared-model",
            nickname: "OpenAI shared",
            provider: "openai",
            is_configured: true
          },
          {
            model: "shared-model",
            nickname: "Anthropic shared",
            provider: "anthropic",
            is_configured: true
          }
        ],
        selectedModel: "shared-model",
        setSelectedModel,
        navigate: vi.fn()
      })
    )

    const firstItem = unwrapFirstMenuItem(result.current.modelDropdownMenuItems)

    expect(result.current.selectedModelMeta?.provider).toBe("anthropic")
    expect(result.current.selectedModelKey).toBe("anthropic:shared-model")
    expect(firstItem?.key).toBe("anthropic:shared-model")

    act(() => {
      firstItem?.onClick?.()
    })

    expect(setSelectedModel).toHaveBeenCalledWith("anthropic:shared-model")
  })

  it("keeps tldw server transport prefixes out of selected provider:model keys", () => {
    const setSelectedModel = vi.fn()

    const { result } = renderHook(() =>
      useModelSelector({
        composerModels: [
          {
            id: "gpt-4o-mini",
            model: "tldw:gpt-4o-mini",
            nickname: "GPT-4o mini",
            provider: "openai",
            is_configured: true
          }
        ],
        selectedModel: "openai:gpt-4o-mini",
        setSelectedModel,
        navigate: vi.fn()
      })
    )

    const firstItem = unwrapFirstMenuItem(result.current.modelDropdownMenuItems)

    expect(result.current.selectedModelKey).toBe("openai:gpt-4o-mini")
    expect(firstItem?.key).toBe("openai:gpt-4o-mini")

    render(<>{firstItem?.label}</>)
    expect(screen.getByTestId("model-selector-option")).toHaveAttribute(
      "data-model-key",
      "openai:gpt-4o-mini"
    )

    act(() => {
      firstItem?.onClick?.()
    })

    expect(setSelectedModel).toHaveBeenCalledWith("openai:gpt-4o-mini")
  })

  it.each(["tldw:llama.cpp:", "llama.cpp:", "llamacpp:"])(
    "resolves setup identity %s without changing the selected model",
    (prefix) => {
      chatModelSettingsState.apiProvider = "openai"
      const setSelectedModel = vi.fn()
      const modelId = "Gemma-4:Q4_K_M"
      const configured = {
        id: modelId,
        model: `tldw:${modelId}`,
        nickname: "Gemma local",
        provider: "llama.cpp",
        is_configured: true,
        details: { capabilities: ["vision"] }
      }
      const { result } = renderHook(() => useModelSelector({
        composerModels: [{ ...configured, provider: "openai", nickname: "Other provider" }, configured],
        selectedModel: `${prefix}${modelId}`,
        setSelectedModel,
        navigate: vi.fn()
      }))
      expect(result.current.selectedModelMeta).toBe(configured)
      expect(result.current.apiModelLabel).toBe("LLAMACPP / Gemma local")
      expect(result.current.selectedModelKey).toBe(`llamacpp:${modelId}`)
      expect(setSelectedModel).not.toHaveBeenCalled()
    }
  )

  it.each(["tldw:llama.cpp:", "llama:"])("matches actual llama catalogue metadata for %s", (prefix) => {
    const [serverModel] = normalizeTldwModels({ models: [{
      provider: "llama", name: "Gemma-4:Q4_K_M",
      capabilities: { vision: true }, context_window: 8192
    }] })
    const model = {
      ...serverModel,
      model: `tldw:${serverModel.id}`,
      nickname: "Gemma local",
      details: { capabilities: ["vision"] }
    }
    const { result } = renderHook(() => useModelSelector({
      composerModels: [model],
      selectedModel: `${prefix}${serverModel.id}`,
      setSelectedModel: vi.fn(), navigate: vi.fn()
    }))
    expect(result.current.selectedModelMeta).toBe(model)
    expect(result.current.apiModelLabel).toBe("LLAMA / Gemma local")
    expect(result.current.modelCapabilities).toContain("vision")
    expect(result.current.modelContextLength).toBe(8192)
    expect(result.current.selectedModelKey).toBe("llama:Gemma-4:Q4_K_M")
  })

  it.each(["local-llm", "local"])("matches the existing %s catalogue provider alias", (provider) => {
    const model = { model: "Gemma:Q4", provider, nickname: "Local Gemma" }
    const { result } = renderHook(() => useModelSelector({
      composerModels: [model], selectedModel: "local-llm:Gemma:Q4",
      setSelectedModel: vi.fn(), navigate: vi.fn()
    }))
    expect(result.current.selectedModelMeta).toBe(model)
    expect(result.current.selectedModelKey).toBe("local:Gemma:Q4")
  })

  it("keeps the explicit setup provider while its catalogue is loading", () => {
    const { result } = renderHook(() => useModelSelector({
      composerModels: [],
      modelsLoading: true,
      selectedModel: "tldw:llama.cpp:Gemma-4:Q4_K_M",
      setSelectedModel: vi.fn(),
      navigate: vi.fn()
    }))
    expect(result.current.apiModelLabel).toBe("LLAMACPP / Gemma-4:Q4_K_M")
    expect(result.current.selectedModelKey).toBe("llamacpp:Gemma-4:Q4_K_M")
  })

  it("does not match a qualified model to a different-case ID or another provider", () => {
    const { result } = renderHook(() => useModelSelector({
      composerModels: [
        { model: "gemma-4", provider: "llama.cpp" },
        { model: "Gemma-4", provider: "openai" }
      ],
      selectedModel: "tldw:llama.cpp:Gemma-4",
      setSelectedModel: vi.fn(),
      navigate: vi.fn()
    }))
    expect(result.current.selectedModelMeta).toBeNull()
    expect(result.current.apiModelLabel).toBe("LLAMACPP / Gemma-4")
  })

  it("shows a loading affordance (not the connect-server error) while models load", () => {
    const { result } = renderHook(() =>
      useModelSelector({
        composerModels: [],
        selectedModel: null,
        setSelectedModel: vi.fn(),
        navigate: vi.fn(),
        modelsLoading: true
      })
    )

    const items = result.current.modelDropdownMenuItems
    expect(items).toHaveLength(1)
    expect(items[0]?.key).toBe("models-loading")

    render(<>{items[0]?.label}</>)
    expect(screen.getByTestId("model-loading")).toBeInTheDocument()
    expect(
      screen.queryByText(/Connect your server in Settings/i)
    ).not.toBeInTheDocument()
  })

  it("shows the connect-server error once loading finishes with no models", () => {
    const navigate = vi.fn()
    const { result } = renderHook(() =>
      useModelSelector({
        composerModels: [],
        selectedModel: null,
        setSelectedModel: vi.fn(),
        navigate,
        modelsLoading: false
      })
    )

    const items = result.current.modelDropdownMenuItems
    expect(items.some((item: any) => item?.key === "no-models")).toBe(true)
    expect(items.some((item: any) => item?.key === "open-model-settings")).toBe(true)

    const noModels = items.find((item: any) => item?.key === "no-models")
    render(<>{noModels?.label}</>)
    expect(
      screen.getByText(/No models available\. Connect your server in Settings\./i)
    ).toBeInTheDocument()
  })

  it.each(["provider", "localFirst"])("names known local providers in %s menu groups", (sortMode) => {
    storageSeed.values.set("modelSelectSortMode", sortMode)
    const { result } = renderHook(() => useModelSelector({
      composerModels: [
        { id: "shared", provider: "llama", is_configured: true },
        { id: "shared", provider: "custom_openai_api", is_configured: true },
        { id: "hosted", provider: "anthropic", is_configured: true }
      ],
      selectedModel: "custom_openai_api:shared",
      setSelectedModel: vi.fn(), navigate: vi.fn()
    }))
    const groups = result.current.modelDropdownMenuItems.filter(item => item?.type === "group")
    const local = groups.find(group => group.children.some((item: { key: string }) => item.key === "llama:shared"))
    render(<>{local?.label}</>)
    expect(screen.getByText("LLAMA")).toBeInTheDocument()
    if (sortMode === "localFirst") {
      expect(groups.indexOf(local)).toBeLessThan(groups.findIndex(group => group.key === "group-anthropic"))
    }
  })

  it("promotes current and recent configured models ahead of provider groups", () => {
    storageSeed.values.set("chatModelUsageByProviderModel", {
      "google:gemini-1.5-pro": {
        selectedCount: 2,
        lastSelectedAt: 200
      }
    })

    const { result } = renderHook(() =>
      useModelSelector({
        composerModels: [
          {
            model: "gpt-4o-mini",
            nickname: "Current",
            provider: "openai",
            is_configured: true
          },
          {
            model: "gemini-1.5-pro",
            nickname: "Recent",
            provider: "google",
            is_configured: true
          },
          {
            model: "claude-3-5-sonnet",
            nickname: "Provider grouped",
            provider: "anthropic",
            is_configured: true
          }
        ],
        selectedModel: "openai:gpt-4o-mini",
        setSelectedModel: vi.fn(),
        navigate: vi.fn()
      })
    )

    const groups = result.current.modelDropdownMenuItems.filter(
      (item: any) => item?.type === "group"
    )

    expect(groups[0]?.key).toBe("current-model")
    expect(groups[0]?.children?.[0]?.key).toBe("openai:gpt-4o-mini")
    expect(groups[1]?.key).toBe("recent-models")
    expect(groups[1]?.children?.[0]?.key).toBe("google:gemini-1.5-pro")
    expect(groups.some((group: any) => group.key === "group-anthropic")).toBe(true)
  })
})
