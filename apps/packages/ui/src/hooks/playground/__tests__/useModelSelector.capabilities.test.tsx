import React from "react"
import { act, render, renderHook, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useModelSelector } from "../useModelSelector"

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
})
