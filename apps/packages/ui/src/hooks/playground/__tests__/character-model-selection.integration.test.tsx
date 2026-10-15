import { buildCockpitProviderRouteSummary } from "@/components/Option/Playground/playground-cockpit-summaries"
import { validateNativeCharacterConfiguration } from "@/hooks/chat/native-character-preflight"
import { useStoreChatModelSettings } from "@/store/model"
import {
  buildCharacterChatReadiness,
  buildChatModelUsability
} from "@/utils/chat-model-availability"
import { act, renderHook } from "@testing-library/react"
import { readFileSync } from "node:fs"
import path from "node:path"
import React from "react"
import ts from "typescript"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useModelSelector } from "../useModelSelector"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback ?? key
  })
}))
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, initial: unknown) => [
    ...React.useState(initial),
    { isLoading: false }
  ]
}))
vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))
vi.mock("@/components/Common/ProviderIcon", () => ({
  ProviderIcons: () => null
}))
vi.mock("@/services/tldw", () => ({
  tldwModels: { getProviderDisplayName: (provider: string) => provider }
}))

// Exercise the actual producer expressions without mounting unrelated cockpit
// services. The shared hook, scoped store and readiness evaluator remain real.
const expression = (
  file: string,
  match: (node: ts.CallExpression) => boolean
) => {
  const ast = ts.createSourceFile(
    file,
    readFileSync(path.resolve(__dirname, file), "utf8"),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX
  )
  let found: ts.CallExpression | undefined
  const visit = (node: ts.Node) => {
    if (ts.isCallExpression(node) && match(node)) found = node
    ts.forEachChild(node, visit)
  }
  visit(ast)
  if (!found) throw new Error(`Missing production adapter in ${file}`)
  return { node: found, ast }
}
const scope = expression(
  "../../../components/Option/Playground/PlaygroundForm.tsx",
  (node) =>
    node.expression.getText() === "React.useEffect" &&
    node.arguments[0]
      ?.getText()
      .includes("setActiveChatModelSettingsScope(") === true &&
    node.arguments[0]?.getText().includes("selectedModel") === true
)
const publishFullFormScope = new Function(
  "selectedModelKey",
  "selectedModelSettingsScope",
  "setActiveChatModelSettingsScope",
  `return (${scope.node.arguments[0].getText(scope.ast)})();`
)
const readiness = expression(
  "../../../components/Option/Playground/Playground.tsx",
  (node) => node.expression.getText() === "buildCharacterChatReadiness"
)
const evaluateRootReadiness = new Function(
  "buildCharacterChatReadiness",
  "serverReadinessState",
  "activeCharacterSelection",
  "selectedModel",
  "apiProvider",
  "readinessChatModels",
  "providerRouteSummary",
  "streaming",
  "isProcessing",
  "isLoading",
  "chatModelsFetching",
  "chatModelsError",
  `return ${readiness.node.getText(readiness.ast)};`
)
const ordinaryReadiness = expression(
  "../../../components/Option/Playground/Playground.tsx",
  (node) => node.expression.getText() === "buildChatModelUsability"
)
const evaluateOrdinaryRootReadiness = new Function(
  "buildChatModelUsability",
  "serverReadinessState",
  "selectedModel",
  "apiProvider",
  "readinessChatModels",
  "chatModelsFetching",
  "chatModelsError",
  `return ${ordinaryReadiness.node.getText(ordinaryReadiness.ast)};`
)
const recovery = expression(
  "../../../components/Option/Playground/Playground.tsx",
  (node) =>
    node.expression.getText() === "React.useCallback" &&
    node.arguments[0]?.getText().includes("openModelSelector({") === true
)
const recoveryCallback = ts.transpile(
  `const callback = (${recovery.node.arguments[0].getText(recovery.ast)});`,
  { target: ts.ScriptTarget.ES2022 }
)
const makeRecoveryAction = new Function(
  "characterChatReadiness",
  "openModelSelector",
  "openCharacterSelectorFromReadiness",
  "openModelSettingsFromCockpit",
  "openServerSettingsFromCockpit",
  "refreshCharacterChatModels",
  `${recoveryCallback}; return callback;`
)
const models = [
  {
    model: "tldw:gpt-4o-mini",
    id: "gpt-4o-mini",
    provider: "openai",
    is_configured: true
  },
  {
    model: "tldw:gpt-4o-mini",
    id: "gpt-4o-mini",
    provider: "custom-openai-api",
    is_configured: false
  }
]

describe("production full chat model ownership and readiness adapters", () => {
  beforeEach(() => useStoreChatModelSettings.getState().reset())
  it("preserves a raw prompt override when the real hook publishes the full form scope", () => {
    const raw = "tldw:gpt-4o-mini"
    const prompt = "KEEP THIS PROMPT\n  exact spacing  "
    useStoreChatModelSettings.getState().setActiveSettingsScope(raw)
    useStoreChatModelSettings.getState().setSystemPrompt(prompt)
    const setSelectedModel = vi.fn()
    const { result, rerender } = renderHook(
      ({ catalog }) => {
        const selection = useModelSelector({
          composerModels: catalog,
          selectedModel: raw,
          setSelectedModel,
          navigate: vi.fn()
        })
        const publish =
          useStoreChatModelSettings.getState().setActiveSettingsScope
        React.useEffect(() => {
          publishFullFormScope(
            selection.selectedModelKey,
            selection.selectedModelSettingsScope,
            publish
          )
        }, [
          selection.selectedModelKey,
          selection.selectedModelSettingsScope,
          publish
        ])
        return selection
      },
      { initialProps: { catalog: [] as typeof models } }
    )
    for (const catalog of [[], models, [...models].reverse(), [models[0]]]) {
      rerender({ catalog })
      expect(result.current.selectedModelMeta).toBeNull()
      expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe(raw)
      expect(useStoreChatModelSettings.getState().systemPrompt).toBe(prompt)
    }
    expect(setSelectedModel).not.toHaveBeenCalled()
    expect(
      useStoreChatModelSettings
        .getState()
        .getEffectiveSettings("openai:gpt-4o-mini").systemPrompt
    ).toBeUndefined()
  })
  it.each(["tldw:gpt-4o-mini", "openai:gpt-4o-mini"])(
    "keeps a scoped provider override in its existing owned %s record",
    (initialScope) => {
      const state = useStoreChatModelSettings.getState()
      state.setActiveSettingsScope(initialScope)
      state.updateSettings({
        apiProvider: initialScope.startsWith("openai:")
          ? "anthropic"
          : "openai",
        systemPrompt: "KEEP",
        temperature: 0.25
      })
      const { result, rerender } = renderHook(
        ({ catalog }) =>
          useModelSelector({
            composerModels: catalog,
            selectedModel: "tldw:gpt-4o-mini",
            setSelectedModel: vi.fn(),
            navigate: vi.fn()
          }),
        { initialProps: { catalog: [] as typeof models } }
      )
      for (const catalog of [[], models, [...models].reverse()]) {
        rerender({ catalog })
        act(() =>
          publishFullFormScope(
            result.current.selectedModelKey,
            result.current.selectedModelSettingsScope,
            state.setActiveSettingsScope
          )
        )
        expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe(
          initialScope
        )
        expect(useStoreChatModelSettings.getState().systemPrompt).toBe("KEEP")
        expect(useStoreChatModelSettings.getState().temperature).toBe(0.25)
        expect(useStoreChatModelSettings.getState().apiProvider).toBe(
          initialScope.startsWith("openai:") ? "anthropic" : "openai"
        )
      }
    }
  )

  it("keeps qualified settings through catalog arrival and equivalent alias reordering", () => {
    const scope = "llamacpp:Gemma:Q4"
    const state = useStoreChatModelSettings.getState()
    state.setActiveSettingsScope(scope)
    state.updateSettings({
      systemPrompt: "KEEP LOCAL INSTRUCTIONS",
      temperature: 0.2
    })
    const aliases = [
      { model: "tldw:Gemma:Q4", provider: "llama" },
      { model: "tldw:Gemma:Q4", provider: "llama.cpp" }
    ]
    const { result, rerender } = renderHook(
      ({ catalog }) =>
        useModelSelector({
          composerModels: catalog,
          selectedModel: "llama_cpp:Gemma:Q4",
          setSelectedModel: vi.fn(),
          navigate: vi.fn()
        }),
      { initialProps: { catalog: [] as typeof aliases } }
    )
    for (const catalog of [[], aliases, [...aliases].reverse()]) {
      rerender({ catalog })
      act(() =>
        publishFullFormScope(
          result.current.selectedModelKey,
          result.current.selectedModelSettingsScope,
          state.setActiveSettingsScope
        )
      )
      expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe(
        scope
      )
      expect(useStoreChatModelSettings.getState().systemPrompt).toBe(
        "KEEP LOCAL INSTRUCTIONS"
      )
      expect(useStoreChatModelSettings.getState().temperature).toBe(0.2)
    }
  })

  it("keeps an exact opaque catalog choice in the production ordinary readiness adapter", () => {
    const catalog = [
      { model: "Model:Q4", provider: "chrome", is_configured: true }
    ]
    expect(
      evaluateOrdinaryRootReadiness(
        buildChatModelUsability,
        "ready",
        "chrome:Model:Q4",
        "openai",
        catalog
      )
    ).toMatchObject({ status: "ready", matchedProvider: "chrome" })
    expect(
      evaluateRootReadiness(
        buildCharacterChatReadiness,
        "ready",
        { id: 4 },
        "chrome:Model:Q4",
        null,
        catalog,
        {},
        false,
        false,
        false
      )
    ).toMatchObject({ reason: "explicit-model-provider-required" })
    const route = validateNativeCharacterConfiguration({
      temporary: false,
      clientToolCount: 0,
      settings: { apiProvider: "openai" },
      model: "chrome:Model:Q4"
    })
    expect(route).toMatchObject({
      provider: "openai",
      model: "chrome:Model:Q4"
    })
    expect(
      evaluateRootReadiness(
        buildCharacterChatReadiness,
        "ready",
        { id: 4 },
        "chrome:Model:Q4",
        "openai",
        catalog,
        {},
        false,
        false,
        false
      )
    ).toMatchObject({ reason: "selected-model-missing" })
    const nativeCatalog = [
      { model: route.model, provider: route.provider, is_configured: true }
    ]
    expect(
      evaluateRootReadiness(
        buildCharacterChatReadiness,
        "ready",
        { id: 4 },
        "chrome:Model:Q4",
        "openai",
        nativeCatalog,
        {},
        false,
        false,
        false
      )
    ).toMatchObject({ status: "ready" })
  })

  it("opens the model chooser for a missing provider and settings for other model blockers", () => {
    const chooser = vi.fn(),
      settings = vi.fn()
    for (const reason of [
      "explicit-model-provider-required",
      "provider-unconfigured"
    ]) {
      const action = makeRecoveryAction(
        { reason },
        chooser,
        vi.fn(),
        settings,
        vi.fn(),
        vi.fn()
      )
      action("open-model-settings")
    }
    expect(chooser).toHaveBeenCalledExactlyOnceWith({
      returnFocusSelector: "[data-testid='model-selector']"
    })
    expect(settings).toHaveBeenCalledOnce()
  })

  it.each([null, "custom-openai-api"])(
    "passes the actual raw selection/provider to native readiness (%s)",
    (provider) => {
      const selectedModel = "tldw:gpt-4o-mini"
      const result = evaluateRootReadiness(
        buildCharacterChatReadiness,
        "ready",
        { id: 4 },
        selectedModel,
        provider,
        models,
        buildCockpitProviderRouteSummary({
          selectedModel,
          selectedProvider: provider
        }),
        false,
        false,
        false
      )
      expect(result).toMatchObject({
        status: "blocked",
        missingRequirement: "chat-model",
        reason: provider
          ? "provider-unconfigured"
          : "explicit-model-provider-required"
      })
    }
  )
})
