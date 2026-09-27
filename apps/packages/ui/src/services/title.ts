import { pageAssistModel } from "@/models"
import { HumanMessage } from "@/types/messages"
import { removeReasoning } from "@/libs/reasoning"
import { coerceBoolean, defineSetting, getSetting, setSetting } from "@/services/settings/registry"
import type {
    ServicePromptRequestScope
} from "@/services/tldw/domains/service-prompts"
import {
    loadServicePromptSnapshot,
    renderServicePromptPart
} from "@/services/service-prompts"
import { LEGACY_SERVICE_PROMPT_DEFAULTS } from "@/services/tldw-server"
import {
    createServicePromptScopeChangedError,
    isRequestConfigScopeChangedError
} from "@/services/tldw/service-prompt-scope-error"

const TITLE_GEN_ENABLED_SETTING = defineSetting(
    "titleGenEnabled",
    false,
    (value) => coerceBoolean(value, false)
)

export const DEFAULT_TITLE_GEN_PROMPT =
    LEGACY_SERVICE_PROMPT_DEFAULTS["chat.title.generation"].user_template


export const isTitleGenEnabled = async () => {
    return await getSetting(TITLE_GEN_ENABLED_SETTING)
}

export const setTitleGenEnabled = async (enabled: boolean) => {
    await setSetting(TITLE_GEN_ENABLED_SETTING, enabled)
}


type TitleGenerationOptions = {
    signal?: AbortSignal
    requestScope?: ServicePromptRequestScope
}

type TitleGenerationStage =
    | "settings"
    | "snapshot"
    | "render"
    | "model"
    | "invoke"
    | "response"

const throwIfAborted = (signal?: AbortSignal): void => {
    if (!signal?.aborted) return
    const error = new Error("Request scope changed")
    error.name = "AbortError"
    throw error
}

const throwIfTitleRequestInvalidated = (
    snapshot: Awaited<ReturnType<typeof loadServicePromptSnapshot>>,
    signal?: AbortSignal
): void => {
    if (snapshot.scopeInvalidatedSignal.aborted) {
        throw createServicePromptScopeChangedError()
    }
    throwIfAborted(signal)
}

export const generateTitle = async (
    model: string,
    query: string,
    fallBackTitle: string,
    options: TitleGenerationOptions = {}
) => {
    let snapshot: Awaited<ReturnType<typeof loadServicePromptSnapshot>> | null = null
    let stage: TitleGenerationStage = "settings"
    try {
        throwIfAborted(options.signal)
        const isEnabled = await isTitleGenEnabled()
        throwIfAborted(options.signal)
        if (!isEnabled) return fallBackTitle

        stage = "snapshot"
        snapshot = await loadServicePromptSnapshot(
            ["chat.title.generation"],
            { signal: options.signal, requestScope: options.requestScope }
        )
        stage = "render"
        const promptConfig = snapshot.definitions["chat.title.generation"]
        if (!promptConfig) {
            throw new Error("Conversation title prompt is unavailable.")
        }
        const prompt = renderServicePromptPart(
            promptConfig.definition,
            "user_template",
            promptConfig.parts.user_template,
            { query }
        )
        stage = "model"
        const titleModel = await pageAssistModel({
            model,
            toolChoice: "none",
            saveToDb: false,
            requestScope: snapshot.requestScope
        })
        throwIfTitleRequestInvalidated(snapshot, options.signal)
        stage = "invoke"
        const title = await titleModel.invoke(
            [new HumanMessage({ content: prompt })],
            { signal: snapshot.scopeSignal }
        )
        throwIfTitleRequestInvalidated(snapshot, options.signal)

        stage = "response"
        return removeReasoning(title.content.toString())
    } catch (error) {
        if (snapshot?.scopeInvalidatedSignal.aborted) {
            throw createServicePromptScopeChangedError()
        }
        if (options.signal?.aborted ||
            (error as { name?: unknown } | null)?.name === "AbortError" ||
            isRequestConfigScopeChangedError(error)
        ) {
            throw error
        }
        console.error("Error generating title", { stage })
        return fallBackTitle
    } finally {
        snapshot?.release()
    }
}
