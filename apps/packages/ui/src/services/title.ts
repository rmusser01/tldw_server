import { pageAssistModel } from "@/models"
import { HumanMessage } from "@/types/messages"
import { removeReasoning } from "@/libs/reasoning"
import { coerceBoolean, defineSetting, getSetting, setSetting } from "@/services/settings/registry"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"
import { isRequestConfigScopeChangedError } from "@/services/tldw/service-prompt-scope-error"

const TITLE_GEN_ENABLED_SETTING = defineSetting(
    "titleGenEnabled",
    false,
    (value) => coerceBoolean(value, false)
)

// this prompt is copied from the OpenWebUI codebase
export const DEFAULT_TITLE_GEN_PROMPT = `Here is the query:

--------------

{{query}}

--------------

Create a concise, 3-5 word phrase as a title for the previous query. Avoid quotation marks or special formatting. RESPOND ONLY WITH THE TITLE TEXT. ANSWER USING THE SAME LANGUAGE AS THE QUERY.


Examples of titles:

Stellar Achievement Celebration
Family Bonding Activities
🇫🇷 Voyage à Paris
🍜 Receta de Ramen Casero
Shakespeare Analyse Literarische
日本の春祭り体験
Древнегреческая Философия Обзор

Response:`


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

const throwIfAborted = (signal?: AbortSignal): void => {
    if (!signal?.aborted) return
    const error = new Error("Request scope changed")
    error.name = "AbortError"
    throw error
}

export const generateTitle = async (
    model: string,
    query: string,
    fallBackTitle: string,
    options: TitleGenerationOptions = {}
) => {

    throwIfAborted(options.signal)

    const isEnabled = await isTitleGenEnabled()
    throwIfAborted(options.signal)

    if (!isEnabled) {
        return fallBackTitle
    }

    try {
        const titleModel = await pageAssistModel({
            model,
            toolChoice: "none",
            saveToDb: false,
            ...(options.requestScope
                ? { requestScope: options.requestScope }
                : {})
        })
        throwIfAborted(options.signal)

        const prompt = DEFAULT_TITLE_GEN_PROMPT.replace("{{query}}", query)

        const messages = [new HumanMessage({ content: prompt })]
        const title = options.signal
            ? await titleModel.invoke(messages, { signal: options.signal })
            : await titleModel.invoke(messages)
        throwIfAborted(options.signal)

        return removeReasoning(title.content.toString())
    } catch (error) {
        if (options.signal?.aborted ||
            (error as { name?: unknown } | null)?.name === "AbortError" ||
            isRequestConfigScopeChangedError(error)
        ) {
            throw error
        }
        console.error(`Error generating title: ${error}`)
        return fallBackTitle
    }
}
