import { useMutation } from "@tanstack/react-query"
import { tldwClient } from "@/services/tldw"
import { useDocumentOperation } from "./useDocumentOperation"

export interface TranslateParams {
  text: string
  targetLanguage?: string
  model?: string
  provider?: string
  signal?: AbortSignal
}

export interface TranslateResult {
  translated_text: string
  target_language: string
  model_used: string
  detected_source_language?: string
}

/**
 * Hook for translating text via the translation API.
 *
 * @returns Mutation for translating text
 *
 * @example
 * const { mutateAsync: translate, isPending } = useTranslate()
 *
 * const result = await translate({
 *   text: "Hello world",
 *   targetLanguage: "Spanish"
 * })
 */
export function useTranslate() {
  const operation = useDocumentOperation()
  return useMutation<TranslateResult, Error, TranslateParams>({
    mutationFn: async ({ text, targetLanguage = "English", model, provider, signal }) => {
      if (!operation?.isCurrent()) throw new Error("Document access changed")
      return await tldwClient.translate(text, targetLanguage, {
        model, provider, requestScope: operation.requestScope,
        signal: signal ? AbortSignal.any([signal, operation.signal]) : operation.signal
      })
    }
  })
}
