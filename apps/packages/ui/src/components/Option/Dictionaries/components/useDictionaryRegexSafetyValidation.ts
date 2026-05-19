import React from "react"
import { useTranslation } from "react-i18next"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import {
  buildTimedEffectsPayload,
  extractRegexSafetyMessage,
  validateRegexPattern,
} from "./dictionaryEntryUtils"

type UseDictionaryRegexSafetyValidationParams = {
  dictionaryName?: string
}

type UseDictionaryRegexSafetyValidationState = {
  validateRegexWithServer: (entryDraft: any) => Promise<string | null>
}

export function useDictionaryRegexSafetyValidation({
  dictionaryName,
}: UseDictionaryRegexSafetyValidationParams): UseDictionaryRegexSafetyValidationState {
  const { t } = useTranslation(["option"])
  const localize = React.useCallback(
    (key: string, fallback: string) => t(key, fallback),
    [t]
  )

  const validateRegexWithServer = React.useCallback(
    async (entryDraft: any): Promise<string | null> => {
      const type = entryDraft?.type === "regex" ? "regex" : "literal"
      if (type !== "regex") return null

      const pattern =
        typeof entryDraft?.pattern === "string" ? entryDraft.pattern : ""
      const replacement =
        typeof entryDraft?.replacement === "string" ? entryDraft.replacement : ""

      if (!pattern.trim()) {
        return localize(
          "option:dictionaries.validation.patternRequired",
          "Pattern is required."
        )
      }

      const clientRegexError = validateRegexPattern(pattern, localize)
      if (clientRegexError) {
        return clientRegexError
      }

      const timedEffectsPayload = buildTimedEffectsPayload(entryDraft?.timed_effects)
      const validationEntry: Record<string, any> = {
        pattern,
        replacement,
        type: "regex",
        probability:
          typeof entryDraft?.probability === "number" &&
          Number.isFinite(entryDraft.probability)
            ? Math.min(1, Math.max(0, entryDraft.probability))
            : 1,
        enabled:
          typeof entryDraft?.enabled === "boolean" ? entryDraft.enabled : true,
        case_sensitive:
          typeof entryDraft?.case_sensitive === "boolean"
            ? entryDraft.case_sensitive
            : true,
        max_replacements:
          Number.isInteger(entryDraft?.max_replacements) &&
          entryDraft.max_replacements >= 0
            ? entryDraft.max_replacements
            : 0,
      }
      if (typeof entryDraft?.group === "string" && entryDraft.group.trim()) {
        validationEntry.group = entryDraft.group.trim()
      }
      if (timedEffectsPayload) {
        validationEntry.timed_effects = timedEffectsPayload
      }

      try {
        await tldwClient.initialize()
        const validationResult = await tldwClient.validateDictionary({
          data: {
            name: dictionaryName || "Entry validation",
            entries: [validationEntry],
          },
          schema_version: 1,
          strict: true,
        })
        return extractRegexSafetyMessage(validationResult)
      } catch (error: any) {
        return (
          error?.message ||
          localize(
            "option:dictionaries.validation.regexServerUnavailable",
            "Unable to validate regex pattern safety with server."
          )
        )
      }
    },
    [dictionaryName, localize]
  )

  return { validateRegexWithServer }
}
