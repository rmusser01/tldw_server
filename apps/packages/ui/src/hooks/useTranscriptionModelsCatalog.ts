import * as React from "react"
import { useTranslation } from "react-i18next"

import { tldwClient } from "@/services/tldw/TldwApiClient"
import { isTimeoutLikeError } from "@/utils/request-timeout"

type UseTranscriptionModelsCatalogOptions = {
  activeModel?: string
  autoRetryOnFailureCount?: number
  defaultModel?: string
  enabled?: boolean
  onInitialModel?: (model: string) => void
  warnLabel?: string
}

type UseTranscriptionModelsCatalogResult = {
  serverModels: string[]
  serverModelsLoading: boolean
  serverModelsError: string | null
  retryServerModels: () => void
}

/**
 * Loads server transcription models with the same retry semantics used by the
 * speech playground, so STT and speech surfaces stay aligned.
 */
export function useTranscriptionModelsCatalog(
  options: UseTranscriptionModelsCatalogOptions = {}
): UseTranscriptionModelsCatalogResult {
  const { t } = useTranslation(["playground"])
  const {
    activeModel,
    autoRetryOnFailureCount = 0,
    defaultModel,
    enabled = true,
    onInitialModel,
    warnLabel
  } = options

  const [serverModels, setServerModels] = React.useState<string[]>([])
  const [serverModelsLoading, setServerModelsLoading] = React.useState(true)
  const [serverModelsError, setServerModelsError] = React.useState<string | null>(
    null
  )
  const [modelsLoadAttempt, setModelsLoadAttempt] = React.useState(0)

  React.useEffect(() => {
    let cancelled = false

    if (!enabled) {
      setServerModelsLoading(false)
      setServerModelsError(null)
      return () => {
        cancelled = true
      }
    }

    const fetchModels = async () => {
      setServerModelsLoading(true)
      setServerModelsError(null)
      let lastError: unknown = null
      const maxAttempts = Math.max(1, autoRetryOnFailureCount + 1)

      for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
        try {
          const res = await tldwClient.getTranscriptionModels({
            timeoutMs: 10_000
          })
          const all = Array.isArray(res?.all_models) ? (res.all_models as string[]) : []
          if (!cancelled && all.length > 0) {
            const unique = Array.from(new Set(all)).sort()
            setServerModels(unique)
            if (onInitialModel && !activeModel) {
              const initial =
                defaultModel && unique.includes(defaultModel) ? defaultModel : unique[0]
              onInitialModel(initial)
            }
          }
          lastError = null
          break
        } catch (e) {
          lastError = e
          if (attempt < maxAttempts - 1) {
            continue
          }
        }
      }

      if (lastError && !cancelled) {
        setServerModelsError(
          isTimeoutLikeError(lastError)
            ? (t(
                "playground:stt.modelsTimeout",
                "Model list took longer than 10 seconds. Check server health and retry."
              ) as string)
            : (t(
                "playground:stt.modelsLoadErrorDesc",
                "Unable to load transcription models. Retry or check server settings."
              ) as string)
        )
        if ((import.meta as any)?.env?.DEV && warnLabel) {
          // eslint-disable-next-line no-console
          console.warn(
            `Failed to load transcription models for ${warnLabel}`,
            lastError
          )
        }
      }
      if (!cancelled) {
        setServerModelsLoading(false)
      }
    }

    void fetchModels()

    return () => {
      cancelled = true
    }
  }, [
    activeModel,
    autoRetryOnFailureCount,
    defaultModel,
    enabled,
    modelsLoadAttempt,
    onInitialModel,
    t,
    warnLabel
  ])

  const retryServerModels = React.useCallback(() => {
    setModelsLoadAttempt((prev) => prev + 1)
  }, [])

  return {
    serverModels,
    serverModelsLoading,
    serverModelsError,
    retryServerModels
  }
}
