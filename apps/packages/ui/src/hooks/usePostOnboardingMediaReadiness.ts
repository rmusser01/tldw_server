import React from "react"

import {
  tldwClient,
  type TldwConfig
} from "@/services/tldw/TldwApiClient"

export type PostOnboardingMediaReadinessStatus =
  | "checking"
  | "ready"
  | "needs_config"
  | "error"

export type PostOnboardingMediaReadinessState = {
  status: PostOnboardingMediaReadinessStatus
  config: TldwConfig | null
  errorMessage: string | null
}

const toErrorMessage = (error: unknown): string => {
  if (error instanceof Error && error.message.trim()) return error.message
  if (typeof error === "string" && error.trim()) return error
  return "Unable to verify media access."
}

const isAuthOrConfigError = (error: unknown): boolean => {
  const message = toErrorMessage(error).toLowerCase()
  return (
    message.includes("api key") ||
    message.includes("not configured") ||
    message.includes("not authenticated") ||
    message.includes("unauthorized") ||
    message.includes("forbidden") ||
    message.includes("401") ||
    message.includes("403")
  )
}

const hasRequiredAuth = (config: TldwConfig | null): boolean => {
  if (!config || !String(config.serverUrl || "").trim()) return false
  if (config.authMode === "multi-user") {
    return Boolean(String(config.accessToken || "").trim())
  }
  return Boolean(String(config.apiKey || "").trim())
}

const getCurrentOrigin = (): string => {
  if (typeof window === "undefined") return ""
  return String(window.location?.origin || "").trim()
}

export function usePostOnboardingMediaReadiness(
  enabled: boolean
): PostOnboardingMediaReadinessState & {
  recoverWithApiKey: (apiKey: string) => Promise<void>
  retry: () => Promise<void>
} {
  const [state, setState] =
    React.useState<PostOnboardingMediaReadinessState>({
      status: "checking",
      config: null,
      errorMessage: null
    })

  const checkReadiness = React.useCallback(async () => {
    setState((current) => ({
      ...current,
      status: "checking",
      errorMessage: null
    }))

    let config: TldwConfig | null = null
    try {
      config = await tldwClient.getConfig().catch(() => null)
      if (!hasRequiredAuth(config)) {
        setState({
          status: "needs_config",
          config,
          errorMessage: null
        })
        return
      }

      await tldwClient.listMedia({ results_per_page: 1 })
      setState({
        status: "ready",
        config,
        errorMessage: null
      })
    } catch (error) {
      setState({
        status: isAuthOrConfigError(error) ? "needs_config" : "error",
        config,
        errorMessage: toErrorMessage(error)
      })
    }
  }, [])

  React.useEffect(() => {
    if (!enabled) return
    let cancelled = false
    const run = async () => {
      await checkReadiness()
      if (cancelled) return
    }
    void run()
    return () => {
      cancelled = true
    }
  }, [checkReadiness, enabled])

  const recoverWithApiKey = React.useCallback(
    async (apiKey: string) => {
      const trimmedKey = apiKey.trim()
      if (!trimmedKey) {
        throw new Error("Enter your single-user API key.")
      }

      const currentConfig = await tldwClient.getConfig().catch(() => null)
      const serverUrl = String(
        currentConfig?.serverUrl || state.config?.serverUrl || getCurrentOrigin()
      ).trim()
      if (!serverUrl) {
        throw new Error("Server URL is missing. Restart quickstart and try again.")
      }

      await tldwClient.updateConfig({
        serverUrl,
        authMode: "single-user",
        apiKey: trimmedKey
      })
      await checkReadiness()
    },
    [checkReadiness, state.config?.serverUrl]
  )

  return {
    ...state,
    recoverWithApiKey,
    retry: checkReadiness
  }
}
