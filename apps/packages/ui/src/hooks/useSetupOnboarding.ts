import * as React from "react"

import { tldwClient } from "@/services/tldw/TldwApiClient"
import type {
  AudioDefaultsRequest,
  AudioRecommendationsResponse,
  FirstChatVerifyRequest,
  FirstChatVerifyResponse,
  FirstRunCompleteRequest,
  FirstRunMetadata,
  FirstRunSkipRequest,
  FirstRunState,
  FirstRunStepUpdateRequest,
  IngestDefaultsRequest,
  OptionalAdvancedRequest,
  SetupCompleteResponse,
  SetupProviderCatalogEntry,
  SetupProviderSaveRequest,
  SetupProviderSaveResponse,
  SetupProviderValidationResponse
} from "@/types/setup-onboarding"

const toError = (value: unknown): Error =>
  value instanceof Error ? value : new Error(String(value || "Setup request failed"))

type UseSetupOnboardingOptions = {
  initialState?: FirstRunState | null
  initialMetadata?: FirstRunMetadata | null
  autoLoad?: boolean
}

export function useSetupOnboarding(options: UseSetupOnboardingOptions = {}) {
  const {
    initialState = null,
    initialMetadata = null,
    autoLoad = true
  } = options
  const [state, setState] = React.useState<FirstRunState | null>(initialState)
  const [metadata, setMetadata] = React.useState<FirstRunMetadata | null>(
    initialMetadata
  )
  const [providerCatalog, setProviderCatalog] = React.useState<
    SetupProviderCatalogEntry[]
  >([])
  const [audioRecommendations, setAudioRecommendations] = React.useState<
    AudioRecommendationsResponse["recommendations"]
  >([])
  const [loading, setLoading] = React.useState(
    autoLoad && (!initialState || !initialMetadata)
  )
  const [error, setError] = React.useState<Error | null>(null)

  const refresh = React.useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      let nextState: FirstRunState | null = null
      let firstError: Error | null = null
      try {
        nextState = await tldwClient.getFirstRunState()
        setState(nextState)
      } catch (err) {
        firstError = toError(err)
      }
      try {
        setMetadata(await tldwClient.getFirstRunMetadata())
      } catch (err) {
        firstError = firstError ?? toError(err)
      }
      if (firstError) {
        setError(firstError)
        if (!nextState) {
          throw firstError
        }
      }
      return nextState
    } finally {
      setLoading(false)
    }
  }, [])

  React.useEffect(() => {
    if (!autoLoad || (state && metadata)) return
    let mounted = true
    let nextState: FirstRunState | null = null
    let firstError: Error | null = null
    setLoading(true)
    setError(null)
    tldwClient
      .getFirstRunState()
      .then((loadedState) => {
        nextState = loadedState
        if (mounted) setState(loadedState)
      })
      .catch((err) => {
        firstError = toError(err)
      })
      .then(() => tldwClient.getFirstRunMetadata())
      .then((nextMetadata) => {
        if (mounted) setMetadata(nextMetadata)
      })
      .catch((err) => {
        firstError = firstError ?? toError(err)
      })
      .finally(() => {
        if (mounted && firstError) setError(firstError)
        if (mounted) setLoading(false)
      })
    return () => {
      mounted = false
    }
  }, [autoLoad, metadata, state])

  const adoptState = React.useCallback((nextState: FirstRunState) => {
    setState(nextState)
  }, [])

  const loadProviderCatalog = React.useCallback(async () => {
    const response = await tldwClient.getSetupProviderCatalog()
    setProviderCatalog(response.providers)
    return response.providers
  }, [])

  const saveStep = React.useCallback(
    async (payload: FirstRunStepUpdateRequest) => {
      const nextState = await tldwClient.updateFirstRunState(payload)
      setState(nextState)
      return nextState
    },
    []
  )

  const skip = React.useCallback(async (payload: FirstRunSkipRequest = {}) => {
    const nextState = await tldwClient.skipFirstRun(payload)
    setState(nextState)
    return nextState
  }, [])

  const saveProvider = React.useCallback(
    async (payload: SetupProviderSaveRequest): Promise<SetupProviderSaveResponse> => {
      const response = await tldwClient.saveSetupProvider(payload)
      await refresh().catch(() => undefined)
      return response
    },
    [refresh]
  )

  const validateProvider = React.useCallback(
    async (
      payload: SetupProviderSaveRequest
    ): Promise<SetupProviderValidationResponse> => {
      return await tldwClient.validateSetupProvider(payload)
    },
    []
  )

  const saveIngestDefaults = React.useCallback(
    async (payload: IngestDefaultsRequest) => {
      const response = await tldwClient.saveIngestDefaults(payload)
      await refresh().catch(() => undefined)
      return response
    },
    [refresh]
  )

  const saveAudioDefaults = React.useCallback(
    async (payload: AudioDefaultsRequest) => {
      const response = await tldwClient.saveAudioDefaults(payload)
      await refresh().catch(() => undefined)
      return response
    },
    [refresh]
  )

  const loadAudioRecommendations = React.useCallback(async () => {
    const response = await tldwClient.getSetupAudioRecommendations()
    setAudioRecommendations(response.recommendations)
    return response.recommendations
  }, [])

  const saveOptionalAdvanced = React.useCallback(
    async (payload: OptionalAdvancedRequest) => {
      const response = await tldwClient.saveOptionalAdvanced(payload)
      await refresh().catch(() => undefined)
      return response
    },
    [refresh]
  )

  const verifyFirstChat = React.useCallback(
    async (payload: FirstChatVerifyRequest): Promise<FirstChatVerifyResponse> => {
      const response = await tldwClient.verifyFirstRunChat(payload)
      await refresh().catch(() => undefined)
      return response
    },
    [refresh]
  )

  const complete = React.useCallback(
    async (
      payload: FirstRunCompleteRequest = {}
    ): Promise<SetupCompleteResponse> => {
      const response = await tldwClient.completeFirstRun(payload)
      await refresh().catch(() => undefined)
      return response
    },
    [refresh]
  )

  return {
    state,
    metadata,
    providerCatalog,
    audioRecommendations,
    loading,
    error,
    refresh,
    adoptState,
    loadProviderCatalog,
    saveStep,
    skip,
    saveProvider,
    validateProvider,
    saveIngestDefaults,
    saveAudioDefaults,
    loadAudioRecommendations,
    saveOptionalAdvanced,
    verifyFirstChat,
    complete
  }
}
