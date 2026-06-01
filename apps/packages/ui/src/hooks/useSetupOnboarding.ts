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
  value instanceof Error
    ? value
    : new Error(String(value || "Setup request failed"))

type InitialSetupLoadResult = {
  nextState: FirstRunState | null
  nextMetadata: FirstRunMetadata | null
  firstError: Error | null
}

type InitialSetupCache = InitialSetupLoadResult & {
  stateLoaded: boolean
  metadataLoaded: boolean
}

type LoadOutcome<T> = {
  value: T | null
  error: Error | null
}

let initialSetupStatePromise: Promise<LoadOutcome<FirstRunState>> | null = null
let initialSetupMetadataPromise: Promise<LoadOutcome<FirstRunMetadata>> | null =
  null
let initialSetupSnapshot: InitialSetupCache | null = null

const rememberInitialSetupSnapshot = (
  patch: Partial<InitialSetupLoadResult> &
    Partial<Pick<InitialSetupCache, "stateLoaded" | "metadataLoaded">>
) => {
  const current = initialSetupSnapshot ?? {
    nextState: null,
    nextMetadata: null,
    firstError: null,
    stateLoaded: false,
    metadataLoaded: false
  }
  const hasNextState = Object.prototype.hasOwnProperty.call(patch, "nextState")
  const hasNextMetadata = Object.prototype.hasOwnProperty.call(
    patch,
    "nextMetadata"
  )
  const hasFirstError = Object.prototype.hasOwnProperty.call(
    patch,
    "firstError"
  )
  initialSetupSnapshot = {
    nextState: hasNextState ? (patch.nextState ?? null) : current.nextState,
    nextMetadata: hasNextMetadata
      ? (patch.nextMetadata ?? null)
      : current.nextMetadata,
    firstError: hasFirstError ? (patch.firstError ?? null) : current.firstError,
    stateLoaded: patch.stateLoaded ?? current.stateLoaded,
    metadataLoaded: patch.metadataLoaded ?? current.metadataLoaded
  }
}

const loadInitialSetupState = (): Promise<LoadOutcome<FirstRunState>> => {
  if (initialSetupSnapshot?.stateLoaded) {
    return Promise.resolve({
      value: initialSetupSnapshot.nextState,
      error: initialSetupSnapshot.nextState
        ? null
        : initialSetupSnapshot.firstError
    })
  }
  if (!initialSetupStatePromise) {
    initialSetupStatePromise = tldwClient
      .getFirstRunState()
      .then((nextState) => {
        rememberInitialSetupSnapshot({
          nextState,
          firstError: null,
          stateLoaded: true
        })
        return { value: nextState, error: null }
      })
      .catch((err) => {
        const error = toError(err)
        rememberInitialSetupSnapshot({
          firstError: error,
          stateLoaded: true
        })
        return { value: null, error }
      })
      .finally(() => {
        initialSetupStatePromise = null
      })
  }
  return initialSetupStatePromise
}

const loadInitialSetupMetadata = (): Promise<LoadOutcome<FirstRunMetadata>> => {
  if (initialSetupSnapshot?.metadataLoaded) {
    return Promise.resolve({
      value: initialSetupSnapshot.nextMetadata,
      error: initialSetupSnapshot.nextMetadata
        ? null
        : initialSetupSnapshot.firstError
    })
  }
  if (!initialSetupMetadataPromise) {
    initialSetupMetadataPromise = tldwClient
      .getFirstRunMetadata()
      .then((nextMetadata) => {
        rememberInitialSetupSnapshot({
          nextMetadata,
          firstError: null,
          metadataLoaded: true
        })
        return { value: nextMetadata, error: null }
      })
      .catch((err) => {
        const error = toError(err)
        rememberInitialSetupSnapshot({
          firstError: error,
          metadataLoaded: true
        })
        return { value: null, error }
      })
      .finally(() => {
        initialSetupMetadataPromise = null
      })
  }
  return initialSetupMetadataPromise
}

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
  const cachedInitialState = autoLoad
    ? (initialSetupSnapshot?.nextState ?? null)
    : null
  const cachedInitialMetadata = autoLoad
    ? (initialSetupSnapshot?.nextMetadata ?? null)
    : null
  const [state, setState] = React.useState<FirstRunState | null>(
    initialState ?? cachedInitialState
  )
  const [metadata, setMetadata] = React.useState<FirstRunMetadata | null>(
    initialMetadata ?? cachedInitialMetadata
  )
  const [providerCatalog, setProviderCatalog] = React.useState<
    SetupProviderCatalogEntry[]
  >([])
  const [audioRecommendations, setAudioRecommendations] = React.useState<
    AudioRecommendationsResponse["recommendations"]
  >([])
  const [loading, setLoading] = React.useState(
    autoLoad &&
      (!(initialState ?? cachedInitialState) ||
        !(initialMetadata ?? cachedInitialMetadata))
  )
  const [error, setError] = React.useState<Error | null>(
    autoLoad ? (initialSetupSnapshot?.firstError ?? null) : null
  )
  const autoLoadInFlightRef = React.useRef(false)

  const refresh = React.useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      let nextState: FirstRunState | null = null
      let firstError: Error | null = null
      try {
        nextState = await tldwClient.getFirstRunState()
        rememberInitialSetupSnapshot({
          nextState,
          firstError: null,
          stateLoaded: true
        })
        setState(nextState)
      } catch (err) {
        firstError = toError(err)
      }
      try {
        const nextMetadata = await tldwClient.getFirstRunMetadata()
        rememberInitialSetupSnapshot({
          nextMetadata,
          firstError: null,
          metadataLoaded: true
        })
        setMetadata(nextMetadata)
      } catch (err) {
        firstError = firstError ?? toError(err)
      }
      if (firstError) {
        rememberInitialSetupSnapshot({ firstError })
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
    if (autoLoadInFlightRef.current) return
    let mounted = true
    autoLoadInFlightRef.current = true
    setLoading(true)
    setError(null)
    loadInitialSetupState()
      .then((stateOutcome) => {
        if (!mounted) return
        if (stateOutcome.value) setState(stateOutcome.value)
        if (stateOutcome.error) setError(stateOutcome.error)
        return loadInitialSetupMetadata()
      })
      .then((metadataOutcome) => {
        if (!mounted || !metadataOutcome) return
        if (metadataOutcome.value) setMetadata(metadataOutcome.value)
        if (metadataOutcome.error) setError(metadataOutcome.error)
      })
      .finally(() => {
        autoLoadInFlightRef.current = false
        if (mounted) setLoading(false)
      })
    return () => {
      mounted = false
      autoLoadInFlightRef.current = false
    }
  }, [autoLoad])

  const adoptState = React.useCallback((nextState: FirstRunState) => {
    rememberInitialSetupSnapshot({
      nextState,
      firstError: null,
      stateLoaded: true
    })
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
      rememberInitialSetupSnapshot({
        nextState,
        firstError: null,
        stateLoaded: true
      })
      setState(nextState)
      return nextState
    },
    []
  )

  const skip = React.useCallback(async (payload: FirstRunSkipRequest = {}) => {
    const nextState = await tldwClient.skipFirstRun(payload)
    rememberInitialSetupSnapshot({
      nextState,
      firstError: null,
      stateLoaded: true
    })
    setState(nextState)
    return nextState
  }, [])

  const saveProvider = React.useCallback(
    async (
      payload: SetupProviderSaveRequest
    ): Promise<SetupProviderSaveResponse> => {
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
    async (
      payload: FirstChatVerifyRequest
    ): Promise<FirstChatVerifyResponse> => {
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
