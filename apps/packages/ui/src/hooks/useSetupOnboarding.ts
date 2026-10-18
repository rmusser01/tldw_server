import * as React from "react"

import type { SetupRequestOptions } from "@/services/tldw/domains/setup-onboarding"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { resolveBrowserRequestTransport } from "@/services/tldw/request-core"
import { resolveAdvancedRequestTransportGuard } from "@/services/tldw/browser-networking"
import { deriveConnectionAuthorityId } from "@/services/chat-surface-scope"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"
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
  McpToolsApplyRequest,
  McpToolsApplyResponse,
  McpToolsCatalogResponse,
  McpToolsValidateRequest,
  McpToolsValidateResponse,
  OptionalAdvancedRequest,
  SetupCompleteResponse,
  SetupProviderCatalogEntry,
  SetupProviderSaveRequest,
  SetupProviderSaveResponse,
  SetupProviderValidationResponse,
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

type InitialSetupOwner = {
  key: string
  controller: AbortController
  requestOptions: SetupRequestOptions
  snapshot: InitialSetupCache | null
  statePromise: Promise<LoadOutcome<FirstRunState>> | null
  metadataPromise: Promise<LoadOutcome<FirstRunMetadata>> | null
}
let initialSetupOwner: InitialSetupOwner | null = null

const rememberInitialSetupSnapshot = (
  owner: InitialSetupOwner,
  patch: Partial<InitialSetupLoadResult> &
    Partial<Pick<InitialSetupCache, "stateLoaded" | "metadataLoaded">>
) => {
  const current = owner.snapshot ?? {
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
  owner.snapshot = {
    nextState: hasNextState ? (patch.nextState ?? null) : current.nextState,
    nextMetadata: hasNextMetadata
      ? (patch.nextMetadata ?? null)
      : current.nextMetadata,
    firstError: hasFirstError ? (patch.firstError ?? null) : current.firstError,
    stateLoaded: patch.stateLoaded ?? current.stateLoaded,
    metadataLoaded: patch.metadataLoaded ?? current.metadataLoaded
  }
}

const loadInitialSetupState = (owner: InitialSetupOwner): Promise<LoadOutcome<FirstRunState>> => {
  if (owner.snapshot?.stateLoaded) {
    return Promise.resolve({
      value: owner.snapshot.nextState,
      error: owner.snapshot.nextState
        ? null
        : owner.snapshot.firstError
    })
  }
  if (!owner.statePromise) {
    owner.statePromise = tldwClient
      .getFirstRunState(owner.requestOptions)
      .then((nextState) => {
        rememberInitialSetupSnapshot(owner, {
          nextState,
          firstError: null,
          stateLoaded: true
        })
        return { value: nextState, error: null }
      })
      .catch((err) => {
        const error = toError(err)
        rememberInitialSetupSnapshot(owner, {
          firstError: error,
          stateLoaded: true
        })
        return { value: null, error }
      })
      .finally(() => {
        owner.statePromise = null
      })
  }
  return owner.statePromise
}

const loadInitialSetupMetadata = (owner: InitialSetupOwner): Promise<LoadOutcome<FirstRunMetadata>> => {
  if (owner.snapshot?.metadataLoaded) {
    return Promise.resolve({
      value: owner.snapshot.nextMetadata,
      error: owner.snapshot.nextMetadata
        ? null
        : owner.snapshot.firstError
    })
  }
  if (!owner.metadataPromise) {
    owner.metadataPromise = tldwClient
      .getFirstRunMetadata(owner.requestOptions)
      .then((nextMetadata) => {
        rememberInitialSetupSnapshot(owner, {
          nextMetadata,
          firstError: null,
          metadataLoaded: true
        })
        return { value: nextMetadata, error: null }
      })
      .catch((err) => {
        const error = toError(err)
        rememberInitialSetupSnapshot(owner, {
          firstError: error,
          metadataLoaded: true
        })
        return { value: null, error }
      })
      .finally(() => {
        owner.metadataPromise = null
      })
  }
  return owner.metadataPromise
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
  const [state, setState] = React.useState<FirstRunState | null>(initialState)
  const [metadata, setMetadata] = React.useState<FirstRunMetadata | null>(initialMetadata)
  const [providerCatalog, setProviderCatalog] = React.useState<SetupProviderCatalogEntry[]>([])
  const [mcpToolsCatalog, setMcpToolsCatalog] = React.useState<McpToolsCatalogResponse | null>(null)
  const [audioRecommendations, setAudioRecommendations] = React.useState<AudioRecommendationsResponse["recommendations"]>([])
  const [loading, setLoading] = React.useState(autoLoad)
  const [error, setError] = React.useState<Error | null>(null)
  const [connectionConfigured, setConnectionConfigured] = React.useState(false)
  const [connectionChecking, setConnectionChecking] = React.useState(true)
  const ownerRef = React.useRef<InitialSetupOwner | null>(null)
  const revisionRef = React.useRef(0)
  const connectionReadyRef = React.useRef<Promise<InitialSetupOwner | null> | null>(null)
  const initialDataRef = React.useRef({ initialState, initialMetadata })
  initialDataRef.current = { initialState, initialMetadata }

  const isCurrentOwner = React.useCallback((owner: InitialSetupOwner) =>
    ownerRef.current === owner && initialSetupOwner === owner, [])
  const clearSetup = React.useCallback(() => {
    setState(null); setMetadata(null); setError(null)
    setProviderCatalog([]); setMcpToolsCatalog(null); setAudioRecommendations([])
  }, [])

  const resolveConnection = React.useCallback(async (reload = false, retire = false) => {
    const revision = ++revisionRef.current
    if (retire) {
      // A connection authority event retires even a return to the same connection.
      initialSetupOwner?.controller.abort()
      initialSetupOwner = null
      ownerRef.current = null
      clearSetup()
    }
    setConnectionChecking(true)
    setLoading(autoLoad)
    try {
      if (reload) await tldwClient.initialize()
      const config = await tldwClient.getConfig()
      if (revisionRef.current !== revision) return null
      const transport = resolveBrowserRequestTransport({ config, path: "/api/v1/setup/first-run/state" })
      const { isUnconfigured } = resolveAdvancedRequestTransportGuard({
        transport, hasConfiguredServerUrl: Boolean(config?.serverUrl), isAbsolute: false
      })
      setConnectionConfigured(!isUnconfigured)
      if (isUnconfigured) {
        initialSetupOwner?.controller.abort()
        initialSetupOwner = null
        ownerRef.current = null
        clearSetup()
        return null
      }
      const key = JSON.stringify([transport, deriveConnectionAuthorityId(config)])
      if (initialSetupOwner?.key !== key) {
        if (ownerRef.current) clearSetup()
        initialSetupOwner?.controller.abort()
        const controller = new AbortController()
        initialSetupOwner = {
          key, controller,
          requestOptions: { configSnapshot: structuredClone(config), abortSignal: controller.signal },
          snapshot: null, statePromise: null, metadataPromise: null
        }
      }
      const owner = initialSetupOwner!
      ownerRef.current = owner
      return owner
    } catch (err) {
      if (revisionRef.current === revision) setError(toError(err))
      return null
    } finally {
      if (revisionRef.current === revision) {
        setConnectionChecking(false)
        setLoading(false)
      }
    }
  }, [autoLoad, clearSetup])

  const loadSetup = React.useCallback(async (owner: InitialSetupOwner, fresh = false) => {
    if (!isCurrentOwner(owner)) return null
    if (fresh) owner.snapshot = null
    setLoading(true)
    setError(null)
    try {
      const stateOutcome = await loadInitialSetupState(owner)
      if (!isCurrentOwner(owner)) return null
      if (stateOutcome.value) setState(stateOutcome.value)
      const metadataOutcome = await loadInitialSetupMetadata(owner)
      if (!isCurrentOwner(owner)) return null
      if (metadataOutcome.value) setMetadata(metadataOutcome.value)
      const firstError = stateOutcome.error ?? metadataOutcome.error
      rememberInitialSetupSnapshot(owner, { firstError })
      setError(firstError)
      if (fresh && firstError && !stateOutcome.value) throw firstError
      return stateOutcome.value
    } finally {
      if (isCurrentOwner(owner)) setLoading(false)
    }
  }, [isCurrentOwner])

  const refresh = React.useCallback(async () => {
    const owner = await resolveConnection()
    if (!owner) return null
    return loadSetup(owner, true)
  }, [loadSetup, resolveConnection])

  React.useEffect(() => {
    let mounted = true
    const connect = (reload = false, retire = false) => {
      connectionReadyRef.current = resolveConnection(reload, retire).then(async owner => {
        if (!mounted || !owner || !isCurrentOwner(owner)) return null
        if (autoLoad || reload) await loadSetup(owner)
        else {
          setState(initialDataRef.current.initialState)
          setMetadata(initialDataRef.current.initialMetadata)
        }
        return owner
      })
    }
    connect()
    const stop = watchChatAccountChanges(invalidated => { connect(true, invalidated) })
    return () => {
      mounted = false
      revisionRef.current += 1
      ownerRef.current = null
      stop()
    }
  }, [autoLoad, isCurrentOwner, loadSetup, resolveConnection])

  const requireOwner = React.useCallback(async () => {
    const ready = connectionReadyRef.current
    const originalOwner = ownerRef.current
    const resolvedOwner = await ready
    const owner = originalOwner ?? resolvedOwner
    if (!owner) throw new Error("Connect your tldw server before continuing setup.")
    if (!isCurrentOwner(owner)) throw new Error("The setup connection changed. Review the current server before continuing.")
    return owner
  }, [isCurrentOwner])
  const assertOwner = React.useCallback((owner: InitialSetupOwner) => {
    if (!isCurrentOwner(owner)) throw new Error("The setup connection changed. Review the current server before continuing.")
  }, [isCurrentOwner])

  const adoptState = React.useCallback((nextState: FirstRunState) => {
    const owner = ownerRef.current
    if (!owner || !isCurrentOwner(owner)) return
    rememberInitialSetupSnapshot(owner, {
      nextState,
      firstError: null,
      stateLoaded: true
    })
    setState(nextState)
  }, [isCurrentOwner])

  const loadProviderCatalog = React.useCallback(async () => {
    const owner = await requireOwner()
    const response = await tldwClient.getSetupProviderCatalog(owner.requestOptions)
    assertOwner(owner)
    setProviderCatalog(response.providers)
    return response.providers
  }, [assertOwner, requireOwner])

  const saveStep = React.useCallback(
    async (payload: FirstRunStepUpdateRequest) => {
      const owner = await requireOwner()
      const nextState = await tldwClient.updateFirstRunState(payload, owner.requestOptions)
      assertOwner(owner)
      rememberInitialSetupSnapshot(owner, {
        nextState,
        firstError: null,
        stateLoaded: true
      })
      setState(nextState)
      return nextState
    },
    [assertOwner, requireOwner]
  )

  const skip = React.useCallback(async (payload: FirstRunSkipRequest = {}) => {
    const owner = await requireOwner()
    const nextState = await tldwClient.skipFirstRun(payload, owner.requestOptions)
    assertOwner(owner)
    rememberInitialSetupSnapshot(owner, {
      nextState,
      firstError: null,
      stateLoaded: true
    })
    setState(nextState)
    return nextState
  }, [assertOwner, requireOwner])

  const saveProvider = React.useCallback(
    async (
      payload: SetupProviderSaveRequest
    ): Promise<SetupProviderSaveResponse> => {
      const owner = await requireOwner()
      const response = await tldwClient.saveSetupProvider(payload, owner.requestOptions)
      assertOwner(owner)
      await refresh().catch(() => undefined)
      assertOwner(owner)
      return response
    },
    [assertOwner, refresh, requireOwner]
  )

  const validateProvider = React.useCallback(
    async (
      payload: SetupProviderSaveRequest
    ): Promise<SetupProviderValidationResponse> => {
      const owner = await requireOwner()
      const response = await tldwClient.validateSetupProvider(payload, owner.requestOptions)
      assertOwner(owner)
      return response
    },
    [assertOwner, requireOwner]
  )

  const saveIngestDefaults = React.useCallback(
    async (payload: IngestDefaultsRequest) => {
      const owner = await requireOwner()
      const response = await tldwClient.saveIngestDefaults(payload, owner.requestOptions)
      assertOwner(owner)
      await refresh().catch(() => undefined)
      assertOwner(owner)
      return response
    },
    [assertOwner, refresh, requireOwner]
  )

  const saveAudioDefaults = React.useCallback(
    async (payload: AudioDefaultsRequest) => {
      const owner = await requireOwner()
      const response = await tldwClient.saveAudioDefaults(payload, owner.requestOptions)
      assertOwner(owner)
      await refresh().catch(() => undefined)
      assertOwner(owner)
      return response
    },
    [assertOwner, refresh, requireOwner]
  )

  const loadAudioRecommendations = React.useCallback(async () => {
    const owner = await requireOwner()
    const response = await tldwClient.getSetupAudioRecommendations(owner.requestOptions)
    assertOwner(owner)
    setAudioRecommendations(response.recommendations)
    return response.recommendations
  }, [assertOwner, requireOwner])

  const loadMcpToolsCatalog = React.useCallback(async () => {
    const owner = await requireOwner()
    const response = await tldwClient.getMcpToolsCatalog(owner.requestOptions)
    assertOwner(owner)
    setMcpToolsCatalog(response)
    return response
  }, [assertOwner, requireOwner])

  const applyMcpTools = React.useCallback(
    async (payload: McpToolsApplyRequest): Promise<McpToolsApplyResponse> => {
      const owner = await requireOwner()
      const response = await tldwClient.applyMcpTools(payload, owner.requestOptions)
      assertOwner(owner)
      await refresh().catch(() => undefined)
      assertOwner(owner)
      return response
    },
    [assertOwner, refresh, requireOwner]
  )

  const validateMcpTools = React.useCallback(
    async (
      payload: McpToolsValidateRequest = {}
    ): Promise<McpToolsValidateResponse> => {
      const owner = await requireOwner()
      const response = await tldwClient.validateMcpTools(payload, owner.requestOptions)
      assertOwner(owner)
      await refresh().catch(() => undefined)
      assertOwner(owner)
      return response
    },
    [assertOwner, refresh, requireOwner]
  )

  const saveOptionalAdvanced = React.useCallback(
    async (payload: OptionalAdvancedRequest) => {
      const owner = await requireOwner()
      const response = await tldwClient.saveOptionalAdvanced(payload, owner.requestOptions)
      assertOwner(owner)
      await refresh().catch(() => undefined)
      assertOwner(owner)
      return response
    },
    [assertOwner, refresh, requireOwner]
  )

  const verifyFirstChat = React.useCallback(
    async (
      payload: FirstChatVerifyRequest
    ): Promise<FirstChatVerifyResponse> => {
      const owner = await requireOwner()
      const response = await tldwClient.verifyFirstRunChat(payload, owner.requestOptions)
      assertOwner(owner)
      await refresh().catch(() => undefined)
      assertOwner(owner)
      return response
    },
    [assertOwner, refresh, requireOwner]
  )

  const complete = React.useCallback(
    async (
      payload: FirstRunCompleteRequest = {}
    ): Promise<SetupCompleteResponse> => {
      const owner = await requireOwner()
      const response = await tldwClient.completeFirstRun(payload, owner.requestOptions)
      assertOwner(owner)
      await refresh().catch(() => undefined)
      assertOwner(owner)
      return response
    },
    [assertOwner, refresh, requireOwner]
  )

  return {
    state,
    metadata,
    providerCatalog,
    mcpToolsCatalog,
    audioRecommendations,
    loading,
    error,
    connectionConfigured,
    connectionChecking,
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
    loadMcpToolsCatalog,
    applyMcpTools,
    validateMcpTools,
    saveOptionalAdvanced,
    verifyFirstChat,
    complete
  }
}
