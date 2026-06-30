import * as React from 'react'

import {
  applyBrowserSpeechSynthesisVoice,
  resolveTtsProviderContext,
  type TtsProviderContext
} from '@/services/tts-provider'
import { tldwClient } from '@/services/tldw/TldwApiClient'

import {
  buildReadAlongCacheKey,
  buildTtsSettingsSignature
} from './media-read-along-cache-key'
import {
  getMediaReadAlongAudioCacheEntry,
  saveMediaReadAlongAudioCacheEntry
} from './media-read-along-cache'
import {
  buildReadAlongSegments,
  resolveReadAlongScope,
  splitSegmentForTtsRequest
} from './media-read-along-segments'
import type {
  ReadAlongScope,
  ReadAlongSegment,
  ReadAlongSelection,
  ReadAlongSessionState,
  ReadAlongTtsRequestSegment
} from './types'

const LOOKAHEAD_SEGMENT_COUNT = 4
const TTS_REQUEST_MAX_TEXT_LENGTH = 4000

type SessionToken = symbol
type PlayAttemptToken = symbol

type AudioSource = {
  blob: Blob
  format: string
  mimeType: string
}

type InFlightSegmentAudio = {
  controller: AbortController
  promise: Promise<AudioSource[]>
  unlinkAbort: () => void
}

type ReadAlongSession = {
  token: SessionToken
  scope: ReadAlongScope
  queue: ReadAlongSegment[]
  providerContext: TtsProviderContext
  settingsSignature: string
  serverScope: string
  currentController: AbortController
  lookaheadController: AbortController
  playAttemptToken: PlayAttemptToken | null
  pendingIndex: number
  audio: HTMLAudioElement | null
  objectUrl: string | null
  browserUtterance: SpeechSynthesisUtterance | null
  browserVoiceCleanup: (() => void) | null
  prefetched: Map<string, AudioSource[]>
  inFlight: Map<string, InFlightSegmentAudio>
  cacheDisabled: boolean
}

type UseMediaReadAlongSessionArgs = {
  mediaId: string | null
  mediaKind: string | null
  content: string
  displayContent: string
  renderMode: string
  hideTranscriptTimings: boolean
  selection: ReadAlongSelection | null
  contentBodyRef: React.RefObject<HTMLElement | null>
  contentScrollContainerRef: React.RefObject<HTMLElement | null>
  embeddedMediaRef: React.RefObject<HTMLMediaElement | null>
}

const idleState: ReadAlongSessionState = {
  status: 'idle',
  scope: null,
  activeSegmentId: null,
  activeIndex: -1,
  totalSegments: 0,
  error: null,
  cacheDisabled: false
}

const errorMessage = (error: unknown): string =>
  error instanceof Error ? error.message : String(error || 'Unknown read-along error')

const isAbortError = (error: unknown): boolean =>
  error instanceof DOMException
    ? error.name === 'AbortError'
    : error instanceof Error && error.name === 'AbortError'

const readAlongAbortError = (): DOMException =>
  new DOMException('Read-along playback cancelled', 'AbortError')

const revokeObjectUrl = (url: string | null): void => {
  if (!url || typeof URL === 'undefined' || typeof URL.revokeObjectURL !== 'function') {
    return
  }
  URL.revokeObjectURL(url)
}

const createBlob = (source: AudioSource): Blob =>
  source.blob instanceof Blob
    ? source.blob
    : new Blob([source.blob], { type: source.mimeType })

const normalizeServerUrlForScope = (serverUrl: unknown): string => {
  const value = String(serverUrl || '').trim()
  if (!value) return 'unknown-server'

  try {
    const url = new URL(value)
    url.username = ''
    url.password = ''
    url.hash = ''
    url.search = ''
    return url.toString().replace(/\/$/, '').toLowerCase()
  } catch {
    return value
      .replace(/\/\/[^/@]+@/, '//')
      .replace(/[?#].*$/, '')
      .replace(/\/+$/, '')
      .toLowerCase()
  }
}

const resolveReadAlongServerScope = async (): Promise<string> => {
  const config = await tldwClient.getConfig().catch(() => null)
  const serverUrl = normalizeServerUrlForScope(config?.serverUrl)
  const authMode = String(config?.authMode || 'unknown')
  const credentialScope = config?.accessToken
    ? 'access-token'
    : config?.apiKey
      ? 'api-key'
      : 'none'
  const orgId = config?.orgId == null ? 'none' : String(config.orgId)

  return [
    `server:${serverUrl}`,
    `auth:${authMode}`,
    `credential:${credentialScope}`,
    `org:${orgId}`
  ].join('|')
}

const toRequestParts = (segment: ReadAlongSegment): ReadAlongTtsRequestSegment[] => {
  if (segment.text.length <= TTS_REQUEST_MAX_TEXT_LENGTH) {
    return [
      {
        id: segment.id,
        parentSegmentId: segment.id,
        index: 0,
        text: segment.text,
        sourceStart: segment.sourceStart,
        sourceEnd: segment.sourceEnd
      }
    ]
  }

  return splitSegmentForTtsRequest(segment, TTS_REQUEST_MAX_TEXT_LENGTH)
}

const linkAbortSignal = (
  signal: AbortSignal,
  controller: AbortController
): (() => void) => {
  if (signal.aborted) {
    controller.abort()
    return () => undefined
  }

  const abort = () => controller.abort()
  signal.addEventListener('abort', abort, { once: true })
  return () => signal.removeEventListener('abort', abort)
}

const cleanupBrowserVoiceListener = (session: ReadAlongSession): void => {
  session.browserVoiceCleanup?.()
  session.browserVoiceCleanup = null
}

export function useMediaReadAlongSession(args: UseMediaReadAlongSessionArgs) {
  const {
    mediaId,
    mediaKind,
    content,
    displayContent,
    renderMode,
    hideTranscriptTimings,
    selection,
    embeddedMediaRef
  } = args

  const [state, setState] = React.useState<ReadAlongSessionState>(idleState)
  const sessionRef = React.useRef<ReadAlongSession | null>(null)
  const activeTokenRef = React.useRef<SessionToken | null>(null)
  const playSegmentRef = React.useRef<
    (session: ReadAlongSession, index: number) => Promise<void>
  >(async () => undefined)

  const isCurrentSession = React.useCallback((token: SessionToken): boolean => {
    return activeTokenRef.current === token
  }, [])

  const isCurrentPlayAttempt = React.useCallback(
    (session: ReadAlongSession, playToken: PlayAttemptToken): boolean => {
      return isCurrentSession(session.token) && session.playAttemptToken === playToken
    },
    [isCurrentSession]
  )

  const mutateState = React.useCallback(
    (
      token: SessionToken,
      update:
        | ReadAlongSessionState
        | ((previous: ReadAlongSessionState) => ReadAlongSessionState)
    ): void => {
      if (!isCurrentSession(token)) return
      setState(update)
    },
    [isCurrentSession]
  )

  const cleanupSession = React.useCallback(
    (session: ReadAlongSession | null, cancelBrowserSpeech = true): void => {
      if (!session) return
      session.currentController.abort()
      session.lookaheadController.abort()
      session.inFlight.forEach((entry) => {
        entry.unlinkAbort()
        entry.controller.abort()
      })
      session.inFlight.clear()
      session.audio?.pause()
      session.audio = null
      revokeObjectUrl(session.objectUrl)
      session.objectUrl = null
      cleanupBrowserVoiceListener(session)
      session.browserUtterance = null
      session.playAttemptToken = null
      if (
        cancelBrowserSpeech &&
        session.providerContext.provider === 'browser' &&
        typeof window !== 'undefined' &&
        window.speechSynthesis
      ) {
        window.speechSynthesis.cancel()
      }
    },
    []
  )

  const stop = React.useCallback((): void => {
    const session = sessionRef.current
    activeTokenRef.current = null
    cleanupSession(session)
    sessionRef.current = null
    setState({
      ...idleState,
      status: 'stopped'
    })
  }, [cleanupSession])

  const loadRequestPartAudio = React.useCallback(
    async (
      session: ReadAlongSession,
      part: ReadAlongTtsRequestSegment,
      signal: AbortSignal,
      isCurrentLoad: () => boolean
    ): Promise<AudioSource> => {
      const requestText = session.providerContext.normalizeText(part.text)
      const key = await buildReadAlongCacheKey({
        serverScope: session.serverScope,
        mediaId: mediaId || 'unknown-media',
        mediaKind: mediaKind || 'unknown',
        segmentId: part.id,
        segmentText: requestText,
        sourceStart: part.sourceStart,
        sourceEnd: part.sourceEnd,
        settingsSignature: session.settingsSignature
      })

      const cached = await getMediaReadAlongAudioCacheEntry(key.id)
      if (signal.aborted || !isCurrentLoad()) {
        throw readAlongAbortError()
      }
      if (cached) {
        return {
          blob: cached.blob,
          format: cached.format,
          mimeType: cached.mimeType
        }
      }

      if (!session.providerContext.synthesize) {
        throw new Error('The selected TTS provider cannot synthesize generated audio')
      }

      const generated = await session.providerContext.synthesize(requestText, { signal })
      if (signal.aborted || !isCurrentLoad()) {
        throw readAlongAbortError()
      }
      const blob = new Blob([generated.buffer], { type: generated.mimeType })
      const saved = await saveMediaReadAlongAudioCacheEntry(
        {
          id: key.id,
          createdAt: Date.now(),
          lastUsedAt: Date.now(),
          mediaId: key.mediaId,
          mediaKind: key.mediaKind,
          segmentId: key.segmentId,
          settingsSignature: key.settingsSignature,
          textHash: key.textHash,
          blob,
          mimeType: generated.mimeType,
          format: generated.format,
          sizeBytes: blob.size
        },
        {
          signal,
          shouldContinue: isCurrentLoad
        }
      )
      if (signal.aborted || !isCurrentLoad()) {
        throw readAlongAbortError()
      }
      if (!saved) {
        session.cacheDisabled = true
        mutateState(session.token, (previous) => ({
          ...previous,
          cacheDisabled: true
        }))
      }

      return {
        blob,
        format: generated.format,
        mimeType: generated.mimeType
      }
    },
    [mediaId, mediaKind, mutateState]
  )

  const fetchSegmentAudioSources = React.useCallback(
    async (
      session: ReadAlongSession,
      segment: ReadAlongSegment,
      signal: AbortSignal,
      isCurrentLoad: () => boolean
    ): Promise<AudioSource[]> => {
      const parts = toRequestParts(segment)
      const sources: AudioSource[] = []
      for (const part of parts) {
        if (signal.aborted || !isCurrentLoad()) {
          throw readAlongAbortError()
        }
        sources.push(
          await loadRequestPartAudio(session, part, signal, isCurrentLoad)
        )
      }
      return sources
    },
    [loadRequestPartAudio]
  )

  const getSegmentAudioSources = React.useCallback(
    async (
      session: ReadAlongSession,
      segment: ReadAlongSegment,
      signal: AbortSignal,
      isCurrentLoad: () => boolean
    ): Promise<AudioSource[]> => {
      const prefetched = session.prefetched.get(segment.id)
      if (prefetched) {
        return prefetched
      }

      const inFlight = session.inFlight.get(segment.id)
      if (inFlight) {
        const sources = await inFlight.promise
        if (signal.aborted || !isCurrentLoad()) {
          throw readAlongAbortError()
        }
        return sources
      }

      return fetchSegmentAudioSources(session, segment, signal, isCurrentLoad)
    },
    [fetchSegmentAudioSources]
  )

  const resetLookahead = React.useCallback(
    (session: ReadAlongSession, preserveSegmentId?: string): void => {
      const preserved = preserveSegmentId
        ? session.inFlight.get(preserveSegmentId)
        : undefined
      preserved?.unlinkAbort()

      session.lookaheadController.abort()
      session.lookaheadController = new AbortController()
      session.inFlight.forEach((entry, segmentId) => {
        if (segmentId === preserveSegmentId) return
        entry.unlinkAbort()
        entry.controller.abort()
        session.inFlight.delete(segmentId)
      })
    },
    []
  )

  const prefetchLookahead = React.useCallback(
    async (session: ReadAlongSession, activeIndex: number): Promise<void> => {
      const lookahead = session.queue.slice(
        activeIndex + 1,
        activeIndex + 1 + LOOKAHEAD_SEGMENT_COUNT
      )
      for (const segment of lookahead) {
        if (!isCurrentSession(session.token)) return
        if (session.lookaheadController.signal.aborted) return
        if (session.prefetched.has(segment.id) || session.inFlight.has(segment.id)) {
          continue
        }

        const controller = new AbortController()
        const unlinkAbort = linkAbortSignal(
          session.lookaheadController.signal,
          controller
        )
        const inFlight: InFlightSegmentAudio = {
          controller,
          unlinkAbort,
          promise: fetchSegmentAudioSources(
            session,
            segment,
            controller.signal,
            () => isCurrentSession(session.token) && !controller.signal.aborted
          )
            .then((sources) => {
              if (isCurrentSession(session.token) && !controller.signal.aborted) {
                session.prefetched.set(segment.id, sources)
              }
              return sources
            })
            .finally(() => {
              unlinkAbort()
              controller.abort()
              if (session.inFlight.get(segment.id) === inFlight) {
                session.inFlight.delete(segment.id)
              }
            })
        }
        session.inFlight.set(segment.id, inFlight)

        try {
          await inFlight.promise
        } catch (error) {
          if (isAbortError(error)) return
          continue
        }
      }
    },
    [fetchSegmentAudioSources, isCurrentSession]
  )

  const playBrowserSegment = React.useCallback(
    (
      session: ReadAlongSession,
      index: number,
      playToken: PlayAttemptToken
    ): void => {
      const segment = session.queue[index]
      if (!segment || typeof window === 'undefined' || !window.speechSynthesis) {
        mutateState(session.token, {
          status: 'segment-error',
          scope: session.scope,
          activeSegmentId: segment?.id ?? null,
          activeIndex: segment ? index : -1,
          totalSegments: session.queue.length,
          error: 'Browser speech synthesis is unavailable',
          cacheDisabled: session.cacheDisabled
        })
        return
      }

      const utterance = new SpeechSynthesisUtterance(
        session.providerContext.normalizeText(segment.text)
      )
      session.browserVoiceCleanup = applyBrowserSpeechSynthesisVoice(
        utterance,
        window.speechSynthesis,
        session.providerContext.browserVoiceName
      ) ?? null
      utterance.rate = session.providerContext.playbackSpeed || 1
      utterance.onend = () => {
        if (
          !isCurrentPlayAttempt(session, playToken) ||
          session.browserUtterance !== utterance
        ) {
          return
        }
        cleanupBrowserVoiceListener(session)
        void playSegmentRef.current(session, index + 1)
      }
      utterance.onerror = () => {
        if (
          !isCurrentPlayAttempt(session, playToken) ||
          session.browserUtterance !== utterance
        ) {
          return
        }
        cleanupBrowserVoiceListener(session)
        mutateState(session.token, {
          status: 'segment-error',
          scope: session.scope,
          activeSegmentId: segment.id,
          activeIndex: index,
          totalSegments: session.queue.length,
          error: 'Browser speech synthesis failed',
          cacheDisabled: session.cacheDisabled
        })
      }

      session.browserUtterance = utterance
      mutateState(session.token, {
        status: 'playing',
        scope: session.scope,
        activeSegmentId: segment.id,
        activeIndex: index,
        totalSegments: session.queue.length,
        error: null,
        cacheDisabled: session.cacheDisabled
      })
      window.speechSynthesis.speak(utterance)
    },
    [isCurrentPlayAttempt, mutateState]
  )

  const playGeneratedSegment = React.useCallback(
    async (
      session: ReadAlongSession,
      index: number,
      playToken: PlayAttemptToken
    ): Promise<void> => {
      const segment = session.queue[index]
      if (!segment) return

      try {
        const sources = await getSegmentAudioSources(
          session,
          segment,
          session.currentController.signal,
          () => isCurrentPlayAttempt(session, playToken)
        )
        if (!isCurrentPlayAttempt(session, playToken)) return

        session.prefetched.delete(segment.id)
        const markSegmentError = (message: string) => {
          mutateState(session.token, {
            status: 'segment-error',
            scope: session.scope,
            activeSegmentId: segment.id,
            activeIndex: index,
            totalSegments: session.queue.length,
            error: message,
            cacheDisabled: session.cacheDisabled
          })
        }

        const playSourceAt = async (sourceIndex: number): Promise<void> => {
          const source = sources[sourceIndex]
          if (!source || !isCurrentPlayAttempt(session, playToken)) return

          revokeObjectUrl(session.objectUrl)
          const objectUrl = URL.createObjectURL(createBlob(source))
          const audio = new Audio(objectUrl)
          audio.playbackRate = session.providerContext.playbackSpeed || 1
          session.objectUrl = objectUrl
          session.audio = audio

          audio.addEventListener(
            'ended',
            () => {
              revokeObjectUrl(objectUrl)
              if (session.objectUrl === objectUrl) {
                session.objectUrl = null
              }
              if (!isCurrentPlayAttempt(session, playToken)) return
              if (sourceIndex < sources.length - 1) {
                void playSourceAt(sourceIndex + 1).catch((error) => {
                  if (!isCurrentPlayAttempt(session, playToken) || isAbortError(error)) {
                    return
                  }
                  markSegmentError(errorMessage(error))
                })
                return
              }
              void playSegmentRef.current(session, index + 1)
            },
            { once: true }
          )
          audio.addEventListener(
            'error',
            () => {
              revokeObjectUrl(objectUrl)
              if (!isCurrentPlayAttempt(session, playToken)) return
              markSegmentError('Generated audio playback failed')
            },
            { once: true }
          )

          mutateState(session.token, {
            status: 'playing',
            scope: session.scope,
            activeSegmentId: segment.id,
            activeIndex: index,
            totalSegments: session.queue.length,
            error: null,
            cacheDisabled: session.cacheDisabled
          })
          await audio.play()
          if (!isCurrentPlayAttempt(session, playToken)) return
          if (sourceIndex === 0) {
            void prefetchLookahead(session, index).catch(() => undefined)
          }
        }

        await playSourceAt(0)
      } catch (error) {
        if (!isCurrentPlayAttempt(session, playToken) || isAbortError(error)) return
        if (session.objectUrl) {
          revokeObjectUrl(session.objectUrl)
          session.objectUrl = null
        }
        session.audio = null
        mutateState(session.token, {
          status: 'segment-error',
          scope: session.scope,
          activeSegmentId: segment.id,
          activeIndex: index,
          totalSegments: session.queue.length,
          error: errorMessage(error),
          cacheDisabled: session.cacheDisabled
        })
      }
    },
    [getSegmentAudioSources, isCurrentPlayAttempt, mutateState, prefetchLookahead]
  )

  const playSegment = React.useCallback(
    async (session: ReadAlongSession, index: number): Promise<void> => {
      if (!isCurrentSession(session.token)) return
      session.currentController.abort()
      session.currentController = new AbortController()
      session.audio?.pause()
      cleanupBrowserVoiceListener(session)
      session.browserUtterance = null
      session.pendingIndex = index
      const playToken = Symbol('read-along-play-attempt')
      session.playAttemptToken = playToken

      if (index >= session.queue.length) {
        cleanupSession(session, false)
        sessionRef.current = null
        activeTokenRef.current = null
        setState({
          ...idleState,
          status: 'stopped',
          scope: session.scope,
          totalSegments: session.queue.length,
          cacheDisabled: session.cacheDisabled
        })
        return
      }

      const segment = session.queue[index]
      resetLookahead(session, segment?.id)
      mutateState(session.token, {
        status: 'preparing',
        scope: session.scope,
        activeSegmentId: segment?.id ?? null,
        activeIndex: segment ? index : -1,
        totalSegments: session.queue.length,
        error: null,
        cacheDisabled: session.cacheDisabled
      })

      if (session.providerContext.provider === 'browser' && !session.providerContext.synthesize) {
        playBrowserSegment(session, index, playToken)
        return
      }

      await playGeneratedSegment(session, index, playToken)
    },
    [
      cleanupSession,
      isCurrentSession,
      mutateState,
      playBrowserSegment,
      playGeneratedSegment,
      resetLookahead
    ]
  )

  React.useEffect(() => {
    playSegmentRef.current = playSegment
  }, [playSegment])

  const start = React.useCallback(
    async (scope: ReadAlongScope): Promise<void> => {
      if (!mediaId || !mediaKind || !selection) return

      const token = Symbol('media-read-along-session')
      cleanupSession(sessionRef.current)
      activeTokenRef.current = token
      const segments = buildReadAlongSegments({
        mediaId,
        content,
        displayContent,
        renderMode,
        hideTranscriptTimings
      })
      const queue = resolveReadAlongScope({ scope, segments, selection })
      if (queue.length === 0) {
        sessionRef.current = null
        activeTokenRef.current = null
        setState({
          ...idleState,
          status: 'segment-error',
          scope,
          error: 'No readable content found for the selected read-along scope'
        })
        return
      }

      const currentController = new AbortController()
      const lookaheadController = new AbortController()
      let providerContext: TtsProviderContext
      let serverScope = 'server:unknown-server'
      try {
        const resolved = await Promise.all([
          resolveTtsProviderContext(queue[0].text),
          resolveReadAlongServerScope()
        ])
        providerContext = resolved[0]
        serverScope = resolved[1]
      } catch (error) {
        if (!isCurrentSession(token)) return
        activeTokenRef.current = null
        setState({
          ...idleState,
          status: 'segment-error',
          scope,
          totalSegments: queue.length,
          error: errorMessage(error)
        })
        return
      }
      if (!isCurrentSession(token)) return
      const settingsSignature = buildTtsSettingsSignature({
        provider: providerContext.provider,
        ...providerContext.cacheSettings,
        format: providerContext.cacheSettings?.format ?? providerContext.formatInfo?.resolved
      })
      const session: ReadAlongSession = {
        token,
        scope,
        queue,
        providerContext,
        settingsSignature,
        serverScope,
        currentController,
        lookaheadController,
        audio: null,
        objectUrl: null,
        browserUtterance: null,
        browserVoiceCleanup: null,
        prefetched: new Map(),
        inFlight: new Map(),
        playAttemptToken: null,
        pendingIndex: -1,
        cacheDisabled: false
      }

      sessionRef.current = session
      setState({
        status: 'preparing',
        scope,
        activeSegmentId: null,
        activeIndex: -1,
        totalSegments: queue.length,
        error: null,
        cacheDisabled: false
      })

      const embeddedMedia = embeddedMediaRef.current
      if (embeddedMedia && !embeddedMedia.paused) {
        embeddedMedia.pause()
      }

      await playSegment(session, 0)
    },
    [
      cleanupSession,
      content,
      displayContent,
      embeddedMediaRef,
      hideTranscriptTimings,
      mediaId,
      mediaKind,
      playSegment,
      renderMode,
      selection
    ]
  )

  const pause = React.useCallback((): void => {
    const session = sessionRef.current
    if (!session) return
    if (session.providerContext.provider === 'browser' && typeof window !== 'undefined') {
      window.speechSynthesis?.pause()
    } else {
      session.audio?.pause()
    }
    setState((previous) => ({
      ...previous,
      status: previous.status === 'playing' ? 'paused' : previous.status
    }))
  }, [])

  const resume = React.useCallback((): void => {
    const session = sessionRef.current
    if (!session) return
    if (session.providerContext.provider === 'browser' && typeof window !== 'undefined') {
      window.speechSynthesis?.resume()
      setState((previous) => ({
        ...previous,
        status: previous.status === 'paused' ? 'playing' : previous.status
      }))
      return
    }

    const playPromise = session.audio?.play()
    if (!playPromise) return

    void playPromise.then(() => {
      if (!isCurrentSession(session.token)) return
      setState((previous) => ({
        ...previous,
        status: previous.status === 'paused' ? 'playing' : previous.status
      }))
    }).catch((error) => {
      if (!isCurrentSession(session.token)) return
      setState((previous) => ({
        ...previous,
        status: 'segment-error',
        error: errorMessage(error)
      }))
    })
  }, [isCurrentSession])

  const retry = React.useCallback((): void => {
    const session = sessionRef.current
    if (!session) return
    const index = Math.max(0, state.activeIndex)
    void playSegment(session, index)
  }, [playSegment, state.activeIndex])

  const skip = React.useCallback(
    (direction: 'next' | 'previous' = 'next'): void => {
      const session = sessionRef.current
      if (!session) return
      const offset = direction === 'previous' ? -1 : 1
      const baseIndex = state.activeIndex >= 0 ? state.activeIndex : session.pendingIndex
      const nextIndex = Math.min(
        session.queue.length,
        Math.max(0, baseIndex + offset)
      )
      session.audio?.pause()
      if (session.providerContext.provider === 'browser' && typeof window !== 'undefined') {
        cleanupBrowserVoiceListener(session)
        session.browserUtterance = null
        session.playAttemptToken = null
        window.speechSynthesis?.cancel()
      }
      void playSegment(session, nextIndex)
    },
    [playSegment, state.activeIndex]
  )

  React.useEffect(() => {
    if (!activeTokenRef.current) return
    stop()
  }, [mediaId, mediaKind, content, displayContent, renderMode, hideTranscriptTimings, stop])

  React.useEffect(() => {
    return () => {
      cleanupSession(sessionRef.current)
      sessionRef.current = null
      activeTokenRef.current = null
    }
  }, [cleanupSession])

  return {
    state,
    start,
    pause,
    resume,
    stop,
    retry,
    skip,
    activeSegmentId: state.activeSegmentId
  }
}
