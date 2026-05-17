import * as React from 'react'

import {
  resolveTtsProviderContext,
  type TtsProviderContext
} from '@/services/tts-provider'

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
  resolveReadAlongScope
} from './media-read-along-segments'
import type {
  ReadAlongScope,
  ReadAlongSegment,
  ReadAlongSelection,
  ReadAlongSessionState
} from './types'

const LOOKAHEAD_SEGMENT_COUNT = 4

type SessionToken = symbol

type AudioSource = {
  blob: Blob
  format: string
  mimeType: string
}

type ReadAlongSession = {
  token: SessionToken
  scope: ReadAlongScope
  queue: ReadAlongSegment[]
  providerContext: TtsProviderContext
  settingsSignature: string
  currentController: AbortController
  lookaheadController: AbortController
  audio: HTMLAudioElement | null
  objectUrl: string | null
  browserUtterance: SpeechSynthesisUtterance | null
  prefetched: Map<string, AudioSource>
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

  const isCurrentSession = React.useCallback((token: SessionToken): boolean => {
    return activeTokenRef.current === token
  }, [])

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
      session.audio?.pause()
      session.audio = null
      revokeObjectUrl(session.objectUrl)
      session.objectUrl = null
      session.browserUtterance = null
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

  const loadSegmentAudio = React.useCallback(
    async (
      session: ReadAlongSession,
      segment: ReadAlongSegment,
      signal: AbortSignal
    ): Promise<AudioSource> => {
      const key = await buildReadAlongCacheKey({
        serverScope: 'media-read-along',
        mediaId: mediaId || 'unknown-media',
        mediaKind: mediaKind || 'unknown',
        segmentId: segment.id,
        segmentText: segment.text,
        sourceStart: segment.sourceStart,
        sourceEnd: segment.sourceEnd,
        settingsSignature: session.settingsSignature
      })

      const cached = await getMediaReadAlongAudioCacheEntry(key.id)
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

      const generated = await session.providerContext.synthesize(segment.text, { signal })
      const blob = new Blob([generated.buffer], { type: generated.mimeType })
      const saved = await saveMediaReadAlongAudioCacheEntry({
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
      })
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

  const prefetchLookahead = React.useCallback(
    async (session: ReadAlongSession, activeIndex: number): Promise<void> => {
      const lookahead = session.queue.slice(
        activeIndex + 1,
        activeIndex + 1 + LOOKAHEAD_SEGMENT_COUNT
      )
      for (const segment of lookahead) {
        if (!isCurrentSession(session.token)) return
        if (session.lookaheadController.signal.aborted) return
        if (session.prefetched.has(segment.id)) continue

        try {
          const source = await loadSegmentAudio(
            session,
            segment,
            session.lookaheadController.signal
          )
          if (!isCurrentSession(session.token)) return
          session.prefetched.set(segment.id, source)
        } catch (error) {
          if (isAbortError(error)) return
        }
      }
    },
    [isCurrentSession, loadSegmentAudio]
  )

  const playBrowserSegment = React.useCallback(
    (session: ReadAlongSession, index: number): void => {
      const segment = session.queue[index]
      if (!segment || typeof window === 'undefined' || !window.speechSynthesis) {
        mutateState(session.token, (previous) => ({
          ...previous,
          status: 'segment-error',
          error: 'Browser speech synthesis is unavailable'
        }))
        return
      }

      const utterance = new SpeechSynthesisUtterance(segment.text)
      utterance.rate = session.providerContext.playbackSpeed || 1
      utterance.onend = () => {
        if (!isCurrentSession(session.token)) return
        void playSegment(session, index + 1)
      }
      utterance.onerror = () => {
        mutateState(session.token, (previous) => ({
          ...previous,
          status: 'segment-error',
          error: 'Browser speech synthesis failed'
        }))
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
    [isCurrentSession, mutateState]
  )

  const playGeneratedSegment = React.useCallback(
    async (session: ReadAlongSession, index: number): Promise<void> => {
      const segment = session.queue[index]
      if (!segment) return

      try {
        const source =
          session.prefetched.get(segment.id) ||
          (await loadSegmentAudio(session, segment, session.currentController.signal))
        if (!isCurrentSession(session.token)) return

        session.prefetched.delete(segment.id)
        revokeObjectUrl(session.objectUrl)
        const objectUrl = URL.createObjectURL(createBlob(source))
        const audio = new Audio(objectUrl)
        audio.playbackRate = session.providerContext.playbackSpeed || 1
        session.objectUrl = objectUrl
        session.audio = audio

        audio.addEventListener('ended', () => {
          revokeObjectUrl(objectUrl)
          if (session.objectUrl === objectUrl) {
            session.objectUrl = null
          }
          if (!isCurrentSession(session.token)) return
          void playSegment(session, index + 1)
        })
        audio.addEventListener('error', () => {
          revokeObjectUrl(objectUrl)
          if (!isCurrentSession(session.token)) return
          mutateState(session.token, (previous) => ({
            ...previous,
            status: 'segment-error',
            error: 'Generated audio playback failed'
          }))
        })

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
        if (!isCurrentSession(session.token)) return
        void prefetchLookahead(session, index)
      } catch (error) {
        if (!isCurrentSession(session.token) || isAbortError(error)) return
        mutateState(session.token, (previous) => ({
          ...previous,
          status: 'segment-error',
          error: errorMessage(error)
        }))
      }
    },
    [isCurrentSession, loadSegmentAudio, mutateState, prefetchLookahead]
  )

  const playSegment = React.useCallback(
    async (session: ReadAlongSession, index: number): Promise<void> => {
      if (!isCurrentSession(session.token)) return
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

      if (session.providerContext.provider === 'browser' && !session.providerContext.synthesize) {
        playBrowserSegment(session, index)
        return
      }

      await playGeneratedSegment(session, index)
    },
    [cleanupSession, isCurrentSession, playBrowserSegment, playGeneratedSegment]
  )

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
      try {
        providerContext = await resolveTtsProviderContext(queue[0].text)
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
        speed: providerContext.cacheSettings?.speed ?? providerContext.playbackSpeed,
        format: providerContext.cacheSettings?.format ?? providerContext.formatInfo?.resolved
      })
      const session: ReadAlongSession = {
        token,
        scope,
        queue,
        providerContext,
        settingsSignature,
        currentController,
        lookaheadController,
        audio: null,
        objectUrl: null,
        browserUtterance: null,
        prefetched: new Map(),
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

    void session.audio?.play().then(() => {
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
      const nextIndex = Math.min(
        session.queue.length,
        Math.max(0, state.activeIndex + offset)
      )
      session.audio?.pause()
      if (session.providerContext.provider === 'browser' && typeof window !== 'undefined') {
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
