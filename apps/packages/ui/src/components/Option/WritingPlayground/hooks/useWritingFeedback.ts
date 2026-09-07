import { useCallback, useEffect, useRef, useState } from "react"
import { useStorage } from "@plasmohq/storage/hook"
import { bgRequest } from "@/services/background-proxy"
import type { AllowedPath } from "@/services/tldw/openapi-guard"
import { loadServicePromptSnapshot, subscribeToServicePromptConfigChanges, type ServicePromptSnapshot } from "@/services/service-prompts"
import { requestScopeFields } from "@/services/tldw/domains/service-prompts"
import { isRequestConfigScopeChangedError } from "@/services/tldw/service-prompt-scope-error"

export type Mood = "tense" | "romantic" | "melancholic" | "action" | "calm" | "mysterious" | "humorous" | null

export type EchoReaction = {
  persona: string
  emoji: string
  message: string
  timestamp: number
}

const ECHO_PERSONAS = [
  { name: "Alex", emoji: "🧐", part: "alex_system" },
  { name: "Sam", emoji: "😍", part: "sam_system" },
  { name: "Max", emoji: "🤨", part: "max_system" },
  { name: "Riley", emoji: "🎉", part: "riley_system" },
  { name: "Jordan", emoji: "📚", part: "jordan_system" },
] as const

const VALID_MOODS = new Set(["tense", "romantic", "melancholic", "action", "calm", "mysterious", "humorous"])
const MOOD_DEBOUNCE_MS = 10_000
const ECHO_DEBOUNCE_MS = 30_000
const ECHO_CHAR_THRESHOLD = 500

type UseWritingFeedbackProps = {
  editorText: string
  isOnline: boolean
  isGenerating: boolean
  selectedModel?: string
}

export type UseWritingFeedbackReturn = {
  moodEnabled: boolean
  setMoodEnabled: (v: boolean) => void
  currentMood: Mood
  moodAnalyzing: boolean
  echoEnabled: boolean
  setEchoEnabled: (v: boolean) => void
  echoReactions: EchoReaction[]
  echoAnalyzing: boolean
  charsSinceLastEcho: number
}

type ChatCompletionResponse = {
  choices?: Array<{
    message?: {
      content?: string | null
    } | null
  }>
}

async function callChat(
  systemPrompt: string,
  userText: string,
  model: string | undefined,
  snapshot: ServicePromptSnapshot,
): Promise<string> {
  const scopeFields = requestScopeFields(snapshot.requestScope)
  const data = await bgRequest<ChatCompletionResponse>({
    path: "/api/v1/chat/completions" as AllowedPath,
    method: "POST",
    ...scopeFields,
    headers: { "Content-Type": "application/json", ...scopeFields.headers },
    abortSignal: snapshot.scopeSignal,
    body: {
      model: model || "default",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userText },
      ],
      temperature: 0.7,
      max_tokens: 100,
    },
  })
  return data.choices?.[0]?.message?.content?.trim() || ""
}

export function useWritingFeedback({
  editorText,
  isOnline,
  isGenerating,
  selectedModel,
}: UseWritingFeedbackProps): UseWritingFeedbackReturn {
  const [moodEnabled, setMoodEnabled] = useStorage<boolean>("writing:mood-enabled", false)
  const [echoEnabled, setEchoEnabled] = useStorage<boolean>("writing:echo-enabled", false)
  const [currentMood, setCurrentMood] = useState<Mood>(null)
  const [moodAnalyzing, setMoodAnalyzing] = useState(false)
  const [echoReactions, setEchoReactions] = useState<EchoReaction[]>([])
  const [echoAnalyzing, setEchoAnalyzing] = useState(false)
  const [charsSinceLastEcho, setCharsSinceLastEcho] = useState(0)

  const moodReqIdRef = useRef(0)
  const echoInFlightRef = useRef(false)
  const lastMoodCallRef = useRef(0)
  const lastEchoCallRef = useRef(0)
  const echoIndexRef = useRef(0)
  const prevTextLenRef = useRef(editorText.length)
  const moodTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const controllersRef = useRef<Partial<Record<"mood" | "echo", AbortController>>>({})
  const leasesRef = useRef<Partial<Record<"mood" | "echo", { scopeKey: string; release: () => void }>>>({})
  const textLengthRef = useRef(editorText.length)
  textLengthRef.current = editorText.length

  const cancelFeedback = useCallback(() => {
    moodReqIdRef.current += 1
    for (const controller of Object.values(controllersRef.current)) controller.abort()
    controllersRef.current = {}
    for (const lease of Object.values(leasesRef.current)) lease.release()
    leasesRef.current = {}
    if (moodTimerRef.current) clearTimeout(moodTimerRef.current)
    echoInFlightRef.current = false
  }, [])

  const resetFeedback = useCallback(() => {
    cancelFeedback()
    setCurrentMood(null)
    setEchoReactions([])
    setMoodAnalyzing(false)
    setEchoAnalyzing(false)
    setCharsSinceLastEcho(0)
    prevTextLenRef.current = textLengthRef.current
    lastMoodCallRef.current = 0
    lastEchoCallRef.current = 0
    echoIndexRef.current = 0
  }, [cancelFeedback])

  useEffect(() => {
    // Pending first lookups have no retained scope lease yet.
    const clearUnboundState = () => {
      if (!leasesRef.current.mood && !leasesRef.current.echo) resetFeedback()
    }
    const unsubscribe = subscribeToServicePromptConfigChanges(clearUnboundState)
    window.addEventListener("tldw:auth-credentials-changed", resetFeedback)
    return () => {
      unsubscribe()
      window.removeEventListener("tldw:auth-credentials-changed", resetFeedback)
      cancelFeedback()
    }
  }, [resetFeedback, cancelFeedback])

  const requestFeedback = useCallback(async (
    kind: "mood" | "echo", text: string, model: string | undefined,
    controller: AbortController, personaPart?: typeof ECHO_PERSONAS[number]["part"],
  ): Promise<string> => {
    try {
      const id = kind === "mood" ? "writing.feedback.mood" : "writing.feedback.echo"
      const snapshot = await loadServicePromptSnapshot([id], { signal: controller.signal })
      if (controller.signal.aborted || snapshot.scopeSignal.aborted || snapshot.scopeInvalidatedSignal.aborted) {
        snapshot.release()
        return ""
      }
      if (Object.values(leasesRef.current).some((lease) => lease.scopeKey !== snapshot.scopeKey)) {
        snapshot.release()
        resetFeedback()
        return ""
      }
      leasesRef.current[kind]?.release()
      snapshot.scopeInvalidatedSignal.addEventListener("abort", resetFeedback, { once: true })
      leasesRef.current[kind] = {
        scopeKey: snapshot.scopeKey,
        release: () => {
          snapshot.scopeInvalidatedSignal.removeEventListener("abort", resetFeedback)
          snapshot.release()
        },
      }
      const parts = snapshot.definitions[id]?.parts
      const system = parts?.[kind === "mood" ? "system_semantics" : (personaPart ?? "")]
      const classification = parts?.classification_semantics
      if (!system?.trim() || (kind === "mood" && !classification?.trim())) return ""
      const result = await callChat(
        kind === "mood" ? `${system} Respond with exactly one word.` : system,
        kind === "mood"
          ? `${classification} Respond with ONLY one word from: tense, romantic, melancholic, action, calm, mysterious, humorous\n\nText: ${text.slice(-500)}`
          : `React to this passage:\n\n${text.slice(-1000)}`,
        model, snapshot,
      )
      return controller.signal.aborted || snapshot.scopeSignal.aborted || snapshot.scopeInvalidatedSignal.aborted ? "" : result
    } catch (error) {
      if (!controller.signal.aborted && isRequestConfigScopeChangedError(error)) resetFeedback()
      // Feedback remains best-effort, but failed lookups never dispatch defaults.
      return ""
    }
  }, [resetFeedback])

  // Track chars typed since last echo
  useEffect(() => {
    const delta = editorText.length - prevTextLenRef.current
    prevTextLenRef.current = editorText.length
    if (delta > 0) {
      setCharsSinceLastEcho((prev) => prev + delta)
    }
  }, [editorText])

  // Mood detection (debounced)
  useEffect(() => {
    if (!moodEnabled || !isOnline || isGenerating || !editorText.trim()) return

    if (moodTimerRef.current) clearTimeout(moodTimerRef.current)
    let cancelled = false
    let controller: AbortController | null = null

    moodTimerRef.current = setTimeout(async () => {
      const now = Date.now()
      if (now - lastMoodCallRef.current < MOOD_DEBOUNCE_MS) return
      lastMoodCallRef.current = now

      const reqId = ++moodReqIdRef.current
      controller = new AbortController()
      controllersRef.current.mood = controller
      setMoodAnalyzing(true)
      const result = await requestFeedback("mood", editorText, selectedModel, controller)
      if (cancelled) return
      if (reqId !== moodReqIdRef.current) return
      const word = result.toLowerCase().trim().replace(/[^a-z]/g, "")
      if (VALID_MOODS.has(word)) {
        setCurrentMood(word as Mood)
      }
      setMoodAnalyzing(false)
    }, MOOD_DEBOUNCE_MS)

    return () => {
      if (moodTimerRef.current) clearTimeout(moodTimerRef.current)
      cancelled = true
      controller?.abort()
      setMoodAnalyzing(false)
    }
  }, [editorText, moodEnabled, isOnline, isGenerating, selectedModel, requestFeedback])

  // Echo Chamber
  useEffect(() => {
    if (!echoEnabled || !isOnline || isGenerating || charsSinceLastEcho < ECHO_CHAR_THRESHOLD) return
    if (echoInFlightRef.current) return

    const now = Date.now()
    if (now - lastEchoCallRef.current < ECHO_DEBOUNCE_MS) return

    lastEchoCallRef.current = now
    echoInFlightRef.current = true

    const persona = ECHO_PERSONAS[echoIndexRef.current % ECHO_PERSONAS.length]
    echoIndexRef.current += 1

    let cancelled = false
    const controller = new AbortController()
    controllersRef.current.echo = controller
    setEchoAnalyzing(true)

    void requestFeedback("echo", editorText, selectedModel, controller, persona.part).then(
      (message) => {
        if (cancelled || controller.signal.aborted) return
        if (message) {
          setCharsSinceLastEcho(0)
          setEchoReactions((prev) => [
            { persona: persona.name, emoji: persona.emoji, message, timestamp: Date.now() },
            ...prev,
          ].slice(0, 20)) // Keep last 20 reactions
        }
        setEchoAnalyzing(false)
      },
    ).finally(() => {
      if (controllersRef.current.echo === controller) echoInFlightRef.current = false
    })

    return () => {
      cancelled = true
      controller.abort()
      setEchoAnalyzing(false)
      if (controllersRef.current.echo === controller) echoInFlightRef.current = false
    }
  }, [charsSinceLastEcho, echoEnabled, isOnline, isGenerating, editorText, selectedModel, requestFeedback])

  return {
    moodEnabled: moodEnabled ?? false,
    setMoodEnabled,
    currentMood,
    moodAnalyzing,
    echoEnabled: echoEnabled ?? false,
    setEchoEnabled,
    echoReactions,
    echoAnalyzing,
    charsSinceLastEcho,
  }
}
