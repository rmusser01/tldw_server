import React from "react"
import { Button } from "antd"
import { Mic, Send, Volume2, Square } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useServerDictation } from "@/hooks/useServerDictation"
import { useSttSettings } from "@/hooks/useSttSettings"
import { useTTS } from "@/hooks/useTTS"
import {
  acceptBuddyTurn,
  acknowledgeBuddyResult,
  getBuddyAttachment,
  listBuddyActivity,
  listBuddyTurns,
  readBuddyConversation,
  stopBuddyTurn,
  type BuddyActivity,
  type BuddyAttachment,
  type BuddyTurn
} from "@/services/buddies"
import type { BuddyDraftState } from "./buddy-drafts"
import { useStoreMessageOption } from "@/store/option"
import type { PersonaVisualStateId } from "@/types/persona-visuals"
import type {
  ServerChatMessage,
  ServerChatSummary
} from "@/services/tldw/TldwApiClient"
type ResultActivity = BuddyActivity & {
  result: NonNullable<BuddyActivity["result"]>
}

export const BuddyInteraction = ({
  attachment,
  attachmentVersion,
  conversation,
  conversations,
  visible,
  onSelectConversation,
  onAttentionChange,
  onVisualStateChange,
  draftState,
  conversationPages = 1
}: {
  attachment: BuddyAttachment
  attachmentVersion: number
  conversation: ServerChatSummary | null
  conversations: ServerChatSummary[]
  visible: boolean
  onSelectConversation?: (id: string) => void
  onAttentionChange?: (count: number) => void
  onVisualStateChange?: (state: PersonaVisualStateId) => void
  draftState?: BuddyDraftState
  conversationPages?: number
}) => {
  const { t } = useTranslation("sidepanel")
  const label = (key: string, text: string) =>
    t(`buddyManagement.${key}`, { defaultValue: text })
  const [localDrafts, setLocalDrafts] = React.useState<Record<string, string>>(
    {}
  )
  const drafts = draftState?.drafts ?? localDrafts
  const setDrafts = draftState?.setDrafts ?? setLocalDrafts
  const [transcript, setTranscript] = React.useState<{
    conversationId: string
    messages: ServerChatMessage[]
  }>({ conversationId: "", messages: [] })
  const [turns, setTurns] = React.useState<BuddyTurn[]>([])
  const [activity, setActivity] = React.useState<ResultActivity[]>([])
  const [error, setError] = React.useState<string | null>(null)
  const [loadError, setLoadError] = React.useState<string | null>(null)
  const [sending, setSending] = React.useState(false)
  const [model, setModel] = React.useState("")
  const [provider, setProvider] = React.useState("")
  const [readAloud, setReadAloud] = React.useState(false)
  const [paused, setPaused] = React.useState(false)
  const [queue, setQueue] = React.useState<
    { id: string; conversationId: string }[]
  >([])
  const [speechTick, setSpeechTick] = React.useState(0)
  const [turnPages, setTurnPages] = React.useState(1)
  const [moreTurns, setMoreTurns] = React.useState(false)
  const [revision, setRevision] = React.useState(0)
  const localRequestKeys = React.useRef(
    new Map<string, { text: string; key: string }>()
  )
  const requestKeys = draftState?.requestKeys ?? localRequestKeys
  const mounted = React.useRef(true)
  const seenResults = React.useRef(new Set<string>())
  const firstLoad = React.useRef(true)
  const pendingSend = React.useRef(false)
  const selectedId = conversation?.id ?? ""
  const messages =
    transcript.conversationId === selectedId ? transcript.messages : []
  const currentTarget = React.useRef(selectedId)
  currentTarget.current = selectedId
  const draft = drafts[selectedId] ?? ""
  const { speak, cancel, isSpeaking } = useTTS()
  const speech = React.useRef({ speak, cancel })
  speech.current = { speak, cancel }
  const speechGeneration = React.useRef(0)
  const activeSpeech = React.useRef<string | null>(null)
  const sttSettings = useSttSettings()
  const speechToTextLanguage = useStoreMessageOption(
    (state) => state.speechToTextLanguage
  )
  const captureTarget = React.useRef<string | null>(null)
  const dictation = useServerDictation({
    canUseServerStt:
      attachment.scope_type === "conversation" && Boolean(selectedId),
    speechToTextLanguage,
    sttSettings,
    onTranscript: (text) => {
      const target = captureTarget.current
      if (!target || target !== currentTarget.current) return
      setDrafts((previous) => ({
        ...previous,
        [target]: `${previous[target] ?? ""} ${text}`.trim()
      }))
    },
    onError: () =>
      setError(
        label(
          "dictationFailed",
          "Dictation unavailable. Check audio settings or type your reply."
        )
      )
  })
  const stopCapture = React.useRef(dictation.stopServerDictation)
  stopCapture.current = dictation.stopServerDictation
  React.useEffect(() => {
    setModel("")
    setProvider("")
  }, [selectedId])
  React.useEffect(() => {
    captureTarget.current = null
    stopCapture.current()
  }, [visible, selectedId])
  React.useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      speechGeneration.current++
      captureTarget.current = null
      stopCapture.current()
      speech.current.cancel()
    }
  }, [])
  React.useEffect(() => {
    speechGeneration.current++
    speech.current.cancel()
    setQueue([])
  }, [attachmentVersion])

  React.useEffect(() => {
    let active = true
    let pending = false
    const poll = async () => {
      if (pending) return
      pending = true
      try {
        const state = await getBuddyAttachment()
        if (!state.attachment || state.version !== attachmentVersion)
          throw new Error(
            label(
              "changedTarget",
              "Attachment changed. Refresh before replying."
            )
          )
        const [results, activeTurns, updates] = await Promise.all([
          listBuddyTurns({ pages: turnPages }),
          listBuddyTurns({ status: "active" }),
          listBuddyActivity({ conversationPages })
        ])
        if (!active) return
        const ids = new Set(conversations.map((c) => c.id))
        const scopedTurns = [
          ...new Map(
            [...results.turns, ...activeTurns.turns].map((turn) => [
              turn.id,
              turn
            ])
          ).values()
        ].filter((turn) => ids.has(turn.conversation_id))
        setMoreTurns(results.hasMore ?? results.turns.length >= turnPages * 100)
        setTurns(scopedTurns)
        const scopedActivity = updates.items.filter(
          (item): item is ResultActivity =>
            item.result !== null && ids.has(item.conversation_id)
        )
        setActivity(scopedActivity)
        onAttentionChange?.(
          scopedActivity.filter((item) => !item.acknowledged).length
        )
        if (visible && conversation) {
          const result = await readBuddyConversation(
            attachment,
            conversation.id,
            conversation.workspace_id,
            { isCurrent: () => active && mounted.current }
          )
          if (active)
            setTranscript({
              conversationId: conversation.id,
              messages: result.messages
            })
        }
        if (!active) return
        for (const item of scopedActivity) {
          if (seenResults.current.has(item.result.id)) continue
          seenResults.current.add(item.result.id)
          if (firstLoad.current || !readAloud) continue
          const target = conversations.find(
            (c) => c.id === item.conversation_id
          )
          if (!target) continue
          if (active)
            setQueue((q) => [
              ...q.slice(-19),
              { id: item.result.id, conversationId: target.id }
            ])
        }
        firstLoad.current = false
        if (active) setLoadError(null)
        if (seenResults.current.size > 500)
          seenResults.current = new Set([...seenResults.current].slice(-250))
      } catch (e) {
        if (active) {
          setLoadError(
            e instanceof Error ? e.message : "Buddy interaction unavailable"
          )
          setTranscript({ conversationId: "", messages: [] })
          setTurns([])
          setActivity([])
          onAttentionChange?.(0)
          speechGeneration.current++
          speech.current.cancel()
          setQueue([])
        }
      } finally {
        pending = false
      }
    }
    void poll()
    const timer = window.setInterval(() => void poll(), 4000)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [
    selectedId,
    attachmentVersion,
    visible,
    revision,
    readAloud,
    conversations,
    conversationPages,
    turnPages
  ])

  React.useEffect(() => {
    if (!readAloud || paused || !queue.length || activeSpeech.current) return
    const current = queue[0]
    const generation = speechGeneration.current
    activeSpeech.current = current.id
    void (async () => {
      const state = await getBuddyAttachment()
      if (state.version !== attachmentVersion || !state.attachment)
        throw new Error("Attachment changed before speech playback.")
      const target = conversations.find((c) => c.id === current.conversationId)
      if (!target) throw new Error("Conversation unavailable for speech.")
      const result = await readBuddyConversation(
        attachment,
        target.id,
        target.workspace_id,
        {
          isCurrent: () =>
            mounted.current && generation === speechGeneration.current
        }
      )
      const message = result.messages.find((item) => item.id === current.id)
      if (
        !mounted.current ||
        generation !== speechGeneration.current ||
        !message
      )
        return
      await speech.current.speak({
        utterance: `${result.conversation.title}. ${message.content}`,
        saveClip: false
      })
    })()
      .catch((e) => {
        if (generation === speechGeneration.current) setError(e.message)
      })
      .finally(() => {
        activeSpeech.current = null
        if (generation === speechGeneration.current)
          setQueue((q) => q.filter((item) => item.id !== current.id))
        if (mounted.current) setSpeechTick((value) => value + 1)
      })
  }, [queue, paused, readAloud, speechTick])
  const pause = () => {
    speechGeneration.current++
    speech.current.cancel()
    setPaused((value) => !value)
  }
  const send = async () => {
    if (!conversation || !draft.trim() || pendingSend.current) return
    const target = conversation.id
    const text = draft.trim()
    let request = requestKeys.current.get(target)
    if (!request || request.text !== text) {
      request = { text, key: crypto.randomUUID() }
      requestKeys.current.set(target, request)
    }
    pendingSend.current = true
    setSending(true)
    setError(null)
    captureTarget.current = null
    dictation.stopServerDictation()
    try {
      const turn = await acceptBuddyTurn({
        conversation_id: target,
        text,
        client_request_id: request.key,
        expected_attachment_version: attachmentVersion,
        ...(model.trim() ? { model: model.trim() } : {}),
        ...(provider.trim() ? { provider: provider.trim() } : {})
      })
      if (!mounted.current) return
      setTurns((previous) => [
        turn,
        ...previous.filter((t) => t.id !== turn.id)
      ])
      setDrafts((previous) =>
        previous[target]?.trim() === text
          ? { ...previous, [target]: "" }
          : previous
      )
      requestKeys.current.delete(target)
      setRevision((value) => value + 1)
    } catch (e) {
      if (!mounted.current) return
      setError(
        e instanceof Error
          ? e.message
          : label(
              "sendFailed",
              "Reply could not be confirmed. Your draft is retained. Retry uses the same request identity."
            )
      )
    } finally {
      pendingSend.current = false
      if (mounted.current) setSending(false)
    }
  }
  const pending = turns.filter(
    (turn) => turn.status === "queued" || turn.status === "running"
  )
  React.useEffect(() => {
    onVisualStateChange?.(
      error || loadError
        ? "error"
        : isSpeaking
          ? "speaking"
          : pending.length
            ? "thinking"
            : "idle"
    )
  }, [error, loadError, isSpeaking, pending.length, onVisualStateChange])
  return (
    <section
      className="space-y-3 text-sm text-text"
      aria-label={label("interaction", "Buddy conversation")}
    >
      {error ? (
        <p role="alert" className="text-danger">
          {error}
        </p>
      ) : null}
      {loadError ? (
        <p role="alert" className="text-danger">
          {loadError}
        </p>
      ) : null}
      {pending.length ? (
        <div role="status" className="space-y-2">
          {pending.map((turn) => (
            <div
              key={turn.id}
              className="flex items-center justify-between gap-2"
            >
              <span>
                {turn.conversation_title}:{" "}
                {turn.status === "queued"
                  ? label("queued", "Queued")
                  : label("working", "Working")}
              </span>
              <Button
                size="small"
                onClick={() =>
                  void stopBuddyTurn(turn.id)
                    .then(() => setRevision((r) => r + 1))
                    .catch((e) => setError(e.message))
                }
              >
                {label("stop", "Stop")}
              </Button>
            </div>
          ))}
        </div>
      ) : null}
      {activity.some((item) => !item.acknowledged) ? (
        <div
          className="space-y-2"
          aria-label={label("newResults", "New conversation results")}
        >
          {activity
            .filter((item) => !item.acknowledged)
            .map((item) => (
              <div
                key={item.result.id}
                className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-border p-2"
              >
                <button
                  type="button"
                  className="min-w-0 flex-1 text-left font-medium text-primaryStrong underline"
                  onClick={() => onSelectConversation?.(item.conversation_id)}
                >
                  {item.title} — {label("newResult", "New response")}
                </button>
                <Button
                  size="small"
                  onClick={() =>
                    void acknowledgeBuddyResult({
                      conversation_id: item.conversation_id,
                      result_message_id: item.result.id
                    })
                      .then(() => setRevision((r) => r + 1))
                      .catch((e) => setError(e.message))
                  }
                >
                  {label("markRead", "Mark read")}
                </Button>
              </div>
            ))}
        </div>
      ) : null}
      {turns
        .filter((turn) => turn.status === "failed")
        .slice(0, 3)
        .map((turn) => (
          <p key={turn.id} role="status" className="text-danger">
            {turn.conversation_title}:{" "}
            {label(
              "turnFailed",
              "The reply did not complete. Check the conversation before retrying; earlier effects may already be saved."
            )}{" "}
            {turn.error_code ? (
              <span>{turn.error_code.replace(/_/g, " ")}</span>
            ) : null}
          </p>
        ))}
      {moreTurns ? (
        <Button size="small" onClick={() => setTurnPages((pages) => pages + 1)}>
          {label("olderTurns", "Load older reply status")}
        </Button>
      ) : null}
      {conversation ? (
        <>
          <div
            className="max-h-[35dvh] min-h-24 space-y-3 overflow-y-auto rounded-md border border-border p-3"
            role="log"
            aria-label={`${label("transcript", "Conversation history")}: ${conversation.title}`}
          >
            {messages
              .filter((m) => m.role !== "system")
              .map((message) => (
                <div key={message.id}>
                  <p className="mb-1 font-semibold">
                    {message.role === "user"
                      ? label("you", "You")
                      : conversation.assistant_name ||
                        label("assistant", "Assistant")}
                  </p>
                  <p className="whitespace-pre-wrap break-words leading-6">
                    {message.content}
                  </p>
                </div>
              ))}
            {!messages.length ? (
              <p className="text-text-muted">
                {label("emptyTranscript", "No messages loaded yet.")}
              </p>
            ) : null}
          </div>
          <form
            onSubmit={(e) => {
              e.preventDefault()
              void send()
            }}
            className="space-y-2"
          >
            <label className="block font-medium" htmlFor="buddy-reply">
              {label("replyTo", "Reply to")} {conversation.title}
            </label>
            <textarea
              id="buddy-reply"
              rows={3}
              maxLength={12000}
              className="block w-full resize-y rounded-md border border-border bg-surface p-3 text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary"
              value={draft}
              onChange={(e) => {
                const value = e.target.value
                setDrafts((previous) => ({ ...previous, [selectedId]: value }))
              }}
            />
            <div className="flex flex-wrap items-center justify-between gap-2">
              {attachment.scope_type === "conversation" ? (
                <Button
                  icon={<Mic size={16} />}
                  onClick={() => {
                    if (dictation.isServerDictating) {
                      dictation.stopServerDictation()
                      return
                    }
                    speechGeneration.current++
                    speech.current.cancel()
                    setPaused(true)
                    captureTarget.current = selectedId
                    void dictation.startServerDictation()
                  }}
                >
                  {dictation.isServerDictating
                    ? label("stopDictation", "Finish dictation")
                    : label("dictate", "Dictate a reply")}
                </Button>
              ) : null}
              <Button
                htmlType="submit"
                type="primary"
                icon={<Send size={16} />}
                loading={sending}
                disabled={!draft.trim() || Boolean(loadError)}
              >
                {label("send", "Send")}
              </Button>
            </div>
            <details>
              <summary className="cursor-pointer text-text-muted">
                {label("replySettings", "Reply model settings")}
              </summary>
              <div className="mt-2 grid gap-2 sm:grid-cols-2">
                <label>
                  {label("modelOverride", "Model override (optional)")}
                  <input
                    className="w-full rounded border border-border bg-surface p-2"
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    placeholder={label(
                      "savedModel",
                      "Use conversation setting"
                    )}
                  />
                </label>
                <label>
                  {label("providerOverride", "Provider override (optional)")}
                  <input
                    className="w-full rounded border border-border bg-surface p-2"
                    value={provider}
                    onChange={(e) => setProvider(e.target.value)}
                    placeholder={label(
                      "savedProvider",
                      "Use conversation setting"
                    )}
                  />
                </label>
              </div>
            </details>
          </form>
        </>
      ) : (
        <p>
          {label(
            "selectToReply",
            "Choose a conversation above to read context or reply."
          )}
        </p>
      )}
      <div className="flex flex-wrap items-center gap-2 border-t border-border pt-3">
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={readAloud}
            onChange={(e) => {
              setReadAloud(e.target.checked)
              if (!e.target.checked) {
                speechGeneration.current++
                speech.current.cancel()
                setQueue([])
              }
            }}
          />
          <Volume2 size={16} aria-hidden="true" />
          {label("readResponses", "Read new responses aloud")}
        </label>
        {readAloud ? (
          <>
            <Button size="small" onClick={pause}>
              {paused
                ? label("resumeSpeech", "Resume speech")
                : label("pauseSpeech", "Pause speech")}
            </Button>
            <Button
              size="small"
              icon={<Square size={14} />}
              disabled={!queue.length}
              onClick={() => {
                speechGeneration.current++
                speech.current.cancel()
                setQueue((q) => q.slice(1))
              }}
            >
              {label("skipSpeech", "Skip")}
            </Button>
            <span role="status">
              {queue.length} {label("speechQueued", "queued")}
            </span>
          </>
        ) : null}
      </div>
    </section>
  )
}
