import React from "react"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { loadServicePromptSnapshot } from "@/services/service-prompts"
import type { Character } from "@/types/character"
import type { AssistantSelectionMode } from "@/types/assistant-selection"
import type { ChatScope } from "@/types/chat-scope"

/** Read a fresh card for preview. Preview never writes transcript or request history. */
export const useCharacterGreeting = ({
  playgroundReady, selectedCharacter, selectedCharacterMode,
  serverChatId, historyId, messagesLength, scope, viewKey
}: {
  playgroundReady: boolean
  selectedCharacter: Character | null
  selectedCharacterMode: AssistantSelectionMode | null
  serverChatId: string | number | null
  historyId: string | null
  messagesLength: number
  scope?: ChatScope
  viewKey?: string
}) => {
  const fresh = playgroundReady && selectedCharacterMode === "tracked" &&
    !!selectedCharacter?.id && !serverChatId && !historyId && messagesLength === 0
  const [attempt, retry] = React.useReducer(value => value + 1, 0)
  // Card DTOs may be re-created by a parent. Equal values must not restart hydration on every render.
  const fingerprint = JSON.stringify([selectedCharacter, scope, viewKey, fresh, attempt])
  const identityRef = React.useRef({ fingerprint, selectedCharacter, scope, fresh })
  if (identityRef.current.fingerprint !== fingerprint) identityRef.current = { fingerprint, selectedCharacter, scope, fresh }
  const identity = identityRef.current
  const [preview, setPreview] = React.useState<{
    identity: typeof identity; character: Character | null; loading: boolean; unavailable: boolean
  } | null>(null)
  React.useEffect(() => {
    if (!identity.fresh || !identity.selectedCharacter) return
    const abort = new AbortController()
    const source = identity.selectedCharacter
    let snapshot: Awaited<ReturnType<typeof loadServicePromptSnapshot>> | undefined
    const current = () => !abort.signal.aborted && !snapshot?.scopeSignal.aborted &&
      !snapshot?.scopeInvalidatedSignal.aborted
    const unavailable = () => {
      if (!abort.signal.aborted) setPreview({ identity, character: source, loading: false, unavailable: true })
    }
    setPreview({ identity, character: source, loading: true, unavailable: false })
    void (async () => {
      try {
        snapshot = await loadServicePromptSnapshot([], { signal: abort.signal })
        snapshot.scopeSignal.addEventListener("abort", unavailable, { once: true })
        snapshot.scopeInvalidatedSignal.addEventListener("abort", unavailable, { once: true })
        if (!current()) { unavailable(); return }
        const character = await tldwClient.getCharacter(String(source.id), {
          requestScope: snapshot.requestScope, signal: snapshot.scopeSignal
        })
        if (!current()) return
        if (!character || String(character.id) !== String(source.id)) throw new Error("character_preview_unavailable")
        setPreview({ identity, character, loading: false, unavailable: false })
      } catch {
        if (current()) setPreview({ identity, character: source, loading: false, unavailable: true })
      } finally {
        snapshot?.scopeSignal.removeEventListener("abort", unavailable)
        snapshot?.scopeInvalidatedSignal.removeEventListener("abort", unavailable)
        snapshot?.release()
      }
    })()
    return () => { abort.abort(); snapshot?.release() }
  }, [identity])
  const active = fresh && preview?.identity === identity ? preview : null
  return { character: active?.character ?? selectedCharacter, loading: fresh && (active?.loading ?? true), unavailable: active?.unavailable ?? false, fresh, retry }
}
