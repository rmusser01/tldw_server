import { useEffect, useRef } from "react"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import { capturePromptOwnerScope } from "@/services/prompt-local-owner"
import {
  useQuickChatStore,
  type QuickChatAssistantMode,
  type QuickChatMessage
} from "@/store/quick-chat"
import { normalizeQuickChatRoutePath } from "./workflow-guides"

type QuickChatPopoutStateInput = {
  messages: QuickChatMessage[]
  modelOverride: string | null
  assistantMode: QuickChatAssistantMode
}

export type QuickChatPopoutState = QuickChatPopoutStateInput & {
  ownerId: string
  sourceRoute: string | null
}

const isValidQuickChatMessage = (value: unknown): value is QuickChatMessage => {
  if (!value || typeof value !== "object") return false
  if (
    !("id" in value) ||
    !("role" in value) ||
    !("content" in value) ||
    !("timestamp" in value)
  ) {
    return false
  }
  const candidate = value as {
    id: unknown
    role: unknown
    content: unknown
    timestamp: unknown
  }
  return (
    typeof candidate.id === "string" &&
    (candidate.role === "user" || candidate.role === "assistant") &&
    typeof candidate.content === "string" &&
    typeof candidate.timestamp === "number"
  )
}

const normalizeAssistantMode = (value: unknown): QuickChatAssistantMode =>
  value === "docs_rag" || value === "browse_guides" ? value : "chat"

const normalizeSourceRoute = (value: unknown): string | null =>
  typeof value === "string" ? normalizeQuickChatRoutePath(value) : null

export const buildQuickChatPopoutState = (
  state: QuickChatPopoutStateInput,
  sourceRoute: string | null | undefined,
  ownerId: string
): QuickChatPopoutState => {
  if (!ownerId.trim()) throw new Error("Quick Chat account is unavailable")
  return { ...state, ownerId, sourceRoute: normalizeSourceRoute(sourceRoute) }
}

export const parseQuickChatPopoutState = (
  value: unknown,
  expectedOwnerId: string | null
): QuickChatPopoutState | null => {
  if (!value || typeof value !== "object") return null
  const candidate = value as Record<string, unknown>
  if (!expectedOwnerId || candidate.ownerId !== expectedOwnerId) return null
  const rawMessages = candidate.messages
  if (!Array.isArray(rawMessages)) return null

  const messages = rawMessages.filter(isValidQuickChatMessage)
  if (messages.length !== rawMessages.length) {
    return null
  }

  return {
    ownerId: expectedOwnerId,
    messages,
    modelOverride:
      typeof candidate.modelOverride === "string"
        ? candidate.modelOverride
        : null,
    assistantMode: normalizeAssistantMode(candidate.assistantMode),
    sourceRoute: normalizeSourceRoute(candidate.sourceRoute)
  }
}

/** Revalidate the existing server/principal owner before publishing a one-use transfer. */
export async function restoreQuickChatPopoutState(
  value: unknown,
  restore: (state: QuickChatPopoutState) => void,
  isMounted: () => boolean
): Promise<void> {
  let current = true
  const stop = watchChatAccountChanges((invalidated) => {
    if (invalidated) current = false
  })
  try {
    const scope = await capturePromptOwnerScope()
    try {
      scope.assertCurrent()
      const state = parseQuickChatPopoutState(value, scope.ownerId)
      if (state && current && isMounted()) {
        scope.assertCurrent()
        restore(state)
      }
    } finally {
      scope.release()
    }
  } finally {
    stop()
  }
}

/** Keep the consumed payload in memory so StrictMode replay can revalidate it. */
export function useQuickChatPopoutRestore(
  stateKey: string | null,
  restore: (state: QuickChatPopoutState) => void
): void {
  const pending = useRef<{ key: string; value: unknown } | null>(null)
  const restoreRef = useRef(restore)
  restoreRef.current = restore

  useEffect(() => {
    if (!stateKey || !/^quickchat_[0-9]{10,16}$/.test(stateKey)) return
    let mounted = true
    try {
      if (pending.current?.key !== stateKey) {
        const saved = sessionStorage.getItem(stateKey)
        sessionStorage.removeItem(stateKey)
        pending.current = {
          key: stateKey,
          value: saved ? JSON.parse(saved) : null
        }
      }
      const value = pending.current.value
      if (value) {
        const snapshot = useQuickChatStore.getState()
        void restoreQuickChatPopoutState(
          value,
          (state) => {
            const current = useQuickChatStore.getState()
            const unchanged = (
              [
                "messages",
                "modelOverride",
                "assistantMode",
                "isStreaming"
              ] as const
            ).every((key) => current[key] === snapshot[key])
            if (unchanged) restoreRef.current(state)
          },
          () => mounted
        )
          .catch(() => {
            if (mounted)
              console.warn(
                "Quick Chat transfer account is unavailable or changed"
              )
          })
          .finally(() => {
            if (mounted && pending.current?.key === stateKey)
              pending.current.value = null
          })
      }
    } catch {
      console.warn("Quick Chat transfer could not be restored")
    }
    return () => {
      mounted = false
    }
  }, [stateKey])
}
