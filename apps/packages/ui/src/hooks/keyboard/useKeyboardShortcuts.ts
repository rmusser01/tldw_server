import { useEffect, useLayoutEffect, useCallback, useRef } from 'react'
import { useShortcutConfig, type ShortcutConfig } from './useShortcutConfig'
import { useRouteTransitionStore } from '@/store/route-transition'
import { isEditableTarget } from "@/utils/editable-target"

export { isMac } from '../useKeyboardShortcuts'

export interface KeyboardShortcut {
  key: string
  ctrlKey?: boolean
  altKey?: boolean
  shiftKey?: boolean
  metaKey?: boolean
  preventDefault?: boolean
  stopPropagation?: boolean
}

export interface KeyboardShortcutConfig {
  shortcut: KeyboardShortcut
  action: () => void
  enabled?: boolean
  description?: string
  ignoreIfEditable?: boolean
}

export type ModeNavigationShortcutKey = keyof Pick<
  ShortcutConfig,
  | "modePlayground"
  | "modeSources"
  | "modeMedia"
  | "modeKnowledge"
  | "modeNotes"
  | "modePrompts"
  | "modeFlashcards"
  | "modeWorldBooks"
  | "modeDictionaries"
  | "modeCharacters"
>

export type ModeNavigationTarget = {
  key: ModeNavigationShortcutKey
  path: string
  description: string
}

export const modeNavigationTargets: ModeNavigationTarget[] = [
  { key: "modePlayground", path: "/chat", description: "Go to Chat" },
  { key: "modeSources", path: "/sources", description: "Go to Sources" },
  { key: "modeMedia", path: "/media", description: "Go to Media" },
  { key: "modeKnowledge", path: "/knowledge", description: "Go to Knowledge" },
  { key: "modeNotes", path: "/notes", description: "Go to Notes" },
  { key: "modePrompts", path: "/prompts", description: "Go to Prompts" },
  { key: "modeFlashcards", path: "/flashcards", description: "Go to Flashcards" },
  { key: "modeWorldBooks", path: "/world-books", description: "Go to World Books" },
  { key: "modeDictionaries", path: "/dictionaries", description: "Go to Dictionaries" },
  { key: "modeCharacters", path: "/characters", description: "Go to Characters" }
]

const DIGIT_KEY = /^[0-9]$/
const LETTER_KEY = /^[a-z]$/

/**
 * Alt/Option or Meta with a digit never types wanted text into a field, so
 * those chords (the area shortcuts) keep working while the composer has focus.
 */
const firesInsideEditable = (shortcut: KeyboardShortcut) =>
  Boolean(shortcut.altKey || shortcut.metaKey) && DIGIT_KEY.test(shortcut.key)

/**
 * Match on the typed key first. macOS Option (and AltGr layouts) change
 * `event.key` (Option+5 reports "∞"), so Alt chords and digits fall back to the
 * physical key. Ctrl/Meta+letter stays on the typed letter so Dvorak and other
 * layouts do not get a second binding on a different physical key.
 */
const matchesShortcutKey = (event: KeyboardEvent, key: string) => {
  const expected = key.toLowerCase()
  if (typeof event.key === "string" && event.key.toLowerCase() === expected) {
    return !(DIGIT_KEY.test(expected) && event.code?.startsWith("Numpad"))
  }
  const code = typeof event.code === "string" ? event.code : ""
  if (DIGIT_KEY.test(expected)) return code === `Digit${expected}`
  if (LETTER_KEY.test(expected) && event.altKey) {
    return code === `Key${expected.toUpperCase()}`
  }
  return false
}

export const executeKeyboardShortcuts = (
  event: KeyboardEvent,
  shortcuts: KeyboardShortcutConfig[]
) => {
  shortcuts.forEach(({ shortcut, action, enabled = true, ignoreIfEditable = false }) => {
    if (!enabled) return
    if (ignoreIfEditable && !firesInsideEditable(shortcut)) {
      const activeElement =
        typeof document === "undefined" ? null : document.activeElement
      if (
        isEditableTarget(event.target) ||
        isEditableTarget(activeElement)
      ) {
        return
      }
    }

    const {
      key,
      ctrlKey = false,
      altKey = false,
      shiftKey = false,
      metaKey = false,
      preventDefault = true,
      stopPropagation = true
    } = shortcut

    const keyMatches = matchesShortcutKey(event, key)
    const ctrlMatches = event.ctrlKey === ctrlKey
    const altMatches = event.altKey === altKey
    const shiftMatches = event.shiftKey === shiftKey
    const metaMatches = event.metaKey === metaKey

    if (keyMatches && ctrlMatches && altMatches && shiftMatches && metaMatches) {
      if (preventDefault) {
        event.preventDefault()
      }
      if (stopPropagation) {
        event.stopPropagation()
      }
      action()
    }
  })
}

/**
 * Hook for managing configurable keyboard shortcuts
 * @param shortcuts Array of keyboard shortcut configurations
 * @param target Target element to attach listeners to (defaults to document)
 */
export const useKeyboardShortcuts = (
  shortcuts: KeyboardShortcutConfig[],
  target: Document | HTMLElement | null = document
) => {
  const shortcutsRef = useRef(shortcuts)
  useLayoutEffect(() => {
    shortcutsRef.current = shortcuts
  }, [shortcuts])

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      executeKeyboardShortcuts(event, shortcutsRef.current)
    },
    []
  )

  useEffect(() => {
    if (!target) return

    target.addEventListener('keydown', handleKeyDown)

    return () => {
      target.removeEventListener('keydown', handleKeyDown)
    }
  }, [target, handleKeyDown])
}

export const useModeNavigationShortcuts = (
  navigate: (path: string) => void,
  enabled: boolean = true
) => {
  const { shortcuts: configuredShortcuts } = useShortcutConfig()

  const shortcuts: KeyboardShortcutConfig[] = modeNavigationTargets.map((target) => ({
    shortcut: configuredShortcuts[target.key],
    action: () => {
      useRouteTransitionStore.getState().start(target.path)
      navigate(target.path)
    },
    enabled,
    description: target.description,
    ignoreIfEditable: true
  }))

  useKeyboardShortcuts(shortcuts)

  return {
    shortcuts
  }
}

/**
 * Hook specifically for focus shortcuts in forms
 * @param textareaRef Reference to the textarea element to focus
 * @param enabled Whether the shortcuts are enabled
 */
export const useFocusShortcuts = (
  textareaRef: React.RefObject<HTMLTextAreaElement>,
  enabled: boolean = true
) => {
  const { shortcuts: configuredShortcuts } = useShortcutConfig()

  const focusTextarea = useCallback(() => {
    if (textareaRef.current) {
      textareaRef.current.focus()
      // Place cursor at the end of the text
      const textLength = textareaRef.current.value.length
      textareaRef.current.setSelectionRange(textLength, textLength)
    }
  }, [textareaRef])

  const shortcuts: KeyboardShortcutConfig[] = [
    {
      shortcut: configuredShortcuts.focusTextarea,
      action: focusTextarea,
      enabled,
      description: 'Focus textarea'
    }
  ]

  useKeyboardShortcuts(shortcuts)

  return {
    focusTextarea,
    shortcuts
  }
}

/**
 * Hook specifically for chat shortcuts
 * @param clearChat Function to clear/start new chat
 * @param enabled Whether the shortcuts are enabled
 */
export const useChatShortcuts = (
  clearChat: () => void,
  enabled: boolean = true
) => {
  const { shortcuts: configuredShortcuts } = useShortcutConfig()

  const newChat = useCallback(() => {
    clearChat()
  }, [clearChat])

  const shortcuts: KeyboardShortcutConfig[] = [
    {
      shortcut: configuredShortcuts.newChat,
      action: newChat,
      enabled,
      description: 'Start new chat'
    }
  ]

  useKeyboardShortcuts(shortcuts)

  return {
    newChat,
    shortcuts
  }
}

/**
 * Hook specifically for sidebar shortcuts
 * @param toggleSidebar Function to toggle sidebar
 * @param enabled Whether the shortcuts are enabled
 */
export const useSidebarShortcuts = (
  toggleSidebar: () => void,
  enabled: boolean = true
) => {
  const { shortcuts: configuredShortcuts } = useShortcutConfig()

  const toggleSidebarAction = useCallback(() => {
    toggleSidebar()
  }, [toggleSidebar])

  const shortcuts: KeyboardShortcutConfig[] = [
    {
      shortcut: configuredShortcuts.toggleSidebar,
      action: toggleSidebarAction,
      enabled,
      description: 'Toggle sidebar',
      // Ctrl/Cmd+B means bold while typing (XP-07).
      ignoreIfEditable: true
    }
  ]

  useKeyboardShortcuts(shortcuts)

  return {
    toggleSidebar: toggleSidebarAction,
    shortcuts
  }
}

/**
 * Hook specifically for chat mode shortcuts
 * @param toggleChatMode Function to toggle chat mode between normal and rag
 * @param enabled Whether the shortcuts are enabled
 */
export const useChatModeShortcuts = (
  toggleChatMode: () => void,
  enabled: boolean = true
) => {
  const { shortcuts: configuredShortcuts } = useShortcutConfig()

  const toggleChatModeAction = useCallback(() => {
    toggleChatMode()
  }, [toggleChatMode])

  const shortcuts: KeyboardShortcutConfig[] = [
    {
      shortcut: configuredShortcuts.toggleChatMode,
      action: toggleChatModeAction,
      enabled,
      description: 'Toggle chat with current page',
      // Ctrl+E is "end of line" in macOS text fields; never hijack it (XS-08).
      ignoreIfEditable: true
    }
  ]

  useKeyboardShortcuts(shortcuts)

  return {
    toggleChatMode: toggleChatModeAction,
    shortcuts
  }
}

/**
 * Hook specifically for web search shortcuts
 * @param toggleWebSearch Function to toggle web search
 * @param enabled Whether the shortcuts are enabled
 */
export const useWebSearchShortcuts = (
  toggleWebSearch: () => void,
  enabled: boolean = true
) => {
  const { shortcuts: configuredShortcuts } = useShortcutConfig()

  const toggleWebSearchAction = useCallback(() => {
    toggleWebSearch()
  }, [toggleWebSearch])

  const shortcuts: KeyboardShortcutConfig[] = [
    {
      shortcut: configuredShortcuts.toggleWebSearch,
      action: toggleWebSearchAction,
      enabled,
      description: 'Toggle web search'
    }
  ]

  useKeyboardShortcuts(shortcuts)

  return {
    toggleWebSearch: toggleWebSearchAction,
    shortcuts
  }
}

/**
 * Hook specifically for Quick Chat Helper shortcuts
 * @param toggleQuickChat Function to toggle Quick Chat Helper modal
 * @param enabled Whether the shortcuts are enabled
 */
export const useQuickChatShortcuts = (
  toggleQuickChat: () => void,
  enabled: boolean = true
) => {
  const { shortcuts: configuredShortcuts } = useShortcutConfig()

  const toggleQuickChatAction = useCallback(() => {
    toggleQuickChat()
  }, [toggleQuickChat])

  const shortcuts: KeyboardShortcutConfig[] = [
    {
      shortcut: configuredShortcuts.toggleQuickChatHelper,
      action: toggleQuickChatAction,
      enabled,
      description: 'Toggle Quick Chat Helper'
    }
  ]

  useKeyboardShortcuts(shortcuts)

  return {
    toggleQuickChat: toggleQuickChatAction,
    shortcuts
  }
}
