import { useEffect, useCallback, useRef } from 'react'
import { useShortcutConfig, type ShortcutConfig } from './useShortcutConfig'
import { useRouteTransitionStore } from '@/store/route-transition'

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

export const executeKeyboardShortcuts = (
  event: KeyboardEvent,
  shortcuts: KeyboardShortcutConfig[]
) => {
  shortcuts.forEach(({ shortcut, action, enabled = true }) => {
    if (!enabled) return

    const {
      key,
      ctrlKey = false,
      altKey = false,
      shiftKey = false,
      metaKey = false,
      preventDefault = true,
      stopPropagation = true
    } = shortcut

    const keyMatches = event.key.toLowerCase() === key.toLowerCase()
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
  useEffect(() => {
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
    description: target.description
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
      description: 'Toggle sidebar'
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
      description: 'Toggle chat with current page'
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
