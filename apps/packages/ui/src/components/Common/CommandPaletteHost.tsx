import React, {
  Suspense,
  lazy,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useState
} from "react"
import { useLocation } from "react-router-dom"

import { useShortcut } from "@/hooks/useKeyboardShortcuts"
import { WORKSPACE_PLAYGROUND_PATH } from "@/routes/route-paths"
import { usePromptPaletteCommands } from "@/components/Option/Prompt/usePromptPaletteCommands"

import type { CommandPaletteProps } from "./CommandPalette"

const CommandPalette = lazy(() =>
  import("./CommandPalette").then((m) => ({ default: m.CommandPalette }))
)

const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect

type CommandPaletteHostProps = {
  commandPaletteProps?: CommandPaletteProps
  includePromptCommands?: boolean
}

export const CommandPaletteHost = ({
  commandPaletteProps,
  includePromptCommands = false
}: CommandPaletteHostProps) => {
  const location = useLocation()
  const shortcutEnabled = location.pathname !== WORKSPACE_PLAYGROUND_PATH
  const [hasMountedPalette, setHasMountedPalette] = useState(false)
  const [openSignal, setOpenSignal] = useState(0)
  const [promptQuery, setPromptQuery] = useState("")
  const promptPaletteCommands = usePromptPaletteCommands(
    promptQuery,
    includePromptCommands
  )

  const openPalette = useCallback(() => {
    setHasMountedPalette(true)
    setOpenSignal((current) => current + 1)
  }, [])

  const mergedCommandPaletteProps = useMemo<CommandPaletteProps>(
    () => ({
      ...commandPaletteProps,
      additionalCommands: [
        ...(commandPaletteProps?.additionalCommands ?? []),
        ...promptPaletteCommands
      ],
      onQueryChange: (query: string) => {
        setPromptQuery(query)
        commandPaletteProps?.onQueryChange?.(query)
      }
    }),
    [commandPaletteProps, promptPaletteCommands]
  )

  useShortcut(
    {
      key: "k",
      modifiers: ["meta"],
      action: openPalette,
      description: "Open command palette",
      enabled: shortcutEnabled,
      allowInInput: true
    },
    [openPalette, shortcutEnabled]
  )

  useShortcut(
    {
      key: "k",
      modifiers: ["ctrl"],
      action: openPalette,
      description: "Open command palette",
      enabled: shortcutEnabled,
      allowInInput: true
    },
    [openPalette, shortcutEnabled]
  )

  useIsomorphicLayoutEffect(() => {
    window.addEventListener("tldw:open-command-palette", openPalette)
    return () => {
      window.removeEventListener("tldw:open-command-palette", openPalette)
    }
  }, [openPalette])

  if (!hasMountedPalette) {
    return null
  }

  return (
    <Suspense fallback={null}>
      <CommandPalette
        {...mergedCommandPaletteProps}
        openSignal={openSignal}
        registerGlobalOpenShortcut={false}
        listenForOpenEvents={false}
      />
    </Suspense>
  )
}
