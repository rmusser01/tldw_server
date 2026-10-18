import React, {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useState,
  useRef
} from "react"
import { useTranslation } from "react-i18next"
import { isHiddenBySelfOrAncestor } from "@/utils/focus-return"
import { useLocation } from "react-router-dom"

import { useShortcut } from "@/hooks/useKeyboardShortcuts"
import { RESEARCH_WORKSPACE_PATH } from "@/routes/route-paths"
import { usePromptPaletteCommands } from "@/components/Option/Prompt/usePromptPaletteCommands"

import { useHistorySelectionContext } from "@/hooks/chat/useHistorySelection"
import {
  capturePaletteFocusRequest,
  type PaletteFocusRequest
} from "./command-palette-focus"

import type { CommandPaletteProps } from "./CommandPalette"

const loadCommandPalette = () => import("./CommandPalette")
const reloadCurrentPage = () => window.location.reload()

const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect

type CommandPaletteHostProps = {
  enabled?: boolean;
  commandPaletteProps?: CommandPaletteProps;
  includePromptCommands?: boolean;
  loadPalette?: () => Promise<{
    CommandPalette: React.ComponentType<CommandPaletteProps>
  }>
  reloadPage?: () => void
}

export const CommandPaletteHost = ({
  enabled = true,
  commandPaletteProps,
  includePromptCommands = false,
  loadPalette = loadCommandPalette,
  reloadPage = reloadCurrentPage
}: CommandPaletteHostProps) => {
  const location = useLocation()
  const selection = useHistorySelectionContext()
  const selectionRef = useRef(selection)
  useIsomorphicLayoutEffect(() => {
    selectionRef.current = selection
  }, [selection])
  const { t } = useTranslation("common")
  const loadingRootRef = useRef<HTMLDivElement>(null)
  const requestRef = useRef<PaletteFocusRequest | null>(null)
  const [openRequest, setOpenRequest] = useState<PaletteFocusRequest | null>(
    null
  )
  const [CommandPalette, setCommandPalette] =
    useState<React.ComponentType<CommandPaletteProps> | null>(null);
  const [loadFailed, setLoadFailed] = useState(false);
  const route = JSON.stringify([location.key, location.pathname, location.search, location.hash]);
  const previousRoute = useRef(route);
  const shortcutEnabled = enabled && location.pathname !== RESEARCH_WORKSPACE_PATH;
  const [hasMountedPalette, setHasMountedPalette] = useState(false);
  const [openSignal, setOpenSignal] = useState(0);
  const [promptQuery, setPromptQuery] = useState('');
  useIsomorphicLayoutEffect(() => {
    if (previousRoute.current !== route) {
      requestRef.current?.dispose()
      setHasMountedPalette(false)
    }
    previousRoute.current = route
  }, [route])
  useIsomorphicLayoutEffect(
    () => () => {
      requestRef.current?.dispose()
    },
    []
  )
  const promptPaletteCommands = usePromptPaletteCommands(
    promptQuery,
    includePromptCommands
  )

  useEffect(() => {
    if (!hasMountedPalette || CommandPalette) return
    let cancelled = false
    setLoadFailed(false)
    loadPalette().then(
      (module) => {
        if (!cancelled) setCommandPalette(() => module.CommandPalette)
      },
      () => {
        if (!cancelled) setLoadFailed(true)
      }
    )
    return () => {
      cancelled = true
    }
  }, [hasMountedPalette, CommandPalette, loadPalette, openRequest])

  useIsomorphicLayoutEffect(() => {
    if (!CommandPalette && hasMountedPalette)
      openRequest?.setRoot(loadingRootRef.current)
  }, [CommandPalette, hasMountedPalette, openRequest])

  const dismissPending = useCallback(() => {
    const request = openRequest
    if (
      !request ||
      request !== requestRef.current ||
      !loadingRootRef.current?.isConnected ||
      !request.isSourceCurrent()
    )
      return
    setHasMountedPalette(false)
    window.setTimeout(() => {
      const target = request?.target
      const current =
        request === requestRef.current &&
        request?.isCurrent() &&
        target?.isConnected &&
        document.activeElement === document.body &&
        !isHiddenBySelfOrAncestor(target) &&
        !target.closest("[inert]") &&
        !target.hasAttribute("disabled") &&
        target.getAttribute("aria-disabled") !== "true"
      if (requestRef.current === request) requestRef.current = null
      request?.dispose()
      if (current) target.focus()
    }, 0)
  }, [openRequest])

  const reloadPending = useCallback(() => {
    const request = openRequest
    if (
      !request ||
      request !== requestRef.current ||
      !loadingRootRef.current?.isConnected ||
      !request.isSourceCurrent()
    )
      return
    if (
      window.confirm(
        t("help.confirmReload", "Reload this page? Unsaved edits may be lost.")
      ) &&
      request === requestRef.current &&
      request.isSourceCurrent()
    )
      reloadPage()
  }, [openRequest, reloadPage, t])

  const openPalette = useCallback(() => {
    if (!enabled) return;
    if (requestRef.current?.isOpen() && requestRef.current.isCurrent()) return;
    requestRef.current?.dispose();
    const request = capturePaletteFocusRequest(selectionRef.current);
    requestRef.current = request;
    setOpenRequest(request);
    setHasMountedPalette(true);
    setOpenSignal((current) => current + 1);
  }, [enabled]);

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
    if (!enabled) {
      requestRef.current?.dispose();
      setOpenRequest(null);
      setHasMountedPalette(false);
    }
  }, [enabled]);

  useIsomorphicLayoutEffect(() => {
    window.addEventListener("tldw:open-command-palette", openPalette)
    return () => {
      window.removeEventListener("tldw:open-command-palette", openPalette)
    }
  }, [openPalette])

  if (!hasMountedPalette) {
    return null
  }

  if (!CommandPalette) {
    if (!openRequest?.isSourceCurrent()) return null
    return (
      <div
        ref={loadingRootRef}
        role={loadFailed ? "alert" : "status"}
        className="fixed bottom-4 right-4 z-50 max-w-sm rounded border border-border bg-surface p-4 text-sm text-text shadow-lg">
        <p>
          {loadFailed
            ? t(
                "commandPalette.loadFailed",
                "Commands could not be loaded. Reconnect, save any edits, then reload the page."
              )
            : t("commandPalette.loading", "Loading commands…")}
        </p>
        <div className="mt-2 flex gap-3">
          {loadFailed && (
            <button
              type="button"
              className="min-h-[44px] px-2 text-primary"
              onClick={reloadPending}>
              {t("help.reload", "Reload page")}
            </button>
          )}
          <button
            type="button"
            className="min-h-[44px] px-2 text-text-muted"
            onClick={dismissPending}>
            {t("dismiss", "Dismiss")}
          </button>
        </div>
      </div>
    )
  }
  return (
    <CommandPalette
      {...mergedCommandPaletteProps}
      openSignal={openSignal}
      openRequest={openRequest ?? undefined}
      registerGlobalOpenShortcut={false}
      listenForOpenEvents={false}
    />
  )
}
