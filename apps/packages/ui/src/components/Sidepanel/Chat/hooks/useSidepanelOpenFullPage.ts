/**
 * The side panel's one "Open in full page" action (XP-09, #3101).
 *
 * The header button, More tools, the history strip and the command palette
 * all run this action. It replaces "Expand in full page" (the chat, no draft)
 * and "Copy draft & page references" (the draft, into whichever chat the full
 * page restored).
 *
 * - In the extension it opens the extension's own full page with the open
 *   chat (the history-selection address) and a short-lived copy of the draft
 *   and page references. A new character chat opens on its character route
 *   without the copy, because a copy retires that route on arrival.
 * - When the user has configured a WebUI URL, the extension opens that WebUI
 *   instead, as the web-hosted panel always does. The WebUI hand-off carries
 *   the draft and attachments; chats kept only on this device stay here.
 */
import React from "react"
import { useTranslation } from "react-i18next"
import { browser } from "wxt/browser"
import { useHistorySelectionContext } from "@/hooks/chat/useHistorySelection"
import { useSelectedAssistant } from "@/hooks/useSelectedAssistant"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { CHAT_PATH } from "@/routes/route-paths"
import {
  buildSidepanelChatHandoffRoute,
  consumeSidepanelChatHandoff,
  createSidepanelChatHandoff,
  type SidepanelChatHandoffPageContext
} from "@/services/sidepanel-chat-handoff"
import { readExplicitSidepanelChatWebUiUrl } from "@/services/tldw/sidepanel-chat-webui-handoff"
import { buildSidepanelFullAppChatPath } from "@/utils/sidepanel-full-app-route"
import { createSafeStorage } from "@/utils/safe-storage"

export type SidepanelFullPageTarget = "extension" | "webui"

type Translate = (key: string, fallback: string) => string

const isExtensionRuntime = () => {
  try {
    return Boolean(browser.runtime?.id)
  } catch {
    return false
  }
}

/** Where "Open in full page" goes: the WebUI, or the extension's own page. */
export const resolveSidepanelFullPageTarget =
  async (): Promise<SidepanelFullPageTarget> => {
    if (!isExtensionRuntime()) return "webui"
    const storage = createSafeStorage({ area: "local" })
    const webUiUrl = await readExplicitSidepanelChatWebUiUrl((key) =>
      storage.get(key)
    )
    return webUiUrl ? "webui" : "extension"
  }

/** The target, for copy; the extension's own page until storage answers. */
export const useSidepanelFullPageTarget = (): SidepanelFullPageTarget => {
  const [target, setTarget] = React.useState<SidepanelFullPageTarget>(() =>
    isExtensionRuntime() ? "extension" : "webui"
  )
  React.useEffect(() => {
    let active = true
    void resolveSidepanelFullPageTarget()
      .then((next) => {
        if (active) setTarget(next)
      })
      .catch(() => undefined)
    return () => {
      active = false
    }
  }, [])
  return target
}

/** The action's one name, and what it carries to each target. */
export const sidepanelFullPageCopy = (
  t: Translate,
  target: SidepanelFullPageTarget
) => ({
  label: t("sidepanel:openFullPage.label", "Open in full page"),
  description:
    target === "webui"
      ? t(
          "sidepanel:openFullPage.webuiDescription",
          "Opens this chat in the WebUI, with your draft and attachments. Chats kept only on this device stay here."
        )
      : t(
          "sidepanel:openFullPage.extensionDescription",
          "Opens this chat in a new tab, with your draft and page references. Your draft also stays here."
        )
})

type PageContextGetter = () =>
  | Promise<SidepanelChatHandoffPageContext | undefined>
  | SidepanelChatHandoffPageContext
  | undefined

export type SidepanelOpenFullPageDeps = {
  /** The composer's draft has been restored for its chat. */
  draftReady: boolean
  draftText: string
  /** Changes when the chat, draft owner or draft revision changes. */
  handoffSourceKey: string
  /** Captures the current source; the returned check fails once it changes. */
  captureHandoffGuard: () => () => boolean
  hasVisiblePageContext: boolean
  getVisiblePageContext?: PageContextGetter
  selectedCharacterId: string | null
  /** Opens the WebUI with the draft and attachments (the WebUI target). */
  openInWebUi?: () => Promise<boolean>
}

export type SidepanelOpenFullPage = {
  label: string
  description: string
  target: SidepanelFullPageTarget
  pending: boolean
  disabled: boolean
  open: () => Promise<boolean>
}

const pageContextHasContent = (
  pageContext: SidepanelChatHandoffPageContext | undefined
) =>
  Boolean(
    pageContext &&
      (pageContext.title?.trim() ||
        pageContext.url?.trim() ||
        pageContext.snippets.some((snippet) => snippet.text.trim().length > 0))
  )

/** Opens `/options.html#<route>` in a new tab; false when no tab opened. */
const openExtensionRoute = async (
  route: string,
  isCurrent: () => boolean
): Promise<boolean> => {
  if (!isCurrent()) return false
  let url = route
  try {
    if (browser.runtime?.id && browser.runtime.getURL) {
      url = browser.runtime.getURL(`/options.html#${route}` as "/options.html")
      if (browser.tabs?.create) {
        try {
          await browser.tabs.create({ url })
          return true
        } catch {
          // Fall back to a plain window below.
        }
      }
    }
  } catch {
    // Fall back to a plain window below.
  }
  if (!isCurrent()) return false
  return Boolean(window.open(url, "_blank"))
}

export function useSidepanelOpenFullPage(
  deps: SidepanelOpenFullPageDeps
): SidepanelOpenFullPage {
  const { t } = useTranslation(["sidepanel"])
  const notification = useAntdNotification()
  const historySelection = useHistorySelectionContext()
  const [selectedAssistant] = useSelectedAssistant(null)
  const target = useSidepanelFullPageTarget()
  const [pending, setPending] = React.useState(false)
  const fullAppChatPath = React.useMemo(
    () =>
      buildSidepanelFullAppChatPath({
        selectedAssistant,
        selectedCharacterId: deps.selectedCharacterId
      }),
    [deps.selectedCharacterId, selectedAssistant]
  )
  const source = { ...deps, fullAppChatPath, historySelection }
  const sourceRef = React.useRef(source)
  sourceRef.current = source
  const attemptRef = React.useRef<object | null>(null)
  const mountedRef = React.useRef(false)
  React.useLayoutEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      attemptRef.current = null
    }
  }, [])
  // A new source retires an opening still in flight.
  React.useLayoutEffect(() => {
    attemptRef.current = null
    setPending(false)
  }, [
    deps.draftReady,
    deps.handoffSourceKey,
    deps.draftText,
    deps.hasVisiblePageContext,
    deps.getVisiblePageContext,
    deps.captureHandoffGuard,
    fullAppChatPath
  ])

  const notifyFailure = React.useCallback(() => {
    notification.error({
      message: t(
        "sidepanel:openFullPage.failed",
        "Could not open the full page. Your draft stays here."
      )
    })
  }, [notification, t])

  const openInExtension = React.useCallback(
    async (expected: typeof source, isCurrent: () => boolean) => {
      const conversationPath = expected.historySelection
        ? await expected.historySelection.prepareExpansionPath()
        : CHAT_PATH
      if (!isCurrent()) return false
      if (!conversationPath) {
        notification.error({
          message: t(
            "sidepanel:header.restoreBeforeExpand",
            "Restore or start a conversation before opening it in a full page."
          )
        })
        return false
      }
      const characterRoute =
        conversationPath === CHAT_PATH && expected.fullAppChatPath !== CHAT_PATH
      const basePath = characterRoute ? expected.fullAppChatPath : conversationPath
      const hasDraft = expected.draftText.trim().length > 0
      const wantsCopy = hasDraft || expected.hasVisiblePageContext
      if (characterRoute || !wantsCopy) {
        const opened = await openExtensionRoute(basePath, isCurrent)
        if (!isCurrent()) return false
        if (!opened) notifyFailure()
        else if (characterRoute && wantsCopy)
          notification.warning({
            message: t(
              "sidepanel:openFullPage.characterDraftStays",
              "A new character chat opens without your draft. It stays here."
            )
          })
        return opened
      }

      let pageContext: SidepanelChatHandoffPageContext | undefined
      try {
        pageContext = await expected.getVisiblePageContext?.()
      } catch {
        pageContext = undefined
      }
      if (!isCurrent()) return false
      const pageReadFailed =
        expected.hasVisiblePageContext && !pageContextHasContent(pageContext)
      let route = basePath
      let packageId: string | null = null
      let shortened = false
      if (hasDraft || pageContextHasContent(pageContext)) {
        try {
          const pkg = await createSidepanelChatHandoff(
            {
              draftText: expected.draftText,
              pageContext,
              routeIntent: { path: CHAT_PATH }
            },
            isCurrent
          )
          if (!isCurrent()) {
            // This copy was never opened anywhere.
            await consumeSidepanelChatHandoff(pkg.id).catch(() => undefined)
            return false
          }
          packageId = pkg.id
          route = buildSidepanelChatHandoffRoute(basePath, pkg.id)
          shortened = Boolean(
            pkg.draft.truncated ||
              pkg.pageContext?.truncated ||
              pkg.pageContext?.snippets.some((snippet) => snippet.truncated)
          )
        } catch {
          if (isCurrent()) notifyFailure()
          return false
        }
      }
      const opened = await openExtensionRoute(route, isCurrent)
      if (!isCurrent()) return false
      if (!opened) {
        if (packageId)
          await consumeSidepanelChatHandoff(packageId, isCurrent).catch(
            () => undefined
          )
        if (isCurrent()) notifyFailure()
        return false
      }
      if (shortened || pageReadFailed)
        notification.warning({
          message: t(
            "sidepanel:openFullPage.partial",
            "Opened with a partial copy: some of the draft or page references were shortened or could not be read. Review it there. Your draft stays here."
          )
        })
      return true
    },
    [notification, notifyFailure, t]
  )

  const open = React.useCallback(async (): Promise<boolean> => {
    const expected = sourceRef.current
    if (!expected.draftReady || attemptRef.current) return false
    const sourceIsCurrent = expected.captureHandoffGuard()
    const attempt = {}
    const isCurrent = () => {
      const current = sourceRef.current
      return (
        mountedRef.current &&
        attemptRef.current === attempt &&
        sourceIsCurrent() &&
        current.draftReady &&
        current.handoffSourceKey === expected.handoffSourceKey &&
        current.draftText === expected.draftText &&
        current.fullAppChatPath === expected.fullAppChatPath
      )
    }
    attemptRef.current = attempt
    if (!isCurrent()) {
      attemptRef.current = null
      return false
    }
    setPending(true)
    try {
      const resolved = await resolveSidepanelFullPageTarget().catch(
        () => "extension" as const
      )
      if (!isCurrent()) return false
      if (resolved === "webui" && expected.openInWebUi) {
        try {
          return await expected.openInWebUi()
        } catch (error) {
          if (!isCurrent()) return false
          console.error("[sidepanel] Could not open the WebUI:", error)
          notifyFailure()
          return false
        }
      }
      return await openInExtension(expected, isCurrent)
    } finally {
      if (attemptRef.current === attempt) {
        attemptRef.current = null
        if (mountedRef.current) setPending(false)
      }
    }
  }, [notifyFailure, openInExtension])

  const copy = sidepanelFullPageCopy(t, target)
  return {
    label: copy.label,
    description: copy.description,
    target,
    pending,
    disabled: !deps.draftReady,
    open
  }
}
