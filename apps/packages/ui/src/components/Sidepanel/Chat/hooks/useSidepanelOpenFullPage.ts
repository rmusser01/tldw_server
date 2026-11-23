/**
 * The side panel's one "Open in full page" action (XP-09, #3101).
 *
 * The header button, More tools, the history strip and the command palette
 * all run this action. It replaces "Expand in full page" (the chat, no draft)
 * and "Copy draft & page references" (the draft, into whichever chat the full
 * page restored).
 *
 * - In the extension it always opens the extension's own full page, the one
 *   target that carries the open chat (the history-selection address), with a
 *   short-lived copy of the draft and page references. A new Character or
 *   Persona chat opens on its route without the copy, because a copy retires
 *   that route on arrival. Attachments stay in the panel, and it says so.
 * - When the user has configured a WebUI URL, the Tools menu also offers
 *   "Open in WebUI" (`webUi`): the draft, and settings if the user allows it
 *   there. The chat and its attachments stay here.
 * - The web-hosted panel's "Open in full page" is that WebUI hand-off.
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

/** Where "Open in full page" goes: the extension's own page, or (web-hosted) the WebUI. */
export const sidepanelFullPageTarget = (): SidepanelFullPageTarget =>
  isExtensionRuntime() ? "extension" : "webui"

/** The WebUI URL the user configured, or null; never guessed from the server URL. */
export const readConfiguredWebUiUrl = async (): Promise<string | null> => {
  if (!isExtensionRuntime()) return null
  const storage = createSafeStorage({ area: "local" })
  return await readExplicitSidepanelChatWebUiUrl((key) => storage.get(key)).catch(() => null)
}

const useConfiguredWebUiUrl = (): string | null => {
  const [url, setUrl] = React.useState<string | null>(null)
  React.useEffect(() => {
    let active = true
    void readConfiguredWebUiUrl().then((next) => {
      if (active) setUrl(next)
    })
    return () => {
      active = false
    }
  }, [])
  return url
}

/** The action's one name, and what it carries (`hasConversation`: a saved or local chat is open). */
export const sidepanelFullPageCopy = (
  t: Translate,
  target: SidepanelFullPageTarget,
  hasConversation = true
) => ({
  label: t("sidepanel:openFullPage.label", "Open in full page"),
  description:
    target === "webui"
      ? t(
          "sidepanel:openFullPage.webHostedDescription",
          "Opens the full page with your draft. You can bring this chat's settings along there. This chat and its attachments stay here."
        )
      : hasConversation
        ? t(
            "sidepanel:openFullPage.extensionDescription",
            "Opens this chat in a new tab, with your draft and page references. Your draft also stays here."
          )
        : t(
            "sidepanel:openFullPage.newChatDescription",
            "Opens the chat page in a new tab, with your draft and page references. Your draft also stays here."
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
  /** A saved or local chat is open (not a new or temporary one). */
  hasConversation?: boolean
  /** Files or an image are attached in the panel. */
  hasAttachments?: boolean
  /** The draft owner's account check, while the draft isn't ready. */
  draftOwnerState?: "pending" | "ready" | "error"
  /** Opens the WebUI hand-off: the configured URL, or the web-hosted panel's own resolution. */
  openInWebUi?: (webUiUrl?: string) => Promise<boolean>
}

export type SidepanelOpenFullPage = {
  label: string
  description: string
  target: SidepanelFullPageTarget
  pending: boolean
  disabled: boolean
  /** Why it is disabled, when it is. */
  disabledReason: string | null
  open: () => Promise<boolean>
  /** "Open in WebUI", only in the extension with a configured WebUI URL. */
  webUi: { label: string; description: string; open: () => Promise<boolean> } | null
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
  const target = sidepanelFullPageTarget()
  const configuredWebUiUrl = useConfiguredWebUiUrl()
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

  const warnAttachmentsStay = React.useCallback((expected: { hasAttachments?: boolean }) => {
    if (!expected.hasAttachments) return
    notification.warning({
      message: t(
        "sidepanel:openFullPage.attachmentsStay",
        "Attachments stay in the side panel. Add them again on the full page."
      )
    })
  }, [notification, t])

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
            message:
              selectedAssistant?.kind === "persona"
                ? t("sidepanel:openFullPage.personaDraftStays", "A new Persona chat opens without your draft. It stays here.")
                : t("sidepanel:openFullPage.characterDraftStays", "A new Character chat opens without your draft. It stays here.")
          })
        if (opened) warnAttachmentsStay(expected)
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
      warnAttachmentsStay(expected)
      if (shortened || pageReadFailed)
        notification.warning({
          message: t(
            "sidepanel:openFullPage.partial",
            "Opened with a partial copy: some of the draft or page references were shortened or could not be read. Review it there. Your draft stays here."
          )
        })
      return true
    },
    [notification, notifyFailure, selectedAssistant?.kind, t, warnAttachmentsStay]
  )

  const run = React.useCallback(async (webUiUrl?: string): Promise<boolean> => {
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
      if (webUiUrl !== undefined || target === "webui") {
        if (!expected.openInWebUi) return false
        try {
          return await expected.openInWebUi(webUiUrl)
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
  }, [notifyFailure, openInExtension, target])
  const open = React.useCallback(() => run(), [run])
  const openWebUi = React.useCallback(
    () => (configuredWebUiUrl ? run(configuredWebUiUrl) : Promise.resolve(false)),
    [configuredWebUiUrl, run]
  )

  const copy = sidepanelFullPageCopy(t, target, deps.hasConversation ?? true)
  const disabledReason = deps.draftReady
    ? null
    : deps.draftOwnerState === "pending"
      ? t("sidepanel:openFullPage.checkingDraft", "Checking which account this draft belongs to…")
      : deps.draftOwnerState === "error"
        ? t("sidepanel:openFullPage.draftCheckFailed", "This draft's account couldn't be checked. Retry from the draft notice.")
        : t("sidepanel:header.restoreBeforeExpand", "Restore or start a conversation before opening it in a full page.")
  return {
    label: copy.label,
    description: copy.description,
    target,
    pending,
    disabled: !deps.draftReady,
    disabledReason,
    open,
    webUi:
      target === "extension" && configuredWebUiUrl && deps.openInWebUi
        ? {
            label: t("sidepanel:openFullPage.webuiLabel", "Open in WebUI"),
            description: t(
              "sidepanel:openFullPage.webuiDescription",
              "Opens the WebUI with your draft. You can bring this chat's settings along there. This chat and its attachments stay here."
            ),
            open: openWebUi
          }
        : null
  }
}
