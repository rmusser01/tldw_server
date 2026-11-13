import React from "react"
import { useTranslation } from "react-i18next"
import { markdownToText } from "@/utils/markdown-to-text"

type ChatReplyAnnouncer = {
  /** Read a finished reply once. `key` names the reply, so a repeat call is ignored. */
  announce: (key: string, reply: string) => void
}

const NO_BREAK_SPACE = String.fromCharCode(0xa0)
/** Long enough for a screen reader to take the text, short enough not to linger. */
export const ANNOUNCEMENT_CLEAR_MS = 2000

const ChatReplyAnnouncerContext = React.createContext<ChatReplyAnnouncer | null>(
  null
)

/** The transcript's announcer, or null where a surface has none. */
export const useChatReplyAnnouncer = () =>
  React.useContext(ChatReplyAnnouncerContext)

/**
 * The chat transcript's only announcement (AX-11). The message log is quiet
 * (aria-live="off"), so streaming, the re-render after it and each message's
 * chrome (names, times, toolbars) are not read out. When a reply finishes,
 * its text is read once from this polite region, which sits outside the log.
 */
export const ChatReplyAnnouncerProvider = ({
  children
}: {
  children: React.ReactNode
}) => {
  const { t } = useTranslation(["playground"])
  const [announcement, setAnnouncement] = React.useState("")
  const lastKeyRef = React.useRef<string | null>(null)
  const clearTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  React.useEffect(
    () => () => {
      if (clearTimerRef.current) clearTimeout(clearTimerRef.current)
    },
    []
  )
  const announce = React.useCallback(
    (key: string, reply: string) => {
      if (lastKeyRef.current === key) return
      lastKeyRef.current = key
      const text =
        markdownToText(reply).trim() ||
        String(
          t("playground:message.responseComplete", {
            defaultValue: "Response complete"
          })
        )
      // A new reply with the same words still changes the region, so it is read.
      setAnnouncement((previous) => (previous === text ? `${text}${NO_BREAK_SPACE}` : text))
      // Once read, the copy leaves the page, so the reply's text is on it once.
      if (clearTimerRef.current) clearTimeout(clearTimerRef.current)
      clearTimerRef.current = setTimeout(() => setAnnouncement(""), ANNOUNCEMENT_CLEAR_MS)
    },
    [t]
  )
  const value = React.useMemo(() => ({ announce }), [announce])
  return (
    <ChatReplyAnnouncerContext.Provider value={value}>
      {children}
      {/* A polite region, not role=status, so it does not compete with a surface's own status notices. */}
      <div aria-live="polite" aria-atomic="true" className="sr-only" data-testid="chat-reply-announcer">
        {announcement}
      </div>
    </ChatReplyAnnouncerContext.Provider>
  )
}
