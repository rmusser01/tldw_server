import React from "react"
import { useTranslation } from "react-i18next"

type ChatReplyAnnouncer = {
  /** Say a reply finished. `key` names the reply, so a repeat call is ignored. */
  announceComplete: (key: string) => void
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
 * The transcript's completion announcement (AX-11). The message log is a
 * polite log, busy while a reply streams, so a screen reader reads the new
 * reply once streaming settles. This region, outside the log, then says
 * "Response complete" once. It never repeats the reply's text, so the text
 * is on the page once.
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
  const announceComplete = React.useCallback(
    (key: string) => {
      if (lastKeyRef.current === key) return
      lastKeyRef.current = key
      const text = String(
        t("playground:message.responseComplete", {
          defaultValue: "Response complete"
        })
      )
      // A second reply finishing before the first clears still changes the region.
      setAnnouncement((previous) => (previous === text ? `${text}${NO_BREAK_SPACE}` : text))
      if (clearTimerRef.current) clearTimeout(clearTimerRef.current)
      clearTimerRef.current = setTimeout(() => setAnnouncement(""), ANNOUNCEMENT_CLEAR_MS)
    },
    [t]
  )
  const value = React.useMemo(() => ({ announceComplete }), [announceComplete])
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
