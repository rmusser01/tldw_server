import type { CSSProperties, FC } from "react"
import { ChevronDown } from "lucide-react"
import { useTranslation } from "react-i18next"

type Props = {
  onClick: () => void
  /** A reply is still streaming in below the reader. */
  replyInProgress?: boolean
  /** New messages (or a finished reply) arrived below the reader. */
  hasUnseenMessages?: boolean
  /**
   * Positions the wrapper inside the chat surface, for example
   * "absolute -top-12 right-4". The button always aligns to the right edge.
   */
  className?: string
  style?: CSSProperties
  "data-testid"?: string
}

/**
 * Floating "scroll to latest" control shared by the chat surfaces. It sits at
 * the right edge so it does not cover the conversation, turns into a labelled
 * "Reply in progress" pill while a reply streams below the reader, and shows a
 * small dot when new messages arrived below them.
 */
export const ScrollToLatestButton: FC<Props> = ({
  onClick,
  replyInProgress = false,
  hasUnseenMessages = false,
  className = "",
  style,
  "data-testid": testId
}) => {
  const { t } = useTranslation("playground")
  const replyLabel = t("composer.replyInProgress", "Reply in progress")
  const label = replyInProgress
    ? t(
        "composer.scrollToLatestReplyInProgress",
        "Reply in progress, scroll to latest messages"
      )
    : hasUnseenMessages
      ? t(
          "composer.scrollToLatestNewMessages",
          "New messages below, scroll to latest messages"
        )
      : t("composer.scrollToLatest", "Scroll to latest messages")

  return (
    <div
      className={`pointer-events-none flex justify-end ${className}`}
      style={style}
    >
      <button
        type="button"
        onClick={onClick}
        aria-label={label}
        title={label}
        data-testid={testId}
        className={`pointer-events-auto relative inline-flex items-center gap-1.5 rounded-full border border-border bg-surface text-text-subtle shadow-md transition-colors hover:bg-surface2 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus ${
          replyInProgress ? "py-1.5 pl-3 pr-2" : "p-2"
        }`}
      >
        {replyInProgress ? (
          <span className="text-xs font-medium text-text">{replyLabel}</span>
        ) : null}
        <ChevronDown className="size-4" aria-hidden="true" />
        {hasUnseenMessages && !replyInProgress ? (
          <span
            data-testid="scroll-to-latest-new-dot"
            aria-hidden="true"
            className="absolute -right-0.5 -top-0.5 size-2.5 rounded-full border-2 border-surface bg-primary"
          />
        ) : null}
      </button>
    </div>
  )
}
