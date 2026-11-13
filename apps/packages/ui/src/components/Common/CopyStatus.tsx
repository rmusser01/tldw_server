import type { FC } from "react"
import { useTranslation } from "react-i18next"
import type { CopyFeedbackStatus } from "./useCopyFeedback"

type Props = {
  status: CopyFeedbackStatus
  /**
   * Changes on every copy (useCopyFeedback's `attempt`). The message is
   * re-created for each attempt, so a repeated "Copied" is announced again.
   */
  attempt?: number
  /** Confirmation shown on screen after a copy; without it "Copied" is only announced. */
  copiedLabel?: string
  className?: string
}

/** The hint shown when a copy fails, also used as the copy button's tooltip. */
export const useCopyFailedHint = () => {
  const { t } = useTranslation("common")
  return t(
    "copyFailedHint",
    "Select the text and copy it with Ctrl+C (⌘C on a Mac)."
  )
}

/**
 * Announces the result of a copy. It sits next to the copy button rather than
 * inside it: a button with an accessible name hides its content from screen
 * readers, so a live region inside it may never be announced. The region is
 * always mounted so a change to its text is announced.
 */
export const CopyStatus: FC<Props> = ({
  status,
  attempt = 0,
  copiedLabel,
  className = ""
}) => {
  const { t } = useTranslation("common")
  const hint = useCopyFailedHint()
  return (
    <span role="status" className={`text-[11px] ${className}`}>
      {status === "copied" ? (
        <span key={attempt} className={copiedLabel ? "text-success" : "sr-only"}>
          {copiedLabel || t("codeCopied", "Copied")}
        </span>
      ) : status === "failed" ? (
        <span key={attempt}>
          <span className="text-danger">{t("copyFailed", "Couldn't copy")}</span>
          <span className="sr-only"> {hint}</span>
        </span>
      ) : null}
    </span>
  )
}
