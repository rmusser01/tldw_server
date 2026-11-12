import { Tooltip } from "antd"
import { CopyCheckIcon, CopyIcon } from "lucide-react"
import type { FC } from "react"
import { useCopyFeedback } from "./useCopyFeedback"

type Props = {
  /** The exact source to put on the clipboard. */
  value: string
  /** Accessible name and tooltip, e.g. "Copy code". */
  label: string
  /** Short confirmation shown next to the icon after a copy, e.g. "Copied". */
  copiedLabel?: string
  className?: string
}

/**
 * Copy button shared by every code block surface (the full CodeBlock and the
 * slim chat toolbar), so copy behaviour and feedback stay identical.
 */
export const CodeCopyButton: FC<Props> = ({
  value,
  label,
  copiedLabel,
  className = ""
}) => {
  const { copied, copy } = useCopyFeedback()

  return (
    <Tooltip title={label}>
      <button
        type="button"
        onClick={() => {
          void copy(value)
        }}
        aria-label={label}
        className={`flex items-center gap-1.5 rounded bg-none p-1 text-xs text-text-muted hover:bg-surface2 hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 ${className}`}>
        {!copied ? (
          <CopyIcon className="size-4" aria-hidden="true" />
        ) : (
          <CopyCheckIcon className="size-4 text-success" aria-hidden="true" />
        )}
        {copiedLabel ? (
          <span aria-live="polite" className="text-[11px] text-success">
            {copied ? copiedLabel : ""}
          </span>
        ) : null}
      </button>
    </Tooltip>
  )
}
