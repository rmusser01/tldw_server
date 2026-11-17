import type { FC } from "react"
import { useTranslation } from "react-i18next"
import { CodeCopyButton } from "./CodeCopyButton"

/**
 * Tailwind group name shared by a slim code block frame and its toolbar, so
 * the Copy button can reveal on hover or keyboard focus of the whole block.
 */
export const CODE_BLOCK_GROUP_CLASS = "group/code"

/**
 * On devices with a precise pointer the Copy button stays quiet until the
 * block is hovered or focused; on touch screens (no hover) it is always shown.
 */
const COPY_REVEAL_CLASS =
  "[@media(hover:hover)]:opacity-0 [@media(hover:hover)]:group-hover/code:opacity-100 [@media(hover:hover)]:group-focus-within/code:opacity-100 focus-visible:opacity-100 transition-opacity"

export const resolveCodeBlockLanguageLabel = (normalizedLanguage: string) =>
  !normalizedLanguage || normalizedLanguage === "plaintext"
    ? "text"
    : normalizedLanguage

type Props = {
  /** Normalized language ("python", "plaintext", ...). */
  language: string
  value: string
}

/**
 * Slim header for the chat code block variants ("compact" and "github"):
 * the language on the left and a Copy button on the right.
 */
export const CodeBlockToolbar: FC<Props> = ({ language, value }) => {
  const { t } = useTranslation("common")

  return (
    <div className="flex items-center justify-between gap-2 border-b border-border/60 px-3 py-0.5">
      <span className="font-mono text-[11px] text-text-muted">
        {resolveCodeBlockLanguageLabel(language)}
      </span>
      <CodeCopyButton
        value={value}
        label={t("copyCode", "Copy code")}
        copiedLabel={t("codeCopied", "Copied")}
        className={COPY_REVEAL_CLASS}
      />
    </div>
  )
}
