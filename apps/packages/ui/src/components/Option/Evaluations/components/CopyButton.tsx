/**
 * CopyButton component
 * Small button to copy text to clipboard with feedback
 */

import React, { useCallback } from "react"
import { Button, Tooltip } from "antd"
import { Check, Copy } from "lucide-react"
import { useTranslation } from "react-i18next"
import { copyToClipboard } from "@/utils/clipboard"
import { CopyStatus, useCopyFailedHint } from "@/components/Common/CopyStatus"
import { useCopyFeedback } from "@/components/Common/useCopyFeedback"

interface CopyButtonProps {
  text: string
  className?: string
  size?: "small" | "middle" | "large"
  showTooltip?: boolean
}

export const CopyButton: React.FC<CopyButtonProps> = ({
  text,
  className = "",
  size = "small",
  showTooltip = true
}) => {
  const { t } = useTranslation(["common"])
  const { status, copied, failed, run } = useCopyFeedback(2000)
  const failedHint = useCopyFailedHint()

  // A refused write is shown next to the button, never left uncaught (G17).
  const handleCopy = useCallback(() => {
    void run(() => copyToClipboard({ text }))
  }, [run, text])

  const button = (
    <Button
      type="text"
      size={size}
      className={className}
      onClick={handleCopy}
      icon={
        copied ? (
          <Check className="h-3 w-3 text-success" />
        ) : (
          <Copy className="h-3 w-3" />
        )
      }
    />
  )

  return (
    <span className="inline-flex items-center gap-1">
      {showTooltip ? (
        <Tooltip
          title={
            copied
              ? t("common:copied", { defaultValue: "Copied!" })
              : failed
                ? failedHint
                : t("common:copyToClipboard", { defaultValue: "Copy to clipboard" })
          }
        >
          {button}
        </Tooltip>
      ) : (
        button
      )}
      <CopyStatus status={status} />
    </span>
  )
}

export default CopyButton
