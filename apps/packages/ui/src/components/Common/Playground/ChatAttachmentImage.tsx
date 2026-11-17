import React from "react"
import { Image } from "antd"
import { useTranslation } from "react-i18next"

export function ChatAttachmentImage({
  src,
  className
}: {
  src: string
  className?: string
}) {
  const { t } = useTranslation("playground")
  const currentSource = React.useRef({ src })
  if (currentSource.current.src !== src) currentSource.current = { src }
  const sourceIdentity = currentSource.current
  const [failedSource, setFailedSource] = React.useState<typeof sourceIdentity | null>(null)
  if (failedSource === sourceIdentity) {
    return (
      <div
        role="status"
        className="rounded-md border border-border bg-surface2 px-3 py-2 text-sm text-text-muted">
        {t(
          "imageUnavailableRecovery",
          "Image unavailable. Attach this image again before continuing."
        )}
      </div>
    )
  }
  return (
    <Image
      src={src}
      alt="Uploaded Image"
      width={180}
      className={className}
      onError={() => {
        if (currentSource.current === sourceIdentity) setFailedSource(sourceIdentity)
      }}
    />
  )
}
