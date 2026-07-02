import React from "react"
import { Button } from "antd"
import { Upload } from "lucide-react"

export type ExpressionAssetUploaderProps = {
  label?: string
  accept?: string
  disabled?: boolean
  loading?: boolean
  onSelectFile: (file: File) => void
}

export const ExpressionAssetUploader: React.FC<ExpressionAssetUploaderProps> = ({
  label = "Upload",
  accept = "image/png,image/jpeg,image/webp,image/gif,image/avif",
  disabled,
  loading,
  onSelectFile
}) => {
  const inputRef = React.useRef<HTMLInputElement | null>(null)

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="sr-only"
        aria-label={label}
        disabled={disabled || loading}
        onChange={(event) => {
          const file = event.currentTarget.files?.[0]
          event.currentTarget.value = ""
          if (file) onSelectFile(file)
        }}
      />
      <Button
        size="small"
        type="text"
        icon={<Upload className="h-3.5 w-3.5" />}
        disabled={disabled}
        loading={loading}
        onClick={() => inputRef.current?.click()}
      >
        {label}
      </Button>
    </>
  )
}
