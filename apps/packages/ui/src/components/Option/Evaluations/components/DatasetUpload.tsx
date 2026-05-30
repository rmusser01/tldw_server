/**
 * DatasetUpload component
 * Drag-and-drop upload for JSON/JSONL dataset samples.
 */

import React, { useState } from "react"
import { Upload } from "antd"
import { UploadCloud } from "lucide-react"
import { useTranslation } from "react-i18next"
import { Alert as DsAlert } from "@/components/ui/primitives"

interface DatasetUploadProps {
  onSamplesLoaded: (samples: any[]) => void
  className?: string
}

export const DatasetUpload: React.FC<DatasetUploadProps> = ({
  onSamplesLoaded,
  className = ""
}) => {
  const { t } = useTranslation(["evaluations", "common"])
  const [error, setError] = useState<string | null>(null)

  const parseJsonl = (text: string): any[] => {
    return text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => JSON.parse(line))
  }

  const handleFile = async (file: File) => {
    setError(null)
    try {
      const text = await file.text()
      const isJsonl = file.name.toLowerCase().endsWith(".jsonl")
      const samples = isJsonl
        ? parseJsonl(text)
        : (() => {
            const parsed = JSON.parse(text)
            if (Array.isArray(parsed)) return parsed
            if (parsed?.samples && Array.isArray(parsed.samples)) {
              return parsed.samples
            }
            throw new Error(
              t("evaluations:datasetUploadParseError", {
                defaultValue: "Expected an array of samples."
              }) as string
            )
          })()

      if (!Array.isArray(samples) || samples.length === 0) {
        throw new Error(
          t("evaluations:datasetUploadEmptyError", {
            defaultValue: "No samples found in file."
          }) as string
        )
      }
      onSamplesLoaded(samples)
    } catch (e: any) {
      setError(e?.message || "Failed to parse dataset file.")
    }
    return false
  }

  return (
    <div className={`space-y-2 ${className}`}>
      <Upload.Dragger
        accept=".jsonl,.json"
        beforeUpload={handleFile}
        multiple={false}
        showUploadList={false}
      >
        <p className="ant-upload-drag-icon">
          <UploadCloud className="h-8 w-8" />
        </p>
        <p className="ant-upload-text">
          {t("evaluations:datasetUploadPrompt", {
            defaultValue: "Drop JSONL/JSON file here or click to upload"
          })}
        </p>
        <p className="ant-upload-hint">
          {t("evaluations:datasetUploadHint", {
            defaultValue: "Accepted: .jsonl or .json containing an array of samples."
          })}
        </p>
      </Upload.Dragger>
      {error && (
        <DsAlert
          variant="error"
          title={t("evaluations:datasetUploadErrorTitle", {
            defaultValue: "Upload failed"
          })}
        >
          {error}
        </DsAlert>
      )}
    </div>
  )
}

export default DatasetUpload
