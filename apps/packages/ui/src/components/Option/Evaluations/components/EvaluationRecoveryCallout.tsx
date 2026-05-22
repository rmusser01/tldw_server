import React from "react"
import { RecoveryCallout } from "@/components/ui/state"

type ApiResponseLike = {
  ok?: boolean
  status?: number
  error?: unknown
}

type EvaluationRecoveryCalloutProps = {
  title: React.ReactNode
  endpoint: string
  error?: unknown
  response?: ApiResponseLike | null
  message?: React.ReactNode
  className?: string
  "data-testid"?: string
}

const getErrorMessage = (error: unknown): string | null => {
  if (!error) return null
  if (error instanceof Error) return error.message
  if (typeof error === "string") return error
  if (typeof error === "object") {
    const record = error as Record<string, unknown>
    const nested = record.error || record.message
    return typeof nested === "string" ? nested : null
  }
  return String(error)
}

export const getEvaluationRecoveryDetail = (
  error?: unknown,
  response?: ApiResponseLike | null
): string | null => {
  const responseError = getErrorMessage(response?.error)
  const thrownError = getErrorMessage(error)
  const status = response?.status

  if (responseError && status) return `HTTP ${status}: ${responseError}`
  if (responseError) return responseError
  if (thrownError) return thrownError
  if (status) return `HTTP ${status}`
  return null
}

export const EvaluationRecoveryCallout: React.FC<EvaluationRecoveryCalloutProps> = ({
  title,
  endpoint,
  error,
  response,
  message = "Check server connection and try again. Open Health & diagnostics if this continues.",
  className,
  "data-testid": dataTestId
}) => {
  const detail = getEvaluationRecoveryDetail(error, response)
  const diagnostics = [
    { label: "Request path", value: endpoint },
    ...(detail ? [{ label: "Details", value: detail }] : [])
  ]

  return (
    <RecoveryCallout
      state="unavailable"
      title={title}
      message={message}
      diagnostics={diagnostics}
      className={className}
      data-testid={dataTestId}
    />
  )
}

export default EvaluationRecoveryCallout
