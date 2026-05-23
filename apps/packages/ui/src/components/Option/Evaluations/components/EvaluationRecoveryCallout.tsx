import React from "react"
import { useTranslation } from "react-i18next"
import { RecoveryCallout, type StateAction } from "@/components/ui/state"

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
  primaryAction?: StateAction
  secondaryActions?: StateAction[]
  className?: string
  "data-testid"?: string
}

const getErrorMessage = (error: unknown): string | null => {
  if (!error) return null
  if (error instanceof Error) return error.message
  if (typeof error === "string") return error
  if (Array.isArray(error)) {
    const messages = error.map(getErrorMessage).filter(Boolean)
    return messages.length > 0 ? messages.join("; ") : null
  }
  if (typeof error === "object") {
    const record = error as Record<string, unknown>
    const nested = record.error || record.message || record.detail || record.msg
    if (!nested || nested === error) return null
    return getErrorMessage(nested)
  }
  return String(error)
}

const HTTP_STATUS_PREFIX_PATTERN = /^HTTP\s+\d+\b/i

const hasHttpStatusPrefix = (message: string, status: number): boolean => {
  const trimmedMessage = message.trim()
  const matchingStatusPattern = new RegExp(`^HTTP\\s+${status}\\b`, "i")
  return (
    matchingStatusPattern.test(trimmedMessage) ||
    HTTP_STATUS_PREFIX_PATTERN.test(trimmedMessage)
  )
}

export const getEvaluationRecoveryDetail = (
  error?: unknown,
  response?: ApiResponseLike | null
): string | null => {
  const responseError = getErrorMessage(response?.error)
  const thrownError = getErrorMessage(error)
  const status = response?.status

  if (responseError && status) {
    return hasHttpStatusPrefix(responseError, status)
      ? responseError
      : `HTTP ${status}: ${responseError}`
  }
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
  message,
  primaryAction,
  secondaryActions,
  className,
  "data-testid": dataTestId
}) => {
  const { t } = useTranslation()
  const detail = getEvaluationRecoveryDetail(error, response)
  const diagnostics = [
    {
      label: t("evaluations:recoveryRequestPathLabel", {
        defaultValue: "Request path"
      }),
      value: endpoint
    },
    ...(detail
      ? [
          {
            label: t("evaluations:recoveryDetailsLabel", {
              defaultValue: "Details"
            }),
            value: detail
          }
        ]
      : [])
  ]

  return (
    <RecoveryCallout
      state="unavailable"
      title={title}
      message={
        message ??
        t("evaluations:recoveryDefaultMessage", {
          defaultValue:
            "Check server connection and try again. Open Health & diagnostics if this continues."
        })
      }
      diagnostics={diagnostics}
      primaryAction={primaryAction}
      secondaryActions={secondaryActions}
      className={className}
      data-testid={dataTestId}
    />
  )
}

export default EvaluationRecoveryCallout
