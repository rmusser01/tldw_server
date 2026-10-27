import { useCallback, useEffect, useRef, useState } from "react"
import { message } from "antd"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"
import { deriveAdminGuardFromError, sanitizeAdminErrorMessage } from "./admin-error-utils"

export type RequestScope = { configSnapshot: TldwConfig; abortSignal: AbortSignal }

export const useAdminRequestScope = (config: TldwConfig) => {
  const currentConfig = useRef(config)
  currentConfig.current = config
  const lifetime = useRef<AbortController | null>(null)
  useEffect(() => {
    const controller = new AbortController()
    lifetime.current = controller
    return () => controller.abort()
  }, [])
  return useCallback((): RequestScope | null => {
    const controller = lifetime.current
    return controller && !controller.signal.aborted
      ? {
          configSnapshot: { ...currentConfig.current },
          abortSignal: controller.signal
        }
      : null
  }, [])
}

export const useAdminMutation = (
  captureScope: () => RequestScope | null,
  onGuardError?: (error: unknown) => void
) => {
  const pending = useRef(false)
  const [mutating, setMutating] = useState(false)
  const runMutation = async (failure: string, action: (scope: RequestScope) => Promise<void>) => {
    const scope = captureScope()
    if (!scope || pending.current) return
    pending.current = true
    setMutating(true)
    try {
      await action(scope)
    } catch (error) {
      const validationError = error !== null && typeof error === "object" && "errorFields" in error
      if (!scope.abortSignal.aborted && !validationError) {
        // A missing mutation target does not make the whole Admin API unavailable.
        if (deriveAdminGuardFromError(error) !== "notFound") onGuardError?.(error)
        message.error(sanitizeAdminErrorMessage(error, failure))
      }
    } finally {
      if (!scope.abortSignal.aborted) {
        pending.current = false
        setMutating(false)
      }
    }
  }
  return { runMutation, mutating }
}
