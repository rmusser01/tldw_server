import React from "react"

import {
  getSetupReadinessProfiles,
  getSetupReadinessStatus,
  previewSetupReadiness,
  provisionSetupReadiness,
  verifySetupReadiness,
  type SetupReadinessClientMode,
  type SetupReadinessPreviewRequest,
  type SetupReadinessPreviewResponse,
  type SetupReadinessProfilesResponse,
  type SetupReadinessProvisionRequest,
  type SetupReadinessProvisionResponse,
  type SetupReadinessRequestError,
  type SetupReadinessStatusResponse,
  type SetupReadinessVerifyRequest,
  type SetupReadinessVerifyResponse
} from "@/services/tldw/setup-readiness"

const POLL_INTERVAL_MS = 3000
const ACTIVE_OPERATION_STATUSES = new Set(["queued", "running", "in_progress"])

export type SetupReadinessGuardState =
  | null
  | "remote_setup_blocked"
  | "admin_required"
  | "not_found"

type UseSetupReadinessOptions = {
  mode?: SetupReadinessClientMode
}

const fallbackUrl = "/setup"

const extractStatus = (err: unknown): number | null => {
  const status = (err as SetupReadinessRequestError | null)?.status
  if (typeof status === "number" && Number.isFinite(status)) return status
  const message = err instanceof Error ? err.message : String(err || "")
  const match = message.match(/\b(\d{3})\b/)
  if (!match) return null
  const parsed = Number(match[1])
  return Number.isFinite(parsed) ? parsed : null
}

const deriveGuardFromError = (
  err: unknown,
  mode: SetupReadinessClientMode
): SetupReadinessGuardState => {
  const status = extractStatus(err)
  if (status === 404) return "not_found"
  if (status === 401 || status === 403) {
    return mode === "admin" ? "admin_required" : "remote_setup_blocked"
  }
  return null
}

const errorMessage = (err: unknown, fallback: string): string => {
  if (err instanceof Error && err.message) return err.message
  const detail = (err as SetupReadinessRequestError | null)?.detail
  return detail || fallback
}

const isActiveStatus = (status: SetupReadinessStatusResponse | null): boolean => {
  if (!status) return false
  if (status.readiness_status === "provisioning") return true
  return Boolean(status.operation_status && ACTIVE_OPERATION_STATUSES.has(status.operation_status))
}

const logNonFatalRefreshError = (context: string, err: unknown) => {
  console.warn(`Setup readiness status refresh failed during ${context}.`, err)
}

export const useSetupReadiness = (options: UseSetupReadinessOptions = {}) => {
  const mode = options.mode || "first-run"
  const [loading, setLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)
  const [guard, setGuard] = React.useState<SetupReadinessGuardState>(null)
  const [profiles, setProfiles] = React.useState<SetupReadinessProfilesResponse | null>(null)
  const [status, setStatus] = React.useState<SetupReadinessStatusResponse | null>(null)
  const [preview, setPreview] = React.useState<SetupReadinessPreviewResponse | null>(null)
  const [provisionResult, setProvisionResult] =
    React.useState<SetupReadinessProvisionResponse | null>(null)
  const [verification, setVerification] =
    React.useState<SetupReadinessVerifyResponse | null>(null)
  const [previewing, setPreviewing] = React.useState(false)
  const [provisioning, setProvisioning] = React.useState(false)
  const [verifying, setVerifying] = React.useState(false)

  const refreshStatus = React.useCallback(async () => {
    const nextStatus = await getSetupReadinessStatus({ mode })
    setStatus(nextStatus)
    return nextStatus
  }, [mode])

  const refresh = React.useCallback(async () => {
    try {
      setLoading(true)
      const [nextProfiles, nextStatus] = await Promise.all([
        getSetupReadinessProfiles({ mode }),
        getSetupReadinessStatus({ mode })
      ])
      setProfiles(nextProfiles)
      setStatus(nextStatus)
      setGuard(null)
      setError(null)
      return { profiles: nextProfiles, status: nextStatus }
    } catch (err) {
      setGuard(deriveGuardFromError(err, mode))
      setError(errorMessage(err, "Unable to load setup readiness."))
      return null
    } finally {
      setLoading(false)
    }
  }, [mode])

  React.useEffect(() => {
    void refresh()
  }, [refresh])

  React.useEffect(() => {
    if (!isActiveStatus(status)) return

    const timeout = window.setTimeout(() => {
      void refreshStatus().catch((err) => {
        logNonFatalRefreshError("polling", err)
      })
    }, POLL_INTERVAL_MS)

    return () => {
      window.clearTimeout(timeout)
    }
  }, [refreshStatus, status])

  const previewSelection = React.useCallback(
    async (request: SetupReadinessPreviewRequest) => {
      setPreviewing(true)
      try {
        const result = await previewSetupReadiness(request, { mode })
        setPreview(result)
        setGuard(null)
        setError(null)
        return result
      } catch (err) {
        setGuard(deriveGuardFromError(err, mode))
        setError(errorMessage(err, "Unable to preview setup readiness."))
        return null
      } finally {
        setPreviewing(false)
      }
    },
    [mode]
  )

  const provision = React.useCallback(
    async (request: SetupReadinessProvisionRequest = {}) => {
      setProvisioning(true)
      try {
        const result = await provisionSetupReadiness(
          { ...request, confirmed: request.confirmed ?? true },
          { mode }
        )
        setProvisionResult(result)
        setVerification(null)
        setGuard(null)
        setError(null)
        await refreshStatus().catch((err) => {
          logNonFatalRefreshError("post-provision refresh", err)
        })
        return result
      } catch (err) {
        setGuard(deriveGuardFromError(err, mode))
        setError(errorMessage(err, "Unable to provision setup readiness."))
        return null
      } finally {
        setProvisioning(false)
      }
    },
    [mode, refreshStatus]
  )

  const verify = React.useCallback(
    async (request: SetupReadinessVerifyRequest = {}) => {
      setVerifying(true)
      try {
        const result = await verifySetupReadiness(request, { mode })
        setVerification(result)
        setGuard(null)
        setError(null)
        return result
      } catch (err) {
        setGuard(deriveGuardFromError(err, mode))
        setError(errorMessage(err, "Unable to verify setup readiness."))
        return null
      } finally {
        setVerifying(false)
      }
    },
    [mode]
  )

  return {
    error,
    fallbackUrl,
    guard,
    loading,
    mode,
    preview,
    previewSelection,
    previewing,
    profiles,
    provision,
    provisionResult,
    provisioning,
    refresh,
    refreshStatus,
    status,
    verification,
    verify,
    verifying
  }
}

export type UseSetupReadinessResult = ReturnType<typeof useSetupReadiness>
