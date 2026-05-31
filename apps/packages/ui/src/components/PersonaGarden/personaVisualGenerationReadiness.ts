import type { PersonaVisualGenerationReadinessResponse } from "@/types/persona-visuals"

export type PersonaVisualGenerationReadinessStatus =
  | "loading"
  | "error"
  | "ready"
  | "jobs_unavailable"
  | "image_provider_unavailable"
  | "image_adapter_unavailable"
  | "dependency_check_failed"
  | "backend_unavailable"
  | "default_backend_unavailable"

export interface PersonaVisualGenerationReadinessView {
  status: PersonaVisualGenerationReadinessStatus
  canQueue: boolean
  blocking: boolean
  selectedBackend?: string | null
  defaultBackend?: string | null
  enabledBackends: string[]
  queue?: string | null
  errorMessage?: string | null
}

export function classifyPersonaVisualGenerationReadiness(
  readiness: PersonaVisualGenerationReadinessResponse | null,
  backendInput: string,
  options: { isLoading?: boolean; errorMessage?: string | null } = {}
): PersonaVisualGenerationReadinessView {
  const selectedBackend = backendInput.trim() || null
  if (options.isLoading) {
    return {
      status: "loading",
      canQueue: false,
      blocking: true,
      selectedBackend,
      enabledBackends: []
    }
  }
  if (options.errorMessage) {
    return {
      status: "error",
      canQueue: false,
      blocking: true,
      selectedBackend,
      enabledBackends: [],
      errorMessage: options.errorMessage
    }
  }
  if (!readiness) {
    return {
      status: "loading",
      canQueue: false,
      blocking: true,
      selectedBackend,
      enabledBackends: []
    }
  }

  const enabledBackends = readiness.enabled_backends || []
  const defaultBackend = readiness.default_backend || null
  const selectedBackendAvailable =
    selectedBackend === null || enabledBackends.includes(selectedBackend)

  if (!readiness.worker_enabled || readiness.reasons.includes("jobs_worker_disabled")) {
    return {
      status: "jobs_unavailable",
      canQueue: false,
      blocking: true,
      selectedBackend,
      defaultBackend,
      enabledBackends,
      queue: readiness.queue
    }
  }

  if (readiness.reasons.includes("dependency_check_failed")) {
    return {
      status: "dependency_check_failed",
      canQueue: false,
      blocking: true,
      selectedBackend,
      defaultBackend,
      enabledBackends,
      queue: readiness.queue
    }
  }

  if (enabledBackends.length === 0 || readiness.reasons.includes("image_backend_unavailable")) {
    return {
      status: "image_provider_unavailable",
      canQueue: false,
      blocking: true,
      selectedBackend,
      defaultBackend,
      enabledBackends,
      queue: readiness.queue
    }
  }

  if (selectedBackend && !selectedBackendAvailable) {
    return {
      status: "backend_unavailable",
      canQueue: false,
      blocking: true,
      selectedBackend,
      defaultBackend,
      enabledBackends,
      queue: readiness.queue
    }
  }

  if (!selectedBackend && !defaultBackend) {
    return {
      status: "default_backend_unavailable",
      canQueue: false,
      blocking: true,
      selectedBackend,
      defaultBackend,
      enabledBackends,
      queue: readiness.queue
    }
  }

  if (readiness.reasons.includes("image_adapter_unavailable")) {
    return {
      status: "image_adapter_unavailable",
      canQueue: false,
      blocking: true,
      selectedBackend,
      defaultBackend,
      enabledBackends,
      queue: readiness.queue
    }
  }

  return {
    status: "ready",
    canQueue: true,
    blocking: false,
    selectedBackend: selectedBackend || defaultBackend,
    defaultBackend,
    enabledBackends,
    queue: readiness.queue
  }
}
