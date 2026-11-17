import type { PromptStudioStatus } from "@/services/prompt-studio"

export type QueueHealthLevel = "healthy" | "degraded"

export type QueueHealthSummaryCode =
  | "healthy_idle"
  | "healthy_processing"
  | "degraded_failures"
  | "degraded_backlog"
  | "degraded_success_rate"

export type QueueHealthSummary = {
  level: QueueHealthLevel
  code: QueueHealthSummaryCode
  values: Record<string, number>
}

const toSuccessPercent = (successRate?: number): number =>
  Math.round(successRate ?? 0)

export const summarizeQueueHealth = (
  status: PromptStudioStatus
): QueueHealthSummary => {
  const successPercent = toSuccessPercent(status.success_rate)
  const failedCount = Number(status.by_status?.failed || 0)

  if (successPercent < 90 && failedCount > 0) {
    return {
      level: "degraded",
      code: "degraded_failures",
      values: { failedCount, successPercent }
    }
  }

  if (status.queue_depth > 10) {
    return {
      level: "degraded",
      code: "degraded_backlog",
      values: { queueDepth: status.queue_depth, successPercent }
    }
  }

  if (successPercent < 90) {
    return {
      level: "degraded",
      code: "degraded_success_rate",
      values: { successPercent }
    }
  }

  if (status.processing > 0) {
    return {
      level: "healthy",
      code: "healthy_processing",
      values: { processing: status.processing }
    }
  }

  return {
    level: "healthy",
    code: "healthy_idle",
    values: {}
  }
}

export const getQueueSuccessPercent = (status: PromptStudioStatus): number =>
  toSuccessPercent(status.success_rate)
