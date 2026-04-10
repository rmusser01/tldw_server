import React, { useState, useEffect } from "react"
import { useTranslation } from "react-i18next"
import { Button, Checkbox, Tag, Progress } from "antd"
import {
  Shield,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Clock,
  Wrench,
} from "lucide-react"
import { UI_CONFIG } from "@/services/acp/constants"
import type { ACPPendingPermission, ACPPermissionTier } from "@/services/acp/types"

interface ACPPermissionModalProps {
  pendingPermissions: ACPPendingPermission[]
  approvePermission: (requestId: string, batchApproveTier?: ACPPermissionTier) => void
  denyPermission: (requestId: string) => void
}

export const ACPPermissionModal: React.FC<ACPPermissionModalProps> = ({
  pendingPermissions,
  approvePermission,
  denyPermission,
}) => {
  const { t } = useTranslation(["playground", "common"])

  const [batchApprove, setBatchApprove] = useState(false)
  const [showPolicyDetails, setShowPolicyDetails] = useState(false)
  const [now, setNow] = useState(Date.now())
  const currentPermission = pendingPermissions[0]

  useEffect(() => {
    if (!currentPermission) {
      return
    }
    const interval = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(interval)
  }, [currentPermission?.request_id])

  useEffect(() => {
    if (!currentPermission) {
      return
    }
    setNow(Date.now())
    setBatchApprove(false)
    setShowPolicyDetails(false)
  }, [currentPermission?.request_id])

  if (!currentPermission) {
    return null
  }

  const handleApprove = () => {
    approvePermission(
      currentPermission.request_id,
      batchApprove ? currentPermission.tier : undefined
    )
    setBatchApprove(false)
  }

  const handleDeny = () => {
    denyPermission(currentPermission.request_id)
    setBatchApprove(false)
  }

  const getTierColor = (tier: ACPPermissionTier) => {
    return UI_CONFIG.TIER_COLORS[tier] || "default"
  }

  const getTierIcon = (tier: ACPPermissionTier) => {
    switch (tier) {
      case "auto":
        return <CheckCircle className="h-4 w-4" />
      case "batch":
        return <AlertTriangle className="h-4 w-4" />
      case "individual":
        return <Shield className="h-4 w-4" />
      default:
        return null
    }
  }

  const timeElapsed = now - currentPermission.requestedAt.getTime()
  const totalMs = currentPermission.timeout_seconds * 1000
  // Guard against zero/negative timeouts to avoid Infinity/NaN in progress bar
  const timeRemaining = totalMs <= 0 ? 0 : Math.max(0, totalMs - timeElapsed)
  const progressPercent = totalMs <= 0 ? 0 : (timeRemaining / totalMs) * 100
  const hasPolicyMetadata = Boolean(
    currentPermission.approval_requirement
    || currentPermission.governance_reason
    || currentPermission.runtime_narrowing_reason
    || currentPermission.policy_snapshot_fingerprint
    || currentPermission.provenance_summary
  )

  return (
    <div
      role="dialog"
      aria-label={t("playground:acp.permissionRequired", "Permission Required")}
      className="absolute bottom-4 right-4 z-50 w-[420px] max-w-[calc(100vw-2rem)] max-h-[80vh] overflow-y-auto rounded-xl border border-border bg-surface shadow-2xl p-5"
    >
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-warning/10">
            <Shield className="h-5 w-5 text-warning" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-text">
              {t("playground:acp.permissionRequired", "Permission Required")}
            </h3>
            <p className="text-sm text-text-muted">
              {t("playground:acp.agentWantsTo", "The agent wants to execute a tool")}
            </p>
          </div>
        </div>

        {/* Queue indicator */}
        {pendingPermissions.length > 1 && (
          <div className="rounded-lg bg-info/10 px-3 py-2 text-sm text-info">
            {t("playground:acp.queuedPermissions", "{{count}} more permission requests queued", {
              count: pendingPermissions.length - 1,
            })}
          </div>
        )}

        {/* Tool info */}
        <div className="rounded-lg border border-border bg-surface2 p-4">
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Wrench className="h-4 w-4 text-text-muted" />
              <span className="font-mono text-sm font-medium text-text">
                {currentPermission.tool_name}
              </span>
            </div>
            <Tag
              color={getTierColor(currentPermission.tier)}
              className="flex items-center gap-1"
            >
              {getTierIcon(currentPermission.tier)}
              {UI_CONFIG.TIER_LABELS[currentPermission.tier]}
            </Tag>
          </div>

          {/* Tool arguments */}
          <div className="mb-3">
            <div className="mb-1 text-xs font-medium text-text-muted uppercase">
              {t("playground:acp.arguments", "Arguments")}
            </div>
            <div className="max-h-40 overflow-auto rounded bg-bg p-2">
              <pre className="text-xs text-text">
                {JSON.stringify(currentPermission.tool_arguments, null, 2)}
              </pre>
            </div>
          </div>

          {/* Tier description */}
          <div className="text-xs text-text-muted">
            {UI_CONFIG.TIER_DESCRIPTIONS[currentPermission.tier]}
          </div>
        </div>

        {hasPolicyMetadata && (
          <div className="rounded-lg border border-border bg-bg p-4">
            <div className="mb-2 flex items-center justify-between">
              <div className="text-xs font-medium uppercase text-text-muted">
                {t("playground:acp.policySnapshot", "Policy Snapshot")}
              </div>
              {currentPermission.provenance_summary && (
                <Button
                  type="link"
                  size="small"
                  className="px-0"
                  onClick={() => setShowPolicyDetails((value) => !value)}
                >
                  {showPolicyDetails
                    ? t("playground:acp.hidePolicyDetails", "Hide details")
                    : t("playground:acp.showPolicyDetails", "Show details")}
                </Button>
              )}
            </div>
            <div className="space-y-1 text-xs text-text-muted">
              {currentPermission.approval_requirement && (
                <div>
                  <span className="font-medium text-text">
                    {t("playground:acp.approvalRequirement", "Approval")}:{" "}
                  </span>
                  <span>{currentPermission.approval_requirement}</span>
                </div>
              )}
              {currentPermission.governance_reason && (
                <div>
                  <span className="font-medium text-text">
                    {t("playground:acp.governanceReason", "Reason")}:{" "}
                  </span>
                  <span>{currentPermission.governance_reason}</span>
                </div>
              )}
              {currentPermission.runtime_narrowing_reason && (
                <div>
                  <span className="font-medium text-text">
                    {t("playground:acp.runtimeConstraint", "Runtime constraint")}:{" "}
                  </span>
                  <span>{currentPermission.runtime_narrowing_reason}</span>
                </div>
              )}
              {currentPermission.policy_snapshot_fingerprint && (
                <div>
                  <span className="font-medium text-text">
                    {t("playground:acp.snapshotFingerprint", "Policy version")}:{" "}
                  </span>
                  <span className="font-mono">
                    {currentPermission.policy_snapshot_fingerprint.slice(0, 12)}
                  </span>
                </div>
              )}
            </div>
            {showPolicyDetails && currentPermission.provenance_summary && (
              <div className="mt-3 rounded bg-surface2 p-2 space-y-1">
                {typeof currentPermission.provenance_summary === "object" ? (
                  <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
                    {Object.entries(currentPermission.provenance_summary as Record<string, unknown>)
                      .filter(([, v]) => v !== null && v !== undefined && v !== "")
                      .map(([key, value]) => (
                        <React.Fragment key={key}>
                          <dt className="font-medium text-text capitalize">
                            {key.replace(/_/g, " ")}
                          </dt>
                          <dd className="text-text-muted font-mono truncate">
                            {typeof value === "string" && value.length > 40
                              ? `${value.slice(0, 40)}...`
                              : String(value)}
                          </dd>
                        </React.Fragment>
                      ))}
                  </dl>
                ) : (
                  <pre className="overflow-auto text-xs text-text">
                    {JSON.stringify(currentPermission.provenance_summary, null, 2)}
                  </pre>
                )}
              </div>
            )}
          </div>
        )}

        {/* Timeout progress */}
        <div className="space-y-1">
          <div className="flex items-center justify-between text-xs text-text-muted">
            <span className="flex items-center gap-1">
              <Clock className="h-3 w-3" />
              {t("playground:acp.timeRemaining", "Time remaining")}
            </span>
            <span>{Math.ceil(timeRemaining / 1000)}s</span>
          </div>
          <Progress
            percent={progressPercent}
            showInfo={false}
            strokeColor={
              progressPercent < 20
                ? "rgb(var(--color-danger))"
                : progressPercent < 50
                  ? "rgb(var(--color-warn))"
                  : "rgb(var(--color-success))"
            }
            size="small"
          />
        </div>

        {/* Batch approve checkbox */}
        {currentPermission.tier !== "individual" && (
          <Checkbox
            checked={batchApprove}
            onChange={(e) => setBatchApprove(e.target.checked)}
          >
            <span className="text-sm text-text">
              {currentPermission.tier === "batch"
                ? t("playground:acp.batchApproveWrite",
                    "Automatically approve similar file-write operations for the rest of this session")
                : t("playground:acp.batchApproveGeneral",
                    "Automatically approve similar operations for the rest of this session")
              }
            </span>
          </Checkbox>
        )}

        {/* Actions */}
        <div className="flex gap-3">
          <Button
            danger
            icon={<XCircle className="h-4 w-4" />}
            onClick={handleDeny}
            className="flex-1"
          >
            {t("playground:acp.deny", "Deny")}
          </Button>
          <Button
            type="primary"
            icon={<CheckCircle className="h-4 w-4" />}
            onClick={handleApprove}
            className="flex-1"
          >
            {t("playground:acp.approve", "Approve")}
          </Button>
        </div>
      </div>
    </div>
  )
}
