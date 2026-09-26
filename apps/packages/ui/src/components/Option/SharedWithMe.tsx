import React from "react"
import { CopyOutlined, ExportOutlined, ReloadOutlined } from "@ant-design/icons"
import { Button, Card, Empty, Spin, Tag } from "antd"
import { useNavigate } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { useSharedWithMe } from "@/hooks/useSharing"
import { useSharedWorkspaceClones } from "@/hooks/useSharedWorkspaceClones"
import { getStructuredApiErrorDetail } from "@/services/tldw/api-error"
import {
  ACCESS_LEVEL_COLORS,
  ACCESS_LEVEL_LABELS,
  type AccessLevel
} from "@/types/sharing"
import { SharedWorkspaceCloneStatus } from "./SharedWorkspaceCloneStatus"

export const SharedWithMe: React.FC = () => {
  const clones = useSharedWorkspaceClones()
  const { data, isLoading, error, refetch } = useSharedWithMe(clones.scope)
  const navigate = useNavigate()
  const { t } = useTranslation("common")
  const permissionDenied =
    getStructuredApiErrorDetail(error)?.code === "sharing_permission_required"
  const shares =
    clones.status === "ready" && !permissionDenied ? (data?.items ?? []) : []
  const operations = new Map(
    clones.rows.map((row) => [row.entry.share_id, row])
  )
  const shareIds = [
    ...new Set([...shares.map((share) => share.share_id), ...operations.keys()])
  ]

  if (
    clones.status === "loading" ||
    (clones.status === "ready" && isLoading && !clones.rows.length)
  ) {
    return (
      <div
        className="flex min-h-[240px] items-center justify-center"
        aria-label={t("sharedWithMe.loading", "Loading shared workspaces")}
      >
        <Spin size="large" />
      </div>
    )
  }
  if (
    clones.status === "auth_required" ||
    clones.status === "recovery_conflict"
  ) {
    return (
      <div className="space-y-3" role="alert">
        <p>
          {clones.status === "recovery_conflict"
            ? t(
                "sharedWithMe.recoveryConflict",
                "Copy recovery belongs to another account or server. Switch back to resume. Copies have not been canceled."
              )
            : t(
                "sharedWithMe.authRequired",
                "Your account or access could not be verified. Check your connection, sign-in, and workspace permissions, then try again."
              )}
        </p>
        <Button icon={<ReloadOutlined />} onClick={() => clones.refresh()}>
          {t("sharedWithMe.refresh", "Try again")}
        </Button>
      </div>
    )
  }

  return (
    <Card title={t("sharedWithMe.title", "Shared With Me")}>
      {error && (
        <div className="mb-3" role="alert">
          <p>
            {permissionDenied
              ? t(
                  "sharedWithMe.permissionRequired",
                  "Your account does not have permission to access shared workspaces. Ask your server administrator to grant sharing.read, then try again."
                )
              : t(
                  "sharedWithMe.loadError",
                  "Could not load shared workspaces."
                )}
          </p>
          <Button icon={<ReloadOutlined />} onClick={() => void refetch()}>
            {t("sharedWithMe.refresh", "Try again")}
          </Button>
        </div>
      )}
      {!shareIds.length && !error && (
        <Empty
          description={t(
            "sharedWithMe.empty",
            "No shared workspaces available yet."
          )}
        />
      )}
      {shareIds.length > 0 && (
        <ul
          aria-label={t("sharedWithMe.list", "Shared workspaces")}
          className="m-0 list-none divide-y divide-border p-0"
        >
          {shareIds.map((shareId) => {
            const share = shares.find((item) => item.share_id === shareId)
            const row = operations.get(shareId)
            const label = share
              ? share.workspace_name?.trim() || share.workspace_id
              : t(
                  "sharedWithMe.unavailable",
                  "Shared workspace no longer available"
                )
            const access = String(share?.access_level || "")
            const knownAccess = Object.prototype.hasOwnProperty.call(
              ACCESS_LEVEL_LABELS,
              access
            )
            const canBegin =
              !row || row.issue === "rejected" || row.issue === "recovery_full"
            return (
              <li
                key={shareId}
                className="min-w-0 py-3 first:pt-0 last:pb-0"
                data-testid={`shared-workspace-row-${shareId}`}
              >
                <div className="flex min-w-0 flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                  <div className="min-w-0 flex-1 [overflow-wrap:anywhere]">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="min-w-0 font-medium">{label}</span>
                      {share && (
                        <Tag
                          color={
                            knownAccess
                              ? ACCESS_LEVEL_COLORS[access as AccessLevel]
                              : "default"
                          }
                        >
                          {knownAccess
                            ? ACCESS_LEVEL_LABELS[access as AccessLevel]
                            : access ||
                              t("sharedWithMe.unknownAccess", "Unknown access")}
                        </Tag>
                      )}
                    </div>
                    {share?.workspace_description && (
                      <p className="mb-0 mt-1 text-sm text-text-muted">
                        {share.workspace_description}
                      </p>
                    )}
                    {share && (
                      <p className="mb-0 mt-1 text-sm text-text-muted">
                        {share.owner_username?.trim()
                          ? t("sharedWithMe.owner", {
                              defaultValue: "Shared by {{name}}",
                              name: share.owner_username.trim()
                            })
                          : t(
                              "sharedWithMe.ownerUnknown",
                              "Shared by workspace owner"
                            )}
                      </p>
                    )}
                  </div>
                  {share && (
                    <div className="flex shrink-0 flex-wrap items-center gap-2">
                      <Button
                        onClick={() =>
                          navigate(`/research-workspace?shared=${shareId}`)
                        }
                        icon={<ExportOutlined />}
                        aria-label={t("sharedWithMe.openLabel", {
                          defaultValue: "Open {{name}}",
                          name: label
                        })}
                      >
                        {t("sharedWithMe.open", "Open")}
                      </Button>
                      {(!row?.operation ||
                        row.operation.status === "queued" ||
                        row.operation.status === "running") && (
                        <Button
                          aria-label={t("sharedWithMe.cloneLabel", {
                            defaultValue: "Clone {{name}}",
                            name: label
                          })}
                          disabled={!share.allow_clone || !canBegin}
                          icon={<CopyOutlined />}
                          onClick={() => clones.begin(shareId, label)}
                        >
                          {t("sharedWithMe.cloneAction", "Clone")}
                        </Button>
                      )}
                    </div>
                  )}
                </div>
                {row && (
                  <SharedWorkspaceCloneStatus
                    row={row}
                    label={label}
                    canRetry={Boolean(share?.allow_clone)}
                    onRetry={() => clones.begin(shareId, label)}
                    onRefresh={() => clones.refresh(shareId)}
                  />
                )}
              </li>
            )
          })}
        </ul>
      )}
    </Card>
  )
}
