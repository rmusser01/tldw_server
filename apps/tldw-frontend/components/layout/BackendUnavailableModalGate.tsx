import React from "react"
import { Button, Modal } from "antd"
import type { TFunction } from "i18next"

import type { BackendUnreachableDetail } from "@/services/request-events"

export type BackendUnavailableModalGateProps = {
  backendUnavailableDetail: BackendUnreachableDetail | null
  fatalBackendRecoveryActive: boolean
  isChecking: boolean
  onClose: () => void
  onConsumeHiddenDetail?: () => void
  onOpenHealth: () => void
  onRetry: () => void
  presentation?: "modal" | "inline"
  t: TFunction
}

export const BackendUnavailableModalGate: React.FC<
  BackendUnavailableModalGateProps
> = ({
  backendUnavailableDetail,
  fatalBackendRecoveryActive,
  isChecking,
  onClose,
  onConsumeHiddenDetail,
  onOpenHealth,
  onRetry,
  presentation = "modal",
  t
}) => {
  const inlineTitleId = React.useId()

  React.useEffect(() => {
    if (fatalBackendRecoveryActive && backendUnavailableDetail) {
      onConsumeHiddenDetail?.()
    }
  }, [backendUnavailableDetail, fatalBackendRecoveryActive, onConsumeHiddenDetail])

  const title = t(
    "sidepanel:connectionBanner.unreachableTitle",
    "Can't reach your tldw server"
  )
  const body = t(
    "sidepanel:connectionBanner.unreachableBody",
    "Check that your server is running and accessible."
  )

  if (presentation === "inline") {
    if (!backendUnavailableDetail || fatalBackendRecoveryActive) return null

    return (
      <div
        role="status"
        aria-labelledby={inlineTitleId}
        aria-live="polite"
        className="fixed bottom-4 right-4 z-[1000] w-[min(24rem,calc(100vw-2rem))] rounded-md border border-border bg-surface p-4 text-text shadow-lg sm:bottom-auto sm:top-24"
      >
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p id={inlineTitleId} className="text-sm font-semibold">
              {title}
            </p>
            <p className="mt-1 text-sm text-text-muted">{body}</p>
            <p className="mt-2 break-all text-xs text-text-subtle">
              {`${backendUnavailableDetail.message} (${backendUnavailableDetail.method} ${backendUnavailableDetail.path})`}
            </p>
          </div>
          <Button type="text" size="small" onClick={onClose}>
            {t("common:dismiss", "Dismiss")}
          </Button>
        </div>
        <div className="mt-3 flex justify-end gap-2">
          <Button size="small" onClick={onOpenHealth}>
            {t(
              "settings:healthSummary.diagnostics",
              "Health & diagnostics"
            )}
          </Button>
          <Button
            size="small"
            type="primary"
            loading={isChecking}
            onClick={onRetry}
            className="!bg-primaryStrong !text-white hover:!bg-primaryStrong hover:brightness-95"
          >
            {t("common:retry", "Retry")}
          </Button>
        </div>
      </div>
    )
  }

  return (
    <Modal
      title={title}
      open={Boolean(backendUnavailableDetail) && !fatalBackendRecoveryActive}
      onCancel={onClose}
      maskClosable={false}
      destroyOnHidden
      footer={[
        <Button key="dismiss" onClick={onClose}>
          {t("common:dismiss", "Dismiss")}
        </Button>,
        <Button key="health" onClick={onOpenHealth}>
          {t(
            "settings:healthSummary.diagnostics",
            "Health & diagnostics"
          )}
        </Button>,
        <Button
          key="retry"
          type="primary"
          loading={isChecking}
          onClick={onRetry}
        >
          {t("common:retry", "Retry")}
        </Button>
      ]}
    >
      <p className="text-sm text-text">
        {body}
      </p>
      {backendUnavailableDetail && (
        <p className="mt-2 break-all text-xs text-text-subtle">
          {`${backendUnavailableDetail.message} (${backendUnavailableDetail.method} ${backendUnavailableDetail.path})`}
        </p>
      )}
    </Modal>
  )
}

export default BackendUnavailableModalGate
