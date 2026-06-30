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
  t
}) => {
  React.useEffect(() => {
    if (fatalBackendRecoveryActive && backendUnavailableDetail) {
      onConsumeHiddenDetail?.()
    }
  }, [backendUnavailableDetail, fatalBackendRecoveryActive, onConsumeHiddenDetail])

  return (
    <Modal
      title={t(
        "sidepanel:connectionBanner.unreachableTitle",
        "Can't reach your tldw server"
      )}
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
        {t(
          "sidepanel:connectionBanner.unreachableBody",
          "Check that your server is running and accessible."
        )}
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
