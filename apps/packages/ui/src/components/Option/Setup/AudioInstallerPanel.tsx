import React from "react"
import {
  Alert,
  Button,
  Card,
  Descriptions,
  Radio,
  Select,
  Space,
  Tag,
  Typography
} from "antd"
import { useTranslation } from "react-i18next"

import { useAudioInstaller } from "./hooks/useAudioInstaller"

const { Paragraph, Text } = Typography

const renderRemediation = (
  entry: { code?: string; action?: string; message?: string } | string
) => {
  if (typeof entry === "string") return entry
  return entry.message || entry.code || entry.action || "Unknown remediation item"
}

const renderStepLabel = (step: { label?: string; name?: string; status?: string }) =>
  step.label || step.name || step.status || "step"

const renderInstallerError = (entry: unknown) => {
  if (typeof entry === "string") return entry
  if (entry instanceof Error) return entry.message
  if (entry == null) return "Unknown installer error"
  return String(entry)
}
export const AudioInstallerPanel: React.FC = () => {
  const { t } = useTranslation(["settings", "common", "option"])
  const {
    adminGuard,
    bundleOptions,
    error,
    installStatus,
    loading,
    machineProfile,
    profileOptions,
    provisioning,
    selectedBundle,
    selectedBundleId,
    selectedProfile,
    selectedResourceProfile,
    selectedTtsChoice,
    setSelectedResourceProfile,
    handleBundleChange,
    handleResourceProfileChange,
    handleTtsChoiceChange,
    handleProvision,
    handleVerify,
    refresh,
    ttsChoiceOptions,
    verification,
    verifying
  } = useAudioInstaller()

  if (adminGuard === "forbidden") {
    return (
      <Alert
        type="warning"
        showIcon
        title={t(
          "settings:audioInstaller.forbiddenTitle",
          "Admin access required"
        )}
        description={t(
          "settings:audioInstaller.forbiddenBody",
          "Audio model installation requires server admin access."
        )}
      />
    )
  }

  if (adminGuard === "notFound") {
    return (
      <Alert
        type="info"
        showIcon
        title={t(
          "settings:audioInstaller.unavailableTitle",
          "Audio installer unavailable"
        )}
        description={t(
          "settings:audioInstaller.unavailableBody",
          "This server does not expose the admin audio installer yet."
        )}
      />
    )
  }

  return (
    <Card loading={loading} size="small">
      <Space orientation="vertical" size="middle" className="w-full">
        <div>
          <Text strong>{t("settings:audioInstaller.heading", "Recommended audio bundle")}</Text>
          <Paragraph type="secondary" className="mb-0">
            {t(
              "settings:audioInstaller.connectedServerHint",
              "Install and verify speech-model requirements on the connected server, not in this browser."
            )}
          </Paragraph>
        </div>

        {error && (
          <Alert
            type="error"
            showIcon
            title={t("settings:audioInstaller.errorTitle", "Installer request failed")}
            description={error}
            action={
              <Button size="small" onClick={() => void refresh()}>
                {t("common:retry", "Retry")}
              </Button>
            }
          />
        )}

        <Descriptions size="small" column={2}>
          <Descriptions.Item
            label={t("settings:audioInstaller.machineProfile", "Machine profile")}
          >
            {machineProfile?.apple_silicon
              ? t("settings:audioInstaller.machineAppleSilicon", "Apple Silicon")
              : machineProfile?.cuda_available
                ? t("settings:audioInstaller.machineCuda", "CUDA-capable")
                : machineProfile?.platform || "–"}
          </Descriptions.Item>
          <Descriptions.Item label={t("settings:audioInstaller.disk", "Free disk")}>
            {typeof machineProfile?.free_disk_gb === "number"
              ? `${machineProfile.free_disk_gb} GB`
              : "–"}
          </Descriptions.Item>
        </Descriptions>

        <Space orientation="vertical" size="small" className="w-full">
          <Text strong>{t("settings:audioInstaller.bundleLabel", "Bundle")}</Text>
          <Select
            value={selectedBundleId || undefined}
            onChange={handleBundleChange}
            options={bundleOptions}
            placeholder={t("settings:audioInstaller.bundlePlaceholder", "Select a bundle")}
          />
          {selectedBundle && (
            <Paragraph className="mb-0" type="secondary">
              {selectedBundle.description}
            </Paragraph>
          )}
        </Space>

        <Space orientation="vertical" size="small" className="w-full">
          <Text strong>{t("settings:audioInstaller.profileLabel", "Recommended profile")}</Text>
          <Radio.Group
            value={selectedResourceProfile || undefined}
            onChange={(event) => handleResourceProfileChange(event.target.value)}
          >
            <Space wrap>
              {profileOptions.map((option) => (
                <Radio.Button key={option.value} value={option.value}>
                  {option.label}
                </Radio.Button>
              ))}
            </Space>
          </Radio.Group>
          {selectedProfile && (
            <Space wrap>
              <Text type="secondary">{selectedProfile.description}</Text>
              {selectedProfile.resource_class && <Tag>{selectedProfile.resource_class}</Tag>}
              {typeof selectedProfile.estimated_disk_gb === "number" && (
                <Tag>{`${selectedProfile.estimated_disk_gb} GB`}</Tag>
              )}
            </Space>
          )}
        </Space>

        {ttsChoiceOptions.length > 0 && (
          <Space orientation="vertical" size="small" className="w-full">
            <Text strong>{t("settings:audioInstaller.ttsChoiceLabel", "Curated TTS")}</Text>
            <Radio.Group
              value={selectedTtsChoice || undefined}
              onChange={(event) => handleTtsChoiceChange(event.target.value)}
            >
              <Space wrap>
                {ttsChoiceOptions.map((option) => (
                  <Radio.Button key={option.value} value={option.value}>
                    {option.label}
                  </Radio.Button>
                ))}
              </Space>
            </Radio.Group>
          </Space>
        )}

        <Space wrap>
          <Button
            type="primary"
            onClick={() => void handleProvision(false)}
            loading={provisioning}
            disabled={!selectedBundleId || !selectedResourceProfile}
          >
            {t("settings:audioInstaller.provision", "Provision bundle")}
          </Button>
          <Button
            onClick={() => void handleVerify()}
            loading={verifying}
            disabled={!selectedBundleId || !selectedResourceProfile}
          >
            {t("settings:audioInstaller.verify", "Run verification")}
          </Button>
          <Button
            onClick={() => void handleProvision(true)}
            loading={provisioning}
            disabled={!selectedBundleId || !selectedResourceProfile}
          >
            {t("settings:audioInstaller.safeRerun", "Safe rerun")}
          </Button>
        </Space>

        {installStatus && (
          <Card
            size="small"
            title={t("settings:audioInstaller.installStatusTitle", "Install status")}
          >
            <Space orientation="vertical" size="small" className="w-full">
              <Text>{installStatus.status || "unknown"}</Text>
              {(installStatus.steps || []).length > 0 && (
                <ul className="mb-0 pl-5">
                  {(installStatus.steps || []).map((step, index) => (
                    <li
                      key={`${step.name || step.label || "step"}:${step.status || "unknown"}:${index}`}
                    >
                      <Text>{`${renderStepLabel(step)} (${step.status || "unknown"})`}</Text>
                    </li>
                  ))}
                </ul>
              )}
              {(installStatus.errors || []).length > 0 && (
                <Alert
                  type="warning"
                  showIcon
                  title={t("settings:audioInstaller.installErrors", "Installer issues")}
                  description={
                    <ul className="mb-0 pl-5">
                      {(installStatus.errors || []).map((entry, index) => (
                        <li key={`${renderInstallerError(entry)}:${index}`}>
                          {renderInstallerError(entry)}
                        </li>
                      ))}
                    </ul>
                  }
                />
              )}
            </Space>
          </Card>
        )}

        {verification && (
          <Card
            size="small"
            title={t("settings:audioInstaller.verificationTitle", "Verification result")}
          >
            <Space orientation="vertical" size="small" className="w-full">
              <Text>{verification.status || "unknown"}</Text>
              {(verification.targets_checked || []).length > 0 && (
                <Space wrap>
                  {(verification.targets_checked || []).map((target) => (
                    <Tag key={target}>{target}</Tag>
                  ))}
                </Space>
              )}
              {(verification.remediation_items || []).length > 0 && (
                <Alert
                  type="info"
                  showIcon
                  title={t("settings:audioInstaller.remediationTitle", "Remediation")}
                  description={
                    <ul className="mb-0 pl-5">
                      {(verification.remediation_items || []).map((entry, index) => (
                        <li key={`${renderRemediation(entry)}:${index}`}>
                          {renderRemediation(entry)}
                        </li>
                      ))}
                    </ul>
                  }
                />
              )}
            </Space>
          </Card>
        )}
      </Space>
    </Card>
  )
}

export default AudioInstallerPanel
