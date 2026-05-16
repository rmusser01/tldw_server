import { useEffect, useMemo, useState } from "react"
import { Button, Card, Descriptions, Divider, Empty, Input, List, Modal, Radio, Space, Typography } from "antd"
import {
  ProductStateAlert as Alert,
  ProductStateBadge as Tag,
} from "@/components/Option/productStatePrimitives"
import { BLOCKED_STATE_LABEL } from "@/design-system"

import {
  checkGovernancePackUpdates,
  dryRunGovernancePack,
  dryRunGovernancePackUpgrade,
  dryRunGovernancePackSourceCandidate,
  dryRunGovernancePackSourceUpgrade,
  executeGovernancePackUpgrade,
  executeGovernancePackSourceUpgrade,
  getGovernancePackDetail,
  importGovernancePack,
  importGovernancePackSourceCandidate,
  listGovernancePackUpgradeHistory,
  listGovernancePacks,
  prepareGovernancePackSourceCandidate,
  prepareGovernancePackUpgradeCandidate,
  type McpHubGovernancePackDetail,
  type McpHubGovernancePackDocument,
  type McpHubGovernancePackDryRunReport,
  type McpHubGovernancePackGitRefKind,
  type McpHubGovernancePackSourceCandidate,
  type McpHubGovernancePackSourceRequest,
  type McpHubGovernancePackSourceUpdateCheck,
  type McpHubGovernancePackSourceUpgradePrepareResponse,
  type McpHubGovernancePackUpgradeHistoryEntry,
  type McpHubGovernancePackUpgradeObjectDiff,
  type McpHubGovernancePackUpgradePlan,
  type McpHubGovernancePackSummary
} from "@/services/tldw/mcp-hub"

const DEFAULT_PACK_JSON = JSON.stringify(
  {
    manifest: {},
    profiles: [],
    approvals: [],
    personas: [],
    assignments: []
  },
  null,
  2
)
const DETAIL_LOAD_ERROR = "Failed to load pack details."

const getVerdictColor = (verdict?: McpHubGovernancePackDryRunReport["verdict"]) => {
  if (verdict === "importable") return "green"
  if (verdict === "blocked") return "red"
  return "default"
}

const describeItems = (values: string[], emptyLabel: string) => {
  if (!values.length) {
    return <Typography.Text type="secondary">{emptyLabel}</Typography.Text>
  }
  return (
    <Space wrap>
      {values.map((value) => (
        <Tag key={value}>{value}</Tag>
      ))}
    </Space>
  )
}

const getInstallStateTag = (pack: Pick<McpHubGovernancePackSummary, "is_active_install">) =>
  pack.is_active_install ? (
    <Tag color="green">Active install</Tag>
  ) : (
    <Tag color="default">Inactive install</Tag>
  )

const describeUpgradeDiffs = (diffs: McpHubGovernancePackUpgradeObjectDiff[]) =>
  diffs.map((diff) => `${diff.object_type}:${diff.source_object_id}`)

const sourceRequestKey = (source: McpHubGovernancePackSourceRequest) =>
  JSON.stringify({
    source_type: source.source_type,
    local_path: source.local_path ?? null,
    repo_url: source.repo_url ?? null,
    ref: source.ref ?? null,
    ref_kind: source.ref_kind ?? null,
    subpath: source.subpath ?? null
  })

const getSourceTypeTag = (sourceType?: string | null) => {
  if (sourceType === "git") return <Tag color="geekblue">Git Source</Tag>
  if (sourceType === "local_path") return <Tag color="purple">Local Path</Tag>
  return null
}

const getVerificationTag = (pack: Pick<McpHubGovernancePackSummary, "source_type" | "source_verified" | "source_verification_mode">) => {
  if (pack.source_type !== "git") {
    return null
  }
  if (pack.source_verified) {
    return <Tag color="green">{pack.source_verification_mode ? "Verified Commit" : "Verified Source"}</Tag>
  }
  if (pack.source_verification_mode) {
    return <Tag color="orange">Verification Failed</Tag>
  }
  return <Tag color="default">Unverified Source</Tag>
}

const describeVerificationWarning = (code?: string | null) => {
  switch (code) {
    case "signer_rotated_trusted":
      return "Signer rotated"
    case "signer_revoked":
      return "Signer revoked"
    case "unknown_previous_signer":
      return "Previous signer unknown"
    default:
      return code ?? null
  }
}

const renderVerificationDetails = (
  pack: Pick<
    McpHubGovernancePackSummary,
    | "source_type"
    | "signer_fingerprint"
    | "signer_identity"
    | "verified_object_type"
    | "verification_result_code"
    | "verification_warning_code"
  >
) => {
  if (pack.source_type !== "git") {
    return null
  }
  const signerFingerprint = pack.signer_fingerprint ?? null
  const signerIdentity = pack.signer_identity ?? null
  const verifiedObjectType = pack.verified_object_type ?? null
  const resultCode = pack.verification_result_code ?? null
  const warningCode = describeVerificationWarning(pack.verification_warning_code)
  if (!signerFingerprint && !signerIdentity && !verifiedObjectType && !resultCode && !warningCode) {
    return null
  }
  return (
    <details>
      <summary>Verification details</summary>
      <Descriptions bordered column={1} size="small" style={{ marginTop: 8 }}>
        {signerFingerprint ? (
          <Descriptions.Item label="Signer fingerprint">
            <Typography.Text code>{signerFingerprint}</Typography.Text>
          </Descriptions.Item>
        ) : null}
        {signerIdentity ? (
          <Descriptions.Item label="Signer identity">
            <Typography.Text>{signerIdentity}</Typography.Text>
          </Descriptions.Item>
        ) : null}
        {verifiedObjectType ? (
          <Descriptions.Item label="Verified object type">
            <Typography.Text code>{verifiedObjectType}</Typography.Text>
          </Descriptions.Item>
        ) : null}
        {resultCode ? (
          <Descriptions.Item label="Verification result">
            <Typography.Text code>{resultCode}</Typography.Text>
          </Descriptions.Item>
        ) : null}
        {warningCode ? (
          <Descriptions.Item label="Verification warning">
            <Tag color="orange">{warningCode}</Tag>
          </Descriptions.Item>
        ) : null}
      </Descriptions>
    </details>
  )
}

const describeUpdateStatus = (updateCheck: McpHubGovernancePackSourceUpdateCheck) => {
  const warningText = describeVerificationWarning(updateCheck.verification_warning_code)
  const signerText = updateCheck.signer_fingerprint ? ` (${updateCheck.signer_fingerprint})` : ""
  if (updateCheck.status === "newer_version_available") {
    return {
      type: "success" as const,
      message: `Newer version available: ${String(updateCheck.candidate_manifest?.pack_version ?? "")}`,
      description:
        `Current ${updateCheck.installed_manifest.pack_version} -> candidate ${String(
          updateCheck.candidate_manifest?.pack_version ?? ""
        )}` + (warningText ? `. ${warningText}${signerText}.` : "")
    }
  }
  if (updateCheck.status === "source_drift_same_version") {
    return {
      type: "warning" as const,
      message: "Source drift detected at the same version",
      description:
        `Tracked source now resolves to ${String(updateCheck.source_commit_resolved ?? "a different commit")} without a version bump.` +
        (warningText ? ` ${warningText}${signerText}.` : "")
    }
  }
  return {
    type: "info" as const,
    message: "No update available",
    description: `Tracked source still resolves to ${updateCheck.installed_manifest.pack_version}.`
  }
}

export const GovernancePacksTab = () => {
  const [packs, setPacks] = useState<McpHubGovernancePackSummary[]>([])
  const [selectedPackId, setSelectedPackId] = useState<number | null>(null)
  const [selectedPack, setSelectedPack] = useState<McpHubGovernancePackDetail | null>(null)
  const [upgradeHistory, setUpgradeHistory] = useState<McpHubGovernancePackUpgradeHistoryEntry[]>([])
  const [report, setReport] = useState<McpHubGovernancePackDryRunReport | null>(null)
  const [upgradePlan, setUpgradePlan] = useState<McpHubGovernancePackUpgradePlan | null>(null)
  const [packJson, setPackJson] = useState(DEFAULT_PACK_JSON)
  const [localSourcePath, setLocalSourcePath] = useState("")
  const [gitRepoUrl, setGitRepoUrl] = useState("")
  const [gitRef, setGitRef] = useState("main")
  const [gitRefKind, setGitRefKind] = useState<McpHubGovernancePackGitRefKind>("branch")
  const [gitSubpath, setGitSubpath] = useState("packs/researcher")
  const [preparedSourceCandidate, setPreparedSourceCandidate] = useState<McpHubGovernancePackSourceCandidate | null>(null)
  const [preparedSourceRequestKey, setPreparedSourceRequestKey] = useState<string | null>(null)
  const [updateCheck, setUpdateCheck] = useState<McpHubGovernancePackSourceUpdateCheck | null>(null)
  const [upgradeCandidate, setUpgradeCandidate] = useState<McpHubGovernancePackSourceUpgradePrepareResponse | null>(null)
  const [loadingInventory, setLoadingInventory] = useState(false)
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [loadingHistory, setLoadingHistory] = useState(false)
  const [previewing, setPreviewing] = useState(false)
  const [previewingSource, setPreviewingSource] = useState(false)
  const [previewingUpgrade, setPreviewingUpgrade] = useState(false)
  const [importing, setImporting] = useState(false)
  const [importingSource, setImportingSource] = useState(false)
  const [checkingUpdates, setCheckingUpdates] = useState(false)
  const [executingUpgrade, setExecutingUpgrade] = useState(false)
  const [upgradeModalOpen, setUpgradeModalOpen] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)

  const parsedPack = useMemo<McpHubGovernancePackDocument | null>(() => {
    try {
      const parsed = JSON.parse(packJson) as McpHubGovernancePackDocument
      return {
        manifest: parsed?.manifest ?? {},
        profiles: Array.isArray(parsed?.profiles) ? parsed.profiles : [],
        approvals: Array.isArray(parsed?.approvals) ? parsed.approvals : [],
        personas: Array.isArray(parsed?.personas) ? parsed.personas : [],
        assignments: Array.isArray(parsed?.assignments) ? parsed.assignments : []
      }
    } catch {
      return null
    }
  }, [packJson])

  const loadInventory = async () => {
    setLoadingInventory(true)
    setErrorMessage(null)
    try {
      const rows = await listGovernancePacks()
      const safeRows = Array.isArray(rows) ? rows : []
      setPacks(safeRows)
      setSelectedPackId((current) => {
        if (safeRows.some((row) => row.id === current)) {
          return current
        }
        return safeRows[0]?.id ?? null
      })
    } catch {
      setPacks([])
      setSelectedPackId(null)
      setSelectedPack(null)
      setUpgradeHistory([])
      setErrorMessage("Failed to load governance packs.")
    } finally {
      setLoadingInventory(false)
    }
  }

  useEffect(() => {
    void loadInventory()
  }, [])

  useEffect(() => {
    let cancelled = false
    const loadDetail = async () => {
      if (!selectedPackId) {
        setSelectedPack(null)
        setUpgradeHistory([])
        setErrorMessage((current) => (current === DETAIL_LOAD_ERROR ? null : current))
        return
      }
      setLoadingDetail(true)
      setLoadingHistory(true)
      try {
        const [detail, history] = await Promise.all([
          getGovernancePackDetail(selectedPackId),
          listGovernancePackUpgradeHistory(selectedPackId)
        ])
        if (!cancelled) {
          setSelectedPack(detail)
          setUpgradeHistory(Array.isArray(history) ? history : [])
          setErrorMessage((current) => (current === DETAIL_LOAD_ERROR ? null : current))
        }
      } catch {
        if (!cancelled) {
          setSelectedPack(null)
          setUpgradeHistory([])
          setErrorMessage(DETAIL_LOAD_ERROR)
        }
      } finally {
        if (!cancelled) {
          setLoadingDetail(false)
          setLoadingHistory(false)
        }
      }
    }
    void loadDetail()
    return () => {
      cancelled = true
    }
  }, [selectedPackId])

  useEffect(() => {
    setUpdateCheck(null)
    setUpgradeCandidate(null)
  }, [selectedPackId])

  const handlePreview = async () => {
    setSuccessMessage(null)
    if (!parsedPack) {
      setErrorMessage("Governance pack JSON must be valid JSON.")
      setReport(null)
      return
    }
    setPreviewing(true)
    setErrorMessage(null)
    try {
      const response = await dryRunGovernancePack({
        owner_scope_type: "user",
        pack: parsedPack
      })
      setReport(response.report)
    } catch {
      setReport(null)
      setErrorMessage("Failed to preview governance pack compatibility.")
    } finally {
      setPreviewing(false)
    }
  }

  const previewSourceCandidate = async (source: McpHubGovernancePackSourceRequest, failureMessage: string) => {
    setPreviewingSource(true)
    setErrorMessage(null)
    setSuccessMessage(null)
    try {
      const prepared = await prepareGovernancePackSourceCandidate({ source })
      setPreparedSourceCandidate(prepared.candidate)
      setPreparedSourceRequestKey(sourceRequestKey(source))
      const response = await dryRunGovernancePackSourceCandidate({
        owner_scope_type: "user",
        candidate_id: prepared.candidate.id
      })
      setReport(response.report)
    } catch {
      setPreparedSourceCandidate(null)
      setPreparedSourceRequestKey(null)
      setReport(null)
      setErrorMessage(failureMessage)
    } finally {
      setPreviewingSource(false)
    }
  }

  const importSourceCandidate = async (source: McpHubGovernancePackSourceRequest, failureMessage: string) => {
    setImportingSource(true)
    setErrorMessage(null)
    setSuccessMessage(null)
    try {
      const requestKey = sourceRequestKey(source)
      const prepared =
        preparedSourceCandidate && preparedSourceRequestKey === requestKey
          ? { candidate: preparedSourceCandidate }
          : await prepareGovernancePackSourceCandidate({ source })
      setPreparedSourceCandidate(prepared.candidate)
      setPreparedSourceRequestKey(requestKey)
      const response = await importGovernancePackSourceCandidate({
        owner_scope_type: "user",
        candidate_id: prepared.candidate.id
      })
      setReport(response.report)
      setSelectedPackId(response.governance_pack_id)
      setSuccessMessage(`Imported ${response.report.manifest.title} from source.`)
      await loadInventory()
    } catch {
      setErrorMessage(failureMessage)
    } finally {
      setImportingSource(false)
    }
  }

  const handlePreviewLocalSource = async () => {
    const resolvedPath = localSourcePath.trim()
    if (!resolvedPath) {
      setErrorMessage("Local source path is required.")
      return
    }
    await previewSourceCandidate(
      {
        source_type: "local_path",
        local_path: resolvedPath
      },
      "Failed to preview local governance-pack source."
    )
  }

  const handleImportLocalSource = async () => {
    const resolvedPath = localSourcePath.trim()
    if (!resolvedPath) {
      setErrorMessage("Local source path is required.")
      return
    }
    await importSourceCandidate(
      {
        source_type: "local_path",
        local_path: resolvedPath
      },
      "Failed to import local governance-pack source."
    )
  }

  const handlePreviewGitSource = async () => {
    const repoUrl = gitRepoUrl.trim()
    if (!repoUrl) {
      setErrorMessage("Git repository URL is required.")
      return
    }
    await previewSourceCandidate(
      {
        source_type: "git",
        repo_url: repoUrl,
        ref: gitRef.trim() || null,
        ref_kind: gitRefKind,
        subpath: gitSubpath.trim() || null
      },
      "Failed to preview Git governance-pack source."
    )
  }

  const handleImportGitSource = async () => {
    const repoUrl = gitRepoUrl.trim()
    if (!repoUrl) {
      setErrorMessage("Git repository URL is required.")
      return
    }
    await importSourceCandidate(
      {
        source_type: "git",
        repo_url: repoUrl,
        ref: gitRef.trim() || null,
        ref_kind: gitRefKind,
        subpath: gitSubpath.trim() || null
      },
      "Failed to import Git governance-pack source."
    )
  }

  const handleImport = async () => {
    setSuccessMessage(null)
    if (!parsedPack) {
      setErrorMessage("Governance pack JSON must be valid JSON.")
      return
    }
    setImporting(true)
    setErrorMessage(null)
    try {
      const response = await importGovernancePack({
        owner_scope_type: "user",
        pack: parsedPack
      })
      setReport(response.report)
      setSelectedPackId(response.governance_pack_id)
      setSuccessMessage(`Imported ${response.report.manifest.title}.`)
      await loadInventory()
    } catch {
      setErrorMessage("Failed to import governance pack.")
    } finally {
      setImporting(false)
    }
  }

  const handlePreviewUpgrade = async () => {
    setSuccessMessage(null)
    if (!selectedPackId) {
      setErrorMessage("Select an installed governance pack before previewing an upgrade.")
      setUpgradePlan(null)
      return
    }
    if (!parsedPack) {
      setErrorMessage("Governance pack JSON must be valid JSON.")
      setUpgradePlan(null)
      return
    }
    setUpgradeModalOpen(true)
    setPreviewingUpgrade(true)
    setErrorMessage(null)
    try {
      const response = await dryRunGovernancePackUpgrade({
        source_governance_pack_id: selectedPackId,
        owner_scope_type: "user",
        pack: parsedPack
      })
      setUpgradePlan(response.plan)
    } catch {
      setUpgradePlan(null)
      setUpgradeModalOpen(false)
      setErrorMessage("Failed to preview governance pack upgrade.")
    } finally {
      setPreviewingUpgrade(false)
    }
  }

  const handleCheckForUpdates = async () => {
    if (!selectedPackId) {
      return
    }
    setCheckingUpdates(true)
    setErrorMessage(null)
    setSuccessMessage(null)
    try {
      const response = await checkGovernancePackUpdates(selectedPackId)
      setUpdateCheck(response)
    } catch {
      setUpdateCheck(null)
      setErrorMessage("Failed to check governance-pack updates.")
    } finally {
      setCheckingUpdates(false)
    }
  }

  const handlePreviewSourceUpgrade = async () => {
    if (!selectedPackId) {
      return
    }
    setPreviewingUpgrade(true)
    setErrorMessage(null)
    setSuccessMessage(null)
    try {
      const prepared = await prepareGovernancePackUpgradeCandidate(selectedPackId)
      setUpgradeCandidate(prepared)
      const response = await dryRunGovernancePackSourceUpgrade({
        source_governance_pack_id: selectedPackId,
        owner_scope_type: "user",
        candidate_id: prepared.candidate.id
      })
      setUpgradePlan(response.plan)
      setUpgradeModalOpen(true)
    } catch {
      setUpgradePlan(null)
      setUpgradeCandidate(null)
      setUpgradeModalOpen(false)
      setErrorMessage("Failed to preview governance-pack source upgrade.")
    } finally {
      setPreviewingUpgrade(false)
    }
  }

  const handleExecuteUpgrade = async () => {
    setSuccessMessage(null)
    if (!selectedPackId || !upgradePlan) {
      return
    }
    setExecutingUpgrade(true)
    setErrorMessage(null)
    try {
      const response = upgradeCandidate
        ? await executeGovernancePackSourceUpgrade({
            source_governance_pack_id: selectedPackId,
            owner_scope_type: "user",
            candidate_id: upgradeCandidate.candidate.id,
            planner_inputs_fingerprint: upgradePlan.planner_inputs_fingerprint,
            adapter_state_fingerprint: upgradePlan.adapter_state_fingerprint
          })
        : parsedPack
          ? await executeGovernancePackUpgrade({
              source_governance_pack_id: selectedPackId,
              owner_scope_type: "user",
              planner_inputs_fingerprint: upgradePlan.planner_inputs_fingerprint,
              adapter_state_fingerprint: upgradePlan.adapter_state_fingerprint,
              pack: parsedPack
            })
          : null
      if (!response) {
        return
      }
      setUpgradeModalOpen(false)
      setUpgradePlan(null)
      setUpgradeCandidate(null)
      setSelectedPackId(response.target_governance_pack_id)
      setSuccessMessage(
        `Executed upgrade ${response.from_pack_version} -> ${response.to_pack_version}.`
      )
      await loadInventory()
    } catch {
      setErrorMessage("Failed to execute governance pack upgrade.")
    } finally {
      setExecutingUpgrade(false)
    }
  }

  const canImport = !importing && report?.verdict === "importable" && Boolean(parsedPack)
  const addedDiffs = useMemo(
    () => (upgradePlan?.object_diff ?? []).filter((diff) => diff.change_type === "added"),
    [upgradePlan]
  )
  const modifiedDiffs = useMemo(
    () => (upgradePlan?.object_diff ?? []).filter((diff) => diff.change_type === "modified"),
    [upgradePlan]
  )
  const removedDiffs = useMemo(
    () => (upgradePlan?.object_diff ?? []).filter((diff) => diff.change_type === "removed"),
    [upgradePlan]
  )
  const blockingConflicts = useMemo(
    () => [...(upgradePlan?.structural_conflicts ?? []), ...(upgradePlan?.behavioral_conflicts ?? [])],
    [upgradePlan]
  )
  const canExecuteUpgrade =
    Boolean(selectedPackId) &&
    Boolean(upgradePlan?.upgradeable) &&
    (Boolean(upgradeCandidate) || Boolean(parsedPack)) &&
    !executingUpgrade

  return (
    <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
      <Typography.Text type="secondary">
        Governance packs provide portable MCP Hub policy templates that can be previewed before import.
      </Typography.Text>
      {errorMessage ? <Alert type="error" title={errorMessage} showIcon /> : null}
      {successMessage ? <Alert type="success" title={successMessage} showIcon /> : null}

      <Space align="start" size="middle" style={{ width: "100%", justifyContent: "space-between" }}>
        <Card title="Installed Packs" style={{ flex: 1, minWidth: 320 }}>
          <List
            bordered
            loading={loadingInventory}
            dataSource={packs}
            locale={{ emptyText: <Empty description="No governance packs imported yet" /> }}
            renderItem={(pack) => (
              <List.Item
                style={{
                  cursor: "pointer",
                  borderColor: pack.id === selectedPackId ? "var(--ant-color-primary, #1677ff)" : undefined
                }}
                onClick={() => setSelectedPackId(pack.id)}
              >
                <Space orientation="vertical" size={4} style={{ width: "100%" }}>
                  <Space wrap>
                    <Typography.Text strong>{pack.title}</Typography.Text>
                    <Tag>{pack.owner_scope_type}</Tag>
                    <Tag color="blue">{`${pack.pack_id}@${pack.pack_version}`}</Tag>
                    {getInstallStateTag(pack)}
                  </Space>
                  {pack.description ? (
                    <Typography.Text type="secondary">{pack.description}</Typography.Text>
                  ) : null}
                </Space>
              </List.Item>
            )}
          />
        </Card>

        <Card
          title="Pack Details"
          loading={loadingDetail || loadingHistory}
          style={{ flex: 1, minWidth: 320 }}
        >
          {selectedPack ? (
            <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
              <Descriptions column={1} size="small">
                <Descriptions.Item label="Pack">
                  {`${selectedPack.pack_id}@${selectedPack.pack_version}`}
                </Descriptions.Item>
                <Descriptions.Item label="Install state">
                  {getInstallStateTag(selectedPack)}
                </Descriptions.Item>
                {selectedPack.source_type ? (
                  <Descriptions.Item label="Source">
                    <Space wrap>
                      {getSourceTypeTag(selectedPack.source_type)}
                      {getVerificationTag(selectedPack)}
                    </Space>
                  </Descriptions.Item>
                ) : null}
                {selectedPack.source_location ? (
                  <Descriptions.Item label="Source location">
                    <Typography.Text code>{selectedPack.source_location}</Typography.Text>
                  </Descriptions.Item>
                ) : null}
                {selectedPack.source_ref_requested ? (
                  <Descriptions.Item label="Requested ref">
                    <Typography.Text code>{selectedPack.source_ref_requested}</Typography.Text>
                  </Descriptions.Item>
                ) : null}
                {selectedPack.source_ref_kind ? (
                  <Descriptions.Item label="Requested ref kind">
                    <Typography.Text code>{selectedPack.source_ref_kind}</Typography.Text>
                  </Descriptions.Item>
                ) : null}
                {selectedPack.source_subpath ? (
                  <Descriptions.Item label="Source subpath">
                    <Typography.Text code>{selectedPack.source_subpath}</Typography.Text>
                  </Descriptions.Item>
                ) : null}
                {selectedPack.source_commit_resolved ? (
                  <Descriptions.Item label="Resolved commit">
                    <Typography.Text code>{selectedPack.source_commit_resolved}</Typography.Text>
                  </Descriptions.Item>
                ) : null}
                <Descriptions.Item label="Digest">
                  <Typography.Text code>{selectedPack.bundle_digest}</Typography.Text>
                </Descriptions.Item>
                {selectedPack.pack_content_digest ? (
                  <Descriptions.Item label="Content digest">
                    <Typography.Text code>{selectedPack.pack_content_digest}</Typography.Text>
                  </Descriptions.Item>
                ) : null}
                <Descriptions.Item label="Imported Objects">
                  {describeItems(
                    selectedPack.imported_objects.map(
                      (item) => `${item.object_type}:${item.source_object_id}`
                    ),
                    "No imported objects recorded."
                  )}
                </Descriptions.Item>
              </Descriptions>

              {selectedPack.source_type === "git" ? (
                <Space orientation="vertical" size="small" style={{ width: "100%" }}>
                  {renderVerificationDetails(selectedPack)}
                  <Space wrap>
                    <Button onClick={() => void handleCheckForUpdates()} loading={checkingUpdates}>
                      Check For Updates
                    </Button>
                    {updateCheck?.status === "newer_version_available" ? (
                      <Button onClick={() => void handlePreviewSourceUpgrade()} loading={previewingUpgrade}>
                        Preview Source Upgrade
                      </Button>
                    ) : null}
                  </Space>
                  {updateCheck ? (
                    <Alert
                      type={describeUpdateStatus(updateCheck).type}
                      title={describeUpdateStatus(updateCheck).message}
                      description={describeUpdateStatus(updateCheck).description}
                      showIcon
                    />
                  ) : null}
                </Space>
              ) : null}

              <div>
                <Typography.Title level={5} style={{ marginTop: 0 }}>
                  Upgrade History
                </Typography.Title>
                <List
                  size="small"
                  bordered
                  dataSource={upgradeHistory}
                  locale={{ emptyText: <Empty description="No upgrade history recorded." /> }}
                  renderItem={(entry) => (
                    <List.Item>
                      <Space orientation="vertical" size={2} style={{ width: "100%" }}>
                        <Space wrap>
                          <Typography.Text strong>{`${entry.from_pack_version} -> ${entry.to_pack_version}`}</Typography.Text>
                          <Tag color={entry.status === "executed" ? "green" : "default"}>
                            {entry.status}
                          </Tag>
                        </Space>
                        <Typography.Text type="secondary">
                          {`Object diffs: ${String(entry.plan_summary.object_diff_count ?? 0)}, dependency impacts: ${String(entry.plan_summary.dependency_impact_count ?? 0)}`}
                        </Typography.Text>
                      </Space>
                    </List.Item>
                  )}
                />
              </div>
            </Space>
          ) : (
            <Empty description="Select a governance pack to inspect its imported objects." />
          )}
        </Card>
      </Space>

      <Card title="Preview Import">
        <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
          <Space orientation="vertical" size={4} style={{ width: "100%" }}>
            <label htmlFor="mcp-governance-pack-json">Governance Pack JSON</label>
            <Input.TextArea
              id="mcp-governance-pack-json"
              aria-label="Governance Pack JSON"
              value={packJson}
              rows={12}
              onChange={(event) => setPackJson(event.target.value)}
              spellCheck={false}
            />
          </Space>
          <Space>
            <Button type="primary" onClick={() => void handlePreview()} loading={previewing}>
              Preview Pack
            </Button>
            <Button
              onClick={() => void handlePreviewUpgrade()}
              disabled={!selectedPackId || !parsedPack}
              loading={previewingUpgrade}
            >
              Preview Upgrade
            </Button>
            <Button onClick={() => void handleImport()} disabled={!canImport} loading={importing}>
              Import Pack
            </Button>
          </Space>
          <Divider style={{ marginBlock: 8 }} />
          <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
            <Typography.Title level={5} style={{ margin: 0 }}>
              Install From Local Path
            </Typography.Title>
            <Space orientation="vertical" size={4} style={{ width: "100%" }}>
              <label htmlFor="mcp-governance-pack-local-path">Local Path</label>
              <Input
                id="mcp-governance-pack-local-path"
                aria-label="Local Path"
                value={localSourcePath}
                onChange={(event) => setLocalSourcePath(event.target.value)}
                placeholder="/srv/packs/researcher-pack"
              />
            </Space>
            <Space>
              <Button onClick={() => void handlePreviewLocalSource()} loading={previewingSource}>
                Preview Local Source
              </Button>
              <Button onClick={() => void handleImportLocalSource()} loading={importingSource}>
                Import Local Source
              </Button>
            </Space>
          </Space>
          <Divider style={{ marginBlock: 8 }} />
          <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
            <Typography.Title level={5} style={{ margin: 0 }}>
              Install From Git Source
            </Typography.Title>
            <Space orientation="vertical" size={4} style={{ width: "100%" }}>
              <label htmlFor="mcp-governance-pack-git-url">Git Repository URL</label>
              <Input
                id="mcp-governance-pack-git-url"
                aria-label="Git Repository URL"
                value={gitRepoUrl}
                onChange={(event) => setGitRepoUrl(event.target.value)}
                placeholder="https://github.com/example/researcher-pack.git"
              />
            </Space>
            <Space orientation="vertical" size={4} style={{ width: "100%" }}>
              <label>Git Ref Kind</label>
              <Radio.Group
                value={gitRefKind}
                onChange={(event) => setGitRefKind(event.target.value as McpHubGovernancePackGitRefKind)}
              >
                <Radio.Button value="branch">Branch</Radio.Button>
                <Radio.Button value="tag">Tag</Radio.Button>
                <Radio.Button value="commit">Commit</Radio.Button>
              </Radio.Group>
            </Space>
            <Space orientation="vertical" size={4} style={{ width: "100%" }}>
              <label htmlFor="mcp-governance-pack-git-ref">Git Ref</label>
              <Input
                id="mcp-governance-pack-git-ref"
                aria-label="Git Ref"
                value={gitRef}
                onChange={(event) => setGitRef(event.target.value)}
                placeholder="main"
              />
            </Space>
            <Space orientation="vertical" size={4} style={{ width: "100%" }}>
              <label htmlFor="mcp-governance-pack-git-subpath">Git Subpath</label>
              <Input
                id="mcp-governance-pack-git-subpath"
                aria-label="Git Subpath"
                value={gitSubpath}
                onChange={(event) => setGitSubpath(event.target.value)}
                placeholder="packs/researcher"
              />
            </Space>
            <Space>
              <Button onClick={() => void handlePreviewGitSource()} loading={previewingSource}>
                Preview Git Source
              </Button>
              <Button onClick={() => void handleImportGitSource()} loading={importingSource}>
                Import Git Source
              </Button>
            </Space>
            {preparedSourceCandidate ? (
              <Space orientation="vertical" size="small" style={{ width: "100%" }}>
                <Descriptions bordered column={1} size="small">
                  <Descriptions.Item label="Prepared candidate">
                    <Typography.Text code>{preparedSourceCandidate.source_location}</Typography.Text>
                  </Descriptions.Item>
                  {preparedSourceCandidate.source_commit_resolved ? (
                    <Descriptions.Item label="Prepared commit">
                      <Typography.Text code>{preparedSourceCandidate.source_commit_resolved}</Typography.Text>
                    </Descriptions.Item>
                  ) : null}
                  {preparedSourceCandidate.source_ref_kind ? (
                    <Descriptions.Item label="Prepared ref kind">
                      <Typography.Text code>{preparedSourceCandidate.source_ref_kind}</Typography.Text>
                    </Descriptions.Item>
                  ) : null}
                </Descriptions>
                {renderVerificationDetails(preparedSourceCandidate)}
              </Space>
            ) : null}
          </Space>
          {report ? (
            <Descriptions bordered column={1} size="small">
              <Descriptions.Item label="Pack">{report.manifest.title}</Descriptions.Item>
              <Descriptions.Item label="Verdict">
                <Tag color={getVerdictColor(report.verdict)}>
                  {report.verdict === "importable" ? "Importable" : BLOCKED_STATE_LABEL}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Resolved capabilities">
                {describeItems(report.resolved_capabilities, "None")}
              </Descriptions.Item>
              <Descriptions.Item label="Unresolved capabilities">
                {describeItems(report.unresolved_capabilities, "None")}
              </Descriptions.Item>
              <Descriptions.Item label="Warnings">
                {describeItems(report.warnings, "None")}
              </Descriptions.Item>
              <Descriptions.Item label="Blocked objects">
                {describeItems(report.blocked_objects, "None")}
              </Descriptions.Item>
            </Descriptions>
          ) : null}
        </Space>
      </Card>

      <Modal
        title="Governance Pack Upgrade"
        open={upgradeModalOpen}
        onCancel={() => {
          setUpgradeModalOpen(false)
          setUpgradePlan(null)
        }}
        onOk={() => void handleExecuteUpgrade()}
        okText="Execute Upgrade"
        okButtonProps={{ disabled: !canExecuteUpgrade }}
        confirmLoading={executingUpgrade}
      >
        {upgradePlan ? (
          <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
            <Descriptions bordered column={1} size="small">
              <Descriptions.Item label="Source pack">
                {String(upgradePlan.source_manifest.pack_version ?? "")}
              </Descriptions.Item>
              <Descriptions.Item label="Target pack">
                {String(upgradePlan.target_manifest.pack_version ?? "")}
              </Descriptions.Item>
              <Descriptions.Item label="Upgradeable">
                <Tag color={upgradePlan.upgradeable ? "green" : "red"}>
                  {upgradePlan.upgradeable ? "Ready to execute" : BLOCKED_STATE_LABEL}
                </Tag>
              </Descriptions.Item>
              {upgradeCandidate ? (
                <Descriptions.Item label="Prepared source candidate">
                  <Typography.Text code>{upgradeCandidate.candidate.source_location}</Typography.Text>
                </Descriptions.Item>
              ) : null}
              {upgradeCandidate?.candidate.source_commit_resolved ? (
                <Descriptions.Item label="Prepared commit">
                  <Typography.Text code>{upgradeCandidate.candidate.source_commit_resolved}</Typography.Text>
                </Descriptions.Item>
              ) : null}
              {upgradeCandidate?.candidate.source_ref_kind ? (
                <Descriptions.Item label="Prepared ref kind">
                  <Typography.Text code>{upgradeCandidate.candidate.source_ref_kind}</Typography.Text>
                </Descriptions.Item>
              ) : null}
            </Descriptions>
            {upgradeCandidate ? renderVerificationDetails(upgradeCandidate.candidate) : null}

            {modifiedDiffs.length ? (
              <div>
                <Typography.Title level={5}>Modified objects</Typography.Title>
                {describeItems(describeUpgradeDiffs(modifiedDiffs), "None")}
              </div>
            ) : null}
            {addedDiffs.length ? (
              <div>
                <Typography.Title level={5}>Added objects</Typography.Title>
                {describeItems(describeUpgradeDiffs(addedDiffs), "None")}
              </div>
            ) : null}
            {removedDiffs.length ? (
              <div>
                <Typography.Title level={5}>Removed objects</Typography.Title>
                {describeItems(describeUpgradeDiffs(removedDiffs), "None")}
              </div>
            ) : null}
            {!upgradePlan.object_diff.length ? (
              <Typography.Text type="secondary">No runtime object changes detected.</Typography.Text>
            ) : null}

            {blockingConflicts.length ? (
              <div>
                <Typography.Title level={5}>Blocking conflicts</Typography.Title>
                {describeItems(blockingConflicts, "None")}
              </div>
            ) : null}
            {upgradePlan.warnings.length ? (
              <div>
                <Typography.Title level={5}>Warnings</Typography.Title>
                {describeItems(upgradePlan.warnings, "None")}
              </div>
            ) : null}
            {upgradePlan.dependency_impact.length ? (
              <div>
                <Typography.Title level={5}>Dependency impacts</Typography.Title>
                {describeItems(
                  upgradePlan.dependency_impact.map(
                    (impact) =>
                      `${impact.dependent_type}:${impact.dependent_id} via ${impact.reference_field}`
                  ),
                  "None"
                )}
              </div>
            ) : null}
          </Space>
        ) : (
          <Typography.Text type="secondary">
            {previewingUpgrade ? "Loading upgrade preview..." : "Run an upgrade dry-run to inspect the plan."}
          </Typography.Text>
        )}
      </Modal>
    </Space>
  )
}
