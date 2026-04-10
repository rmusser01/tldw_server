import React, { useState } from "react"
import { Input, Select, Skeleton, Table, Tag, Tooltip } from "antd"
import { Clipboard, Copy, Pen, X } from "lucide-react"
import { useNavigate } from "react-router-dom"
import ConnectFeatureBanner from "@/components/Common/ConnectFeatureBanner"
import FeatureEmptyState from "@/components/Common/FeatureEmptyState"
import { tagColors } from "@/utils/color"
import { usePromptWorkspace } from "./PromptWorkspaceProvider"

// ---------------------------------------------------------------------------
// Contextual help banner (dismissed via localStorage)
// ---------------------------------------------------------------------------

const COPILOT_HELP_KEY = "tldw-copilot-help-dismissed"

function CopilotHelpBanner({ t }: { t: (key: string, opts?: any) => string }) {
  const [dismissed, setDismissed] = useState(() => {
    try {
      return localStorage.getItem(COPILOT_HELP_KEY) === "1"
    } catch {
      return false
    }
  })

  if (dismissed) return null

  return (
    <div
      data-testid="copilot-help-banner"
      className="relative mb-4 rounded-md border border-primary/20 bg-primary/5 px-4 py-3 text-sm text-text-secondary"
    >
      <button
        type="button"
        aria-label={t("managePrompts.copilot.help.dismiss", { defaultValue: "Dismiss" })}
        data-testid="copilot-help-dismiss"
        onClick={() => {
          setDismissed(true)
          try {
            localStorage.setItem(COPILOT_HELP_KEY, "1")
          } catch {
            // localStorage may be unavailable
          }
        }}
        className="absolute right-2 top-2 inline-flex items-center justify-center rounded p-1 text-text-muted hover:bg-bg-muted/70 focus:outline-none focus:ring-2 focus:ring-primary"
      >
        <X className="size-4" />
      </button>
      <p className="pr-6">
        {t("managePrompts.copilot.help.description", {
          defaultValue:
            "Copilot prompts are pre-built templates provided by your server administrator. They help with common tasks like summarizing, translating, or formatting. You can customize them or copy them to your Custom prompts."
        })}
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface CopilotSegmentProps {
  tableDensity: "comfortable" | "compact" | "dense"
  // Copilot data from usePromptInteractions (lives in parent so it can feed modals)
  copilotSearchText: string
  setCopilotSearchText: (v: string) => void
  copilotKeyFilter: string
  setCopilotKeyFilter: (v: string) => void
  copilotData: any[] | undefined
  copilotStatus: "pending" | "error" | "success"
  copilotPromptKeyOptions: { label: string; value: string }[]
  filteredCopilotData: any[]
  onOpenCopilotEdit: (key: string, record: any) => void
  onCopyCopilotToCustom: (record: { key?: string; prompt?: string }) => void
  onCopyCopilotToClipboard: (record: any) => void
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function CopilotSegment({
  tableDensity,
  copilotSearchText,
  setCopilotSearchText,
  copilotKeyFilter,
  setCopilotKeyFilter,
  copilotData,
  copilotStatus,
  copilotPromptKeyOptions,
  filteredCopilotData,
  onOpenCopilotEdit,
  onCopyCopilotToCustom,
  onCopyCopilotToClipboard,
}: CopilotSegmentProps) {
  const { isOnline, t } = usePromptWorkspace()
  const navigate = useNavigate()

  if (!isOnline) {
    return (
      <ConnectFeatureBanner
        title={t("settings:managePrompts.emptyConnectTitle", {
          defaultValue: "Connect to use Prompts"
        })}
        description={t("settings:managePrompts.emptyConnectDescription", {
          defaultValue:
            "To manage reusable prompts, first connect to your tldw server."
        })}
        examples={[
          t("settings:managePrompts.emptyConnectExample1", {
            defaultValue:
              "Open Settings → tldw server to add your server URL."
          }),
          t("settings:managePrompts.emptyConnectExample2", {
            defaultValue:
              "Once connected, create custom prompts you can reuse across chats."
          })
        ]}
      />
    )
  }

  return (
    <div>
      <CopilotHelpBanner t={t} />

      {copilotStatus === "pending" && <Skeleton paragraph={{ rows: 8 }} />}

      {copilotStatus === "success" && Array.isArray(copilotData) && copilotData.length === 0 && (
        <FeatureEmptyState
          title={t("managePrompts.copilotEmptyTitle", {
            defaultValue: "No Copilot prompts available"
          })}
          description={t("managePrompts.copilotEmptyDescription", {
            defaultValue:
              "Copilot prompts are predefined templates provided by your tldw server."
          })}
          examples={[
            t("managePrompts.copilotEmptyExample1", {
              defaultValue:
                "Check your server version or configuration if you expect Copilot prompts to be available."
            }),
            t("managePrompts.copilotEmptyExample2", {
              defaultValue:
                "After updating your server, reload the extension and return to this tab."
            })
          ]}
          primaryActionLabel={t("settings:healthSummary.diagnostics", {
            defaultValue: "Open Diagnostics"
          })}
          onPrimaryAction={() => navigate("/settings/health")}
        />
      )}

      {copilotStatus === "success" && Array.isArray(copilotData) && copilotData.length > 0 && (
        <>
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <Input
              value={copilotSearchText}
              onChange={(event) => setCopilotSearchText(event.target.value)}
              allowClear
              placeholder={t("managePrompts.copilot.search.placeholder", {
                defaultValue: "Search copilot prompts..."
              })}
              style={{ width: 260 }}
              data-testid="copilot-search"
            />
            <Select
              value={copilotKeyFilter}
              onChange={(value) => setCopilotKeyFilter(value)}
              options={[
                {
                  label: t("managePrompts.copilot.filter.all", {
                    defaultValue: "All prompt types"
                  }),
                  value: "all"
                },
                ...copilotPromptKeyOptions
              ]}
              style={{ width: 200 }}
              data-testid="copilot-key-filter"
            />
          </div>

          {filteredCopilotData.length === 0 ? (
            <div className="rounded-md border border-border p-4 text-sm text-text-muted">
              {t("managePrompts.copilot.search.empty", {
                defaultValue: "No copilot prompts match the current filters."
              })}
            </div>
          ) : (
            <Table
              className={`prompts-table prompts-table-density-${tableDensity}`}
              size={tableDensity === "comfortable" ? "middle" : "small"}
              columns={[
                {
                  title: t("managePrompts.columns.title"),
                  dataIndex: "key",
                  key: "key",
                  render: (content) => (
                    <span className="line-clamp-1">
                      <Tag
                        color={
                          tagColors[content ?? "default"] ??
                          tagColors.default ??
                          "default"
                        }
                      >
                        {t(`common:copilot.${content}`)}
                      </Tag>
                    </span>
                  )
                },
                {
                  title: t("managePrompts.columns.prompt"),
                  dataIndex: "prompt",
                  key: "prompt",
                  render: (content) => <span className="line-clamp-1">{content}</span>
                },
                {
                  render: (_, record) => (
                    <div className="flex items-center gap-1">
                      <Tooltip title={t("managePrompts.tooltip.edit")}>
                        <button
                          type="button"
                          aria-label={t("managePrompts.tooltip.edit")}
                          onClick={() => onOpenCopilotEdit(record.key, record)}
                          data-testid={`copilot-action-edit-${record.key}`}
                          className="inline-flex min-h-8 min-w-8 items-center justify-center rounded p-2 text-text-muted hover:bg-bg-muted/70 focus:outline-none focus:ring-2 focus:ring-primary"
                        >
                          <Pen className="size-4" />
                        </button>
                      </Tooltip>
                      <Tooltip
                        title={t("managePrompts.copilot.copyToCustom.button", {
                          defaultValue: "Copy to Custom"
                        })}
                      >
                        <button
                          type="button"
                          aria-label={t("managePrompts.copilot.copyToCustom.button", {
                            defaultValue: "Copy to Custom"
                          })}
                          onClick={() => onCopyCopilotToCustom(record)}
                          data-testid={`copilot-action-copy-custom-${record.key}`}
                          className="inline-flex min-h-8 min-w-8 items-center justify-center rounded p-2 text-text-muted hover:bg-bg-muted/70 focus:outline-none focus:ring-2 focus:ring-primary"
                        >
                          <Copy className="size-4" />
                        </button>
                      </Tooltip>
                      <Tooltip
                        title={t("managePrompts.copilot.copyToClipboard.button", {
                          defaultValue: "Copy to clipboard"
                        })}
                      >
                        <button
                          type="button"
                          aria-label={t("managePrompts.copilot.copyToClipboard.button", {
                            defaultValue: "Copy to clipboard"
                          })}
                          onClick={() => {
                            void onCopyCopilotToClipboard(record)
                          }}
                          data-testid={`copilot-action-copy-clipboard-${record.key}`}
                          className="inline-flex min-h-8 min-w-8 items-center justify-center rounded p-2 text-text-muted hover:bg-bg-muted/70 focus:outline-none focus:ring-2 focus:ring-primary"
                        >
                          <Clipboard className="size-4" />
                        </button>
                      </Tooltip>
                    </div>
                  )
                }
              ]}
              dataSource={filteredCopilotData}
              rowKey={(record) => record.key}
              pagination={{ pageSize: 20, showSizeChanger: true }}
            />
          )}
        </>
      )}
    </div>
  )
}
