import React, { useEffect, useLayoutEffect, useRef } from "react"
import { Alert, Button, Form, Input, Modal, Select, message } from "antd"
import { useTranslation } from "react-i18next"
import { testWatchlistSource, testWatchlistSourceDraft } from "@/services/watchlists"
import type { JobPreviewResult, SourcePreviewDiagnostics } from "@/types/watchlists"
import type { WatchlistSource, SourceType } from "@/types/watchlists"
import { buildWatchlistsModalChrome, useWatchlistsViewport } from "../shared"
import { mapWatchlistsError } from "../shared/watchlists-error"
import {
  buildSourceSettingsPayload,
  SOURCE_SETTINGS_FORM_FIELDS,
  sourceSettingsAreEqual,
  sourceSettingsToFormValues
} from "./source-settings"
import {
  getFocusableActiveElement,
  restoreFocusToElement
} from "../shared/focus-management"

interface SourceFormModalProps {
  open: boolean
  onClose: () => void
  onSubmit: (values: {
    name: string
    url: string
    source_type: SourceType
    tags: string[]
    settings?: Record<string, unknown> | null
  }) => Promise<void>
  initialValues?: WatchlistSource
  existingTags: string[]
  forumsEnabled?: boolean
}

const toText = (value: unknown, fallback = ""): string =>
  typeof value === "string" && value.trim().length > 0 ? value : fallback

const resolveTestSourceErrorHint = (
  rawMessage: string,
  t: (...args: any[]) => unknown
): string => {
  const normalized = rawMessage.toLowerCase()

  if (normalized.includes("forum_sources_disabled")) {
    return toText(
      t(
        "watchlists:sources.form.testSourceErrorHintForumDisabled",
        "Forum feeds are not enabled yet. Switch type to RSS Feed or Website."
      ),
      "Forum feeds are not enabled yet. Switch type to RSS Feed or Website."
    )
  }
  if (normalized.includes("invalid_youtube_rss_url")) {
    return toText(
      t(
        "watchlists:sources.form.testSourceErrorHintYoutube",
        "Use a canonical YouTube feed URL (channel_id or playlist_id) and retry."
      ),
      "Use a canonical YouTube feed URL (channel_id or playlist_id) and retry."
    )
  }
  if (normalized.includes("source_not_found")) {
    return toText(
      t(
        "watchlists:sources.form.testSourceErrorHintNotFound",
        "This saved feed no longer exists. Refresh feeds and open it again."
      ),
      "This saved feed no longer exists. Refresh feeds and open it again."
    )
  }
  if (
    normalized.includes("failed to fetch") ||
    normalized.includes("network") ||
    normalized.includes("timeout")
  ) {
    return toText(
      t(
        "watchlists:sources.form.testSourceErrorHintNetwork",
        "Check server connectivity, then run Test Feed again."
      ),
      "Check server connectivity, then run Test Feed again."
    )
  }
  return toText(
    t(
      "watchlists:sources.form.testSourceErrorHintGeneric",
      "Review URL and feed type, then retry. If this persists, check server logs."
    ),
    "Review URL and feed type, then retry. If this persists, check server logs."
  )
}

const toDiagnosticList = (value: unknown): string[] => {
  if (!Array.isArray(value)) return []
  return value.filter((item): item is string => typeof item === "string" && item.length > 0)
}

const buildDiagnosticsLines = (
  diagnostics: SourcePreviewDiagnostics | null | undefined
): string[] => {
  if (!diagnostics) return []
  const lines: string[] = []
  if (diagnostics.fetch_mode) {
    lines.push(`Fetch mode: ${diagnostics.fetch_mode}`)
  }
  lines.push(...toDiagnosticList(diagnostics.selector_errors))
  lines.push(...toDiagnosticList(diagnostics.selector_warnings))
  lines.push(...toDiagnosticList(diagnostics.no_match_warnings))
  lines.push(...toDiagnosticList(diagnostics.non_unique_warnings))
  lines.push(...toDiagnosticList(diagnostics.fragile_selector_warnings))
  if (diagnostics.dedupe_preview_key) {
    lines.push(`Dedupe preview key: ${diagnostics.dedupe_preview_key}`)
  }
  return lines
}

export const SourceFormModal: React.FC<SourceFormModalProps> = ({
  open,
  onClose,
  onSubmit,
  initialValues,
  existingTags,
  forumsEnabled = false
}) => {
  const { t } = useTranslation(["watchlists", "common"])
  const [form] = Form.useForm()
  const [submitting, setSubmitting] = React.useState(false)
  const [testingSource, setTestingSource] = React.useState(false)
  const [testResult, setTestResult] = React.useState<JobPreviewResult | null>(null)
  const [testError, setTestError] = React.useState<string | null>(null)
  const [testErrorHint, setTestErrorHint] = React.useState<string | null>(null)
  const restoreFocusTargetRef = useRef<HTMLElement | null>(null)
  const wasOpenRef = useRef(false)
  const { isConstrained } = useWatchlistsViewport()

  const isEditing = !!initialValues
  const testSourceId = typeof initialValues?.id === "number" ? initialValues.id : null
  const modalChrome = buildWatchlistsModalChrome(isConstrained, 500)
  const diagnosticsLines = buildDiagnosticsLines(testResult?.diagnostics)

  useLayoutEffect(() => {
    if (open) {
      if (!wasOpenRef.current) {
        restoreFocusTargetRef.current = getFocusableActiveElement()
      }
      wasOpenRef.current = true
      return
    }

    if (wasOpenRef.current) {
      wasOpenRef.current = false
      restoreFocusToElement(restoreFocusTargetRef.current)
    }
  }, [open])

  // Reset form when modal opens/closes or initialValues change
  useEffect(() => {
    if (open) {
      if (initialValues) {
        form.setFieldsValue({
          name: initialValues.name,
          url: initialValues.url,
          source_type: initialValues.source_type,
          tags: initialValues.tags,
          ...sourceSettingsToFormValues(initialValues.settings)
        })
      } else {
        form.resetFields()
        form.setFieldsValue({
          source_type: "rss",
          tags: [],
          ...sourceSettingsToFormValues(null)
        })
      }
      setTestResult(null)
      setTestError(null)
      setTestErrorHint(null)
    }
  }, [open, initialValues, form])

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields()
      const settingsPayload = buildSourceSettingsPayload(initialValues?.settings, values)
      const hasInitialSettings =
        Boolean(initialValues?.settings) &&
        Object.keys(initialValues?.settings || {}).length > 0
      const payload = {
        name: String(values.name || ""),
        url: String(values.url || ""),
        source_type: values.source_type as SourceType,
        tags: Array.isArray(values.tags) ? values.tags : [],
        ...(settingsPayload || hasInitialSettings ? { settings: settingsPayload || null } : {})
      }
      setSubmitting(true)
      await onSubmit(payload)
      form.resetFields()
    } catch (err) {
      // Validation error or submit error - handled by parent
      console.error("Form submit error:", err)
    } finally {
      setSubmitting(false)
    }
  }

  const handleCancel = () => {
    form.resetFields()
    setTestResult(null)
    setTestError(null)
    setTestErrorHint(null)
    onClose()
  }

  const handleTestSource = async () => {
    try {
      const values = await form.validateFields([
        "url",
        "source_type",
        ...SOURCE_SETTINGS_FORM_FIELDS
      ])
      const draftUrl = String(values?.url ?? "")
      const draftType = String(values?.source_type ?? "")
      const settingsPayload = buildSourceSettingsPayload(initialValues?.settings, values)
      const isSavedSourceUnchanged =
        !!testSourceId &&
        !!initialValues &&
        draftUrl === String(initialValues.url) &&
        draftType === String(initialValues.source_type) &&
        sourceSettingsAreEqual(initialValues.settings, settingsPayload)

      setTestingSource(true)
      setTestError(null)
      setTestErrorHint(null)

      const preview = isSavedSourceUnchanged
        ? await testWatchlistSource(testSourceId, { limit: 10 })
        : await testWatchlistSourceDraft(
            {
              url: draftUrl,
              source_type: draftType as SourceType,
              settings: settingsPayload || null
            },
            { limit: 10 }
          )

      setTestResult(preview)
      const previewCount = Number(preview?.total || 0)
      if (previewCount > 0) {
        message.success(
          t(
            "watchlists:sources.form.testSourceSuccess",
            "Test succeeded: found {{count}} preview item{{plural}}.",
            { count: previewCount, plural: previewCount === 1 ? "" : "s" }
          )
        )
      } else {
        message.warning(
          t(
            "watchlists:sources.form.testSourceNoItems",
            "Test completed, but no preview items were returned."
          )
        )
      }
    } catch (err) {
      if (err && typeof err === "object" && "errorFields" in err) {
        return
      }
      console.error("Source test failed:", err)
      const fallback = t("watchlists:sources.form.testSourceError", "Source test failed")
      const mapped = mapWatchlistsError(err, {
        t,
        context: t("watchlists:sources.form.testSourceContext", "feed preflight"),
        fallbackMessage: fallback,
        operationLabel: t("watchlists:errors.operation.test", "test")
      })
      setTestResult(null)
      setTestError(mapped.title)
      const contextualHint = resolveTestSourceErrorHint(mapped.rawMessage, t)
      setTestErrorHint(`${mapped.description} ${contextualHint}`.trim())
      if (mapped.severity === "warning") {
        message.warning(mapped.title)
      } else {
        message.error(mapped.title)
      }
    } finally {
      setTestingSource(false)
    }
  }

  return (
    <Modal
      title={
        isEditing
          ? t("watchlists:sources.editSource", "Edit Source")
          : t("watchlists:sources.addSource", "Add Source")
      }
      open={open}
      onOk={handleSubmit}
      onCancel={handleCancel}
      okText={
        isEditing
          ? t("common:save", "Save")
          : t("common:create", "Create")
      }
      cancelText={t("common:cancel", "Cancel")}
      confirmLoading={submitting}
      destroyOnHidden
      data-testid="source-form-modal"
      width={modalChrome.width}
      style={modalChrome.style}
      styles={modalChrome.styles}
    >
      <Form
        form={form}
        layout="vertical"
        className="mt-4"
        initialValues={{
          source_type: "rss",
          tags: []
        }}
      >
        <Form.Item
          name="name"
          label={t("watchlists:sources.form.name", "Name")}
          rules={[
            {
              required: true,
              message: t("watchlists:sources.form.nameRequired", "Please enter a name")
            },
            {
              max: 200,
              message: t(
                "watchlists:sources.form.nameTooLong",
                "Name must be less than 200 characters"
              )
            }
          ]}
        >
          <Input
            placeholder={t(
              "watchlists:sources.form.namePlaceholder",
              "e.g., Tech News Daily"
            )}
          />
        </Form.Item>

        <Form.Item
          name="url"
          label={t("watchlists:sources.form.url", "URL")}
          rules={[
            {
              required: true,
              message: t("watchlists:sources.form.urlRequired", "Please enter a URL")
            },
            {
              type: "url",
              message: t("watchlists:sources.form.urlInvalid", "Please enter a valid URL")
            }
          ]}
        >
          <Input
            placeholder={t(
              "watchlists:sources.form.urlPlaceholder",
              "e.g., https://example.com/feed.xml"
            )}
          />
        </Form.Item>

        <div className="mb-4 space-y-2">
          <div className="flex items-center gap-2">
            <Button
              size="small"
              onClick={() => void handleTestSource()}
              loading={testingSource}
            >
              {t("watchlists:sources.form.testSource", "Test Feed")}
            </Button>
            <span className="text-xs text-text-muted">
              {isEditing
                ? t(
                    "watchlists:sources.form.testSourceHint",
                    "Runs a quick fetch preview for this feed. Unsaved URL/type edits are tested."
                  )
                : t(
                    "watchlists:sources.form.testSourceDraftHint",
                    "Run Test Feed to validate URL/type connectivity before saving."
                  )}
            </span>
          </div>
          {testResult && (
            <Alert
              type={Number(testResult.total || 0) > 0 ? "success" : "warning"}
              showIcon
              title={t("watchlists:sources.form.testSourceSummary", "Test Summary")}
              description={t(
                "watchlists:sources.form.testSourceSummaryDescription",
                "{{total}} preview item{{plural}}, {{ingestable}} ingestable, {{filtered}} filtered.",
                {
                  total: Number(testResult.total || 0),
                  ingestable: Number(testResult.ingestable || 0),
                  filtered: Number(testResult.filtered || 0),
                  plural: Number(testResult.total || 0) === 1 ? "" : "s"
                }
              )}
            />
          )}
          {diagnosticsLines.length > 0 && (
            <Alert
              type="info"
              showIcon
              message={t(
                "watchlists:sources.form.validationDiagnostics",
                "Validation diagnostics"
              )}
              description={(
                <div className="space-y-1">
                  {diagnosticsLines.map((line) => (
                    <div key={line}>{line}</div>
                  ))}
                </div>
              )}
            />
          )}
          {testError && (
            <Alert
              type="error"
              showIcon
              title={testError}
              description={testErrorHint || testError}
              action={(
                <Button
                  size="small"
                  onClick={() => void handleTestSource()}
                  loading={testingSource}
                >
                  {t("watchlists:errors.retry", "Retry")}
                </Button>
              )}
            />
          )}
        </div>

        <Form.Item
          name="source_type"
          label={t("watchlists:sources.form.type", "Type")}
          extra={
            forumsEnabled
              ? undefined
              : t(
                  "watchlists:sources.form.forumDisabledHelp",
                  "Forum monitoring is coming soon. Use RSS Feed or Website for now."
                )
          }
          rules={[
            {
              required: true,
              message: t("watchlists:sources.form.typeRequired", "Please select a type")
            }
          ]}
        >
          <Select
            options={[
              {
                label: t("watchlists:sources.types.rss", "RSS Feed"),
                value: "rss"
              },
              {
                label: t("watchlists:sources.types.site", "Website"),
                value: "site"
              },
              {
                label: forumsEnabled
                  ? t("watchlists:sources.types.forum", "Forum")
                  : t("watchlists:sources.types.forumComingSoon", "Forum (coming soon)"),
                value: "forum",
                disabled: !forumsEnabled
              }
            ]}
          />
        </Form.Item>

        <details className="mb-4 rounded-md border border-border p-3">
          <summary className="cursor-pointer text-sm font-medium">
            {t("watchlists:sources.form.advancedRules", "Advanced fetch and extraction rules")}
          </summary>
          <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-2">
            <Form.Item
              name="source_top_n"
              label={t("watchlists:sources.form.topN", "Top links limit")}
              extra={t(
                "watchlists:sources.form.topNHelp",
                "For websites without scrape rules, choose how many discovered links to inspect."
              )}
            >
              <Input
                placeholder={t("watchlists:sources.form.topNPlaceholder", "e.g., 10")}
              />
            </Form.Item>
            <Form.Item
              name="discover_method"
              label={t("watchlists:sources.form.discoverMethod", "Discovery method")}
            >
              <Select
                options={[
                  {
                    label: t("watchlists:sources.form.discoverAuto", "Auto"),
                    value: "auto"
                  },
                  {
                    label: t("watchlists:sources.form.discoverFrontpage", "Front page links"),
                    value: "frontpage"
                  },
                  {
                    label: t("watchlists:sources.form.discoverSearch", "Search provider"),
                    value: "search"
                  }
                ]}
              />
            </Form.Item>
          </div>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <Form.Item
              name="scrape_list_url"
              label={t("watchlists:sources.form.scrapeListUrl", "List page URL")}
            >
              <Input
                placeholder={t(
                  "watchlists:sources.form.scrapeListUrlPlaceholder",
                  "Defaults to source URL"
                )}
              />
            </Form.Item>
            <Form.Item
              name="scrape_limit"
              label={t("watchlists:sources.form.scrapeLimit", "Scrape item limit")}
            >
              <Input
                placeholder={t("watchlists:sources.form.scrapeLimitPlaceholder", "e.g., 20")}
              />
            </Form.Item>
            <Form.Item
              name="scrape_item_selector"
              label={t("watchlists:sources.form.itemSelector", "Item selector")}
            >
              <Input
                placeholder={t(
                  "watchlists:sources.form.itemSelectorPlaceholder",
                  "css:article"
                )}
              />
            </Form.Item>
            <Form.Item
              name="scrape_link_selector"
              label={t("watchlists:sources.form.linkSelector", "Link XPath")}
            >
              <Input
                placeholder={t(
                  "watchlists:sources.form.linkSelectorPlaceholder",
                  ".//a/@href"
                )}
              />
            </Form.Item>
            <Form.Item
              name="scrape_title_selector"
              label={t("watchlists:sources.form.titleSelector", "Title selector")}
            >
              <Input
                placeholder={t("watchlists:sources.form.titleSelectorPlaceholder", "css:h2")}
              />
            </Form.Item>
            <Form.Item
              name="scrape_summary_selector"
              label={t("watchlists:sources.form.summarySelector", "Summary selector")}
            >
              <Input
                placeholder={t(
                  "watchlists:sources.form.summarySelectorPlaceholder",
                  "css:.summary"
                )}
              />
            </Form.Item>
            <Form.Item
              name="scrape_content_selector"
              label={t("watchlists:sources.form.contentSelector", "Content selector")}
            >
              <Input
                placeholder={t(
                  "watchlists:sources.form.contentSelectorPlaceholder",
                  "css:.article-body"
                )}
              />
            </Form.Item>
            <Form.Item
              name="scrape_date_selector"
              label={t("watchlists:sources.form.dateSelector", "Published date selector")}
            >
              <Input
                placeholder={t("watchlists:sources.form.dateSelectorPlaceholder", "css:time")}
              />
            </Form.Item>
            <Form.Item
              name="scrape_guid_selector"
              label={t("watchlists:sources.form.guidSelector", "Dedupe identity XPath")}
              extra={t(
                "watchlists:sources.form.guidSelectorHelp",
                "Optional stable per-item identity when URLs alone are not reliable."
              )}
            >
              <Input
                placeholder={t(
                  "watchlists:sources.form.guidSelectorPlaceholder",
                  ".//@data-id"
                )}
              />
            </Form.Item>
          </div>
        </details>

        <Form.Item
          name="tags"
          label={t("watchlists:sources.form.tags", "Tags")}
          extra={t(
            "watchlists:sources.form.tagsHelp",
            "Add tags to organize and filter your sources"
          )}
        >
          <Select
            mode="tags"
            placeholder={t(
              "watchlists:sources.form.tagsPlaceholder",
              "Add or select tags"
            )}
            options={existingTags.map((tag) => ({
              label: tag,
              value: tag
            }))}
            tokenSeparators={[","]}
          />
        </Form.Item>
      </Form>
    </Modal>
  )
}
