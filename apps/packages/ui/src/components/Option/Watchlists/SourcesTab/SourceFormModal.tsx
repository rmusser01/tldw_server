import React, { useEffect, useLayoutEffect, useRef } from "react"
import { Alert, Button, Form, Input, Modal, Select, message } from "antd"
import { useTranslation } from "react-i18next"
import { testWatchlistSource, testWatchlistSourceDraft } from "@/services/watchlists"
import type { JobPreviewResult } from "@/types/watchlists"
import type { WatchlistSource, SourceType } from "@/types/watchlists"
import { buildWatchlistsModalChrome, useWatchlistsViewport } from "../shared"
import { mapWatchlistsError } from "../shared/watchlists-error"
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
  }) => Promise<void>
  initialValues?: WatchlistSource
  existingTags: string[]
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

export const SourceFormModal: React.FC<SourceFormModalProps> = ({
  open,
  onClose,
  onSubmit,
  initialValues,
  existingTags
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
          tags: initialValues.tags
        })
      } else {
        form.resetFields()
        form.setFieldsValue({
          source_type: "rss",
          tags: []
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
      setSubmitting(true)
      await onSubmit(values)
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
      const values = await form.validateFields(["url", "source_type"])
      const draftUrl = String(values?.url ?? "")
      const draftType = String(values?.source_type ?? "")
      const isSavedSourceUnchanged =
        !!testSourceId &&
        !!initialValues &&
        draftUrl === String(initialValues.url) &&
        draftType === String(initialValues.source_type)

      setTestingSource(true)
      setTestError(null)
      setTestErrorHint(null)

      const preview = isSavedSourceUnchanged
        ? await testWatchlistSource(testSourceId, { limit: 10 })
        : await testWatchlistSourceDraft(
            {
              url: draftUrl,
              source_type: draftType as SourceType
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
              message={t("watchlists:sources.form.testSourceSummary", "Test Summary")}
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
          {testError && (
            <Alert
              type="error"
              showIcon
              message={testError}
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
          extra={t(
            "watchlists:sources.form.forumDisabledHelp",
            "Forum monitoring is coming soon. Use RSS Feed or Website for now."
          )}
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
                label: t("watchlists:sources.types.forumComingSoon", "Forum (coming soon)"),
                value: "forum",
                disabled: true // Forum support coming later
              }
            ]}
          />
        </Form.Item>

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
