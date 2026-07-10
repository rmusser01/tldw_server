import React, { useEffect, useState } from "react"
import { Alert, Input, Switch } from "antd"
import { Loader2 } from "lucide-react"
import { Link } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useMessageOption } from "@/hooks/useMessageOption"

export const ChatbooksSettings = () => {
  const { t } = useTranslation(["settings", "common"])
  const notification = useAntdNotification()
  const { capabilities } = useServerCapabilities()
  const { serverChatId } = useMessageOption()
  const [chatbookName, setChatbookName] = useState("")
  const [chatbookDescription, setChatbookDescription] = useState("")
  const [chatbookConversationIds, setChatbookConversationIds] = useState("")
  const [chatbookIncludeMedia, setChatbookIncludeMedia] = useState(false)
  const [chatbookIncludeEmbeddings, setChatbookIncludeEmbeddings] =
    useState(false)
  const [chatbookIncludeGenerated, setChatbookIncludeGenerated] =
    useState(true)
  const [chatbookAsync, setChatbookAsync] = useState(true)
  const [chatbookExporting, setChatbookExporting] = useState(false)

  const parseIdList = (raw: string) =>
    raw
      .split(/[\n,]+/)
      .map((id) => id.trim())
      .filter(Boolean)

  const handleChatbookExport = async () => {
    if (!capabilities?.hasChatbooks) return
    const conversationIds = parseIdList(chatbookConversationIds)
    if (conversationIds.length === 0) {
      notification.error({
        message: t(
          "settings:chatbooks.exportMissingIds",
          "Add at least one conversation ID."
        )
      })
      return
    }
    setChatbookExporting(true)
    try {
      await tldwClient.initialize().catch(() => null)
      const res = await tldwClient.exportChatbook({
        name:
          chatbookName ||
          t("settings:chatbooks.defaultName", "Conversation export {{date}}", {
            date: new Date().toLocaleDateString()
          }),
        description:
          chatbookDescription ||
          t(
            "settings:chatbooks.exportDescriptionFallback",
            "Selective conversation export"
          ),
        content_selections: {
          conversation: conversationIds
        },
        include_media: chatbookIncludeMedia,
        include_embeddings: chatbookIncludeEmbeddings,
        include_generated_content: chatbookIncludeGenerated,
        async_mode: chatbookAsync
      })
      notification.success({
        message: res?.job_id
          ? t(
              "settings:chatbooks.exportQueued",
              "Selective conversation export job created"
            )
          : t(
              "settings:chatbooks.exportComplete",
              "Selective conversation export complete"
            )
      })
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error)
      notification.error({
        message: t(
          "settings:chatbooks.exportError",
          "Selective conversation export failed"
        ),
        description: msg
      })
    } finally {
      setChatbookExporting(false)
    }
  }

  useEffect(() => {
    if (!chatbookName) {
      setChatbookName(
        t("settings:chatbooks.defaultName", "Conversation export {{date}}", {
          date: new Date().toLocaleDateString()
        })
      )
    }
  }, [chatbookName, t])

  useEffect(() => {
    if (serverChatId && !chatbookConversationIds) {
      setChatbookConversationIds(serverChatId)
    }
  }, [chatbookConversationIds, serverChatId])

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-base font-semibold text-text">
          {t("settings:chatbooks.heading", "Chatbooks settings")}
        </h2>
        <p className="text-sm text-text-muted">
          {t(
            "settings:chatbooks.subheading",
            "Use Backup & Import for complete account backups and archive imports. Settings keeps only a selected-conversation export shortcut."
          )}
        </p>
      </div>

      <div className="rounded-md border border-border bg-surface p-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h3 className="text-sm font-semibold text-text">
              {t("settings:chatbooks.fullFlowTitle", "Full backup and archive import")}
            </h3>
            <p className="mt-1 text-sm text-text-muted">
              {t(
                "settings:chatbooks.fullFlowDescription",
                "Complete account backups and archive imports live in the dedicated Backup & Import workflow."
              )}
            </p>
          </div>
          <Link
            to="/chatbooks"
            className="inline-flex min-h-9 items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-primaryStrong focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2"
          >
            {t("settings:chatbooks.openBackupImport", "Open Backup & Import")}
          </Link>
        </div>
      </div>

      {capabilities && !capabilities.hasChatbooks && (
        <Alert
          type="info"
          showIcon
          message={t(
            "settings:chatbooks.unavailable",
            "Chatbooks are not available on this server."
          )}
        />
      )}

      <div className="rounded-md border border-border bg-surface p-4">
        <div>
          <h3 className="text-sm font-semibold text-text">
            {t("settings:chatbooks.exportTitle", "Selective conversation export")}
          </h3>
          <p className="mt-1 text-sm text-text-muted">
            {t(
              "settings:chatbooks.exportScopeNote",
              "Exports only the conversation IDs listed here. This is not a full account backup."
            )}
          </p>
        </div>

        <div className="mt-4 space-y-3">
          <label className="block space-y-1">
            <span className="text-xs font-medium text-text-muted">
              {t("settings:chatbooks.exportName", "Conversation export name")}
            </span>
            <Input
              value={chatbookName}
              onChange={(e) => setChatbookName(e.target.value)}
              placeholder={t(
                "settings:chatbooks.exportName",
                "Conversation export name"
              )}
              aria-label={t(
                "settings:chatbooks.exportName",
                "Conversation export name"
              )}
            />
          </label>
          <label className="block space-y-1">
            <span className="text-xs font-medium text-text-muted">
              {t("settings:chatbooks.exportDescription", "Short description")}
            </span>
            <Input.TextArea
              rows={2}
              value={chatbookDescription}
              onChange={(e) => setChatbookDescription(e.target.value)}
              placeholder={t(
                "settings:chatbooks.exportDescription",
                "Short description"
              )}
              aria-label={t(
                "settings:chatbooks.exportDescription",
                "Short description"
              )}
            />
          </label>
          <label className="block space-y-1">
            <span className="text-xs font-medium text-text-muted">
              {t(
                "settings:chatbooks.exportConversationIds",
                "Conversation IDs"
              )}
            </span>
            <Input.TextArea
              rows={2}
              value={chatbookConversationIds}
              onChange={(e) => setChatbookConversationIds(e.target.value)}
              placeholder={t(
                "settings:chatbooks.exportConversationIds",
                "Conversation IDs (comma-separated)"
              )}
              aria-label={t(
                "settings:chatbooks.exportConversationIds",
                "Conversation IDs"
              )}
            />
          </label>
          <div className="flex flex-wrap items-center gap-3 text-xs text-text-muted">
            <label className="flex items-center gap-2">
              <Switch
                checked={chatbookIncludeMedia}
                onChange={setChatbookIncludeMedia}
              />
              {t(
                "settings:chatbooks.includeMedia",
                "Include selected conversation media"
              )}
            </label>
            <label className="flex items-center gap-2">
              <Switch
                checked={chatbookIncludeEmbeddings}
                onChange={setChatbookIncludeEmbeddings}
              />
              {t(
                "settings:chatbooks.includeEmbeddings",
                "Include selected conversation embeddings"
              )}
            </label>
            <label className="flex items-center gap-2">
              <Switch
                checked={chatbookIncludeGenerated}
                onChange={setChatbookIncludeGenerated}
              />
              {t(
                "settings:chatbooks.includeGenerated",
                "Include selected generated conversation content"
              )}
            </label>
            <label className="flex items-center gap-2">
              <Switch checked={chatbookAsync} onChange={setChatbookAsync} />
              {t("settings:chatbooks.runAsync", "Run as background job")}
            </label>
          </div>
          <button
            onClick={handleChatbookExport}
            disabled={chatbookExporting || !capabilities?.hasChatbooks}
            className="inline-flex cursor-pointer items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-primaryStrong disabled:cursor-not-allowed disabled:opacity-60"
            type="button"
          >
            {chatbookExporting ? (
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
            ) : null}
            {t(
              "settings:chatbooks.exportButton",
              "Export selected conversations"
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
