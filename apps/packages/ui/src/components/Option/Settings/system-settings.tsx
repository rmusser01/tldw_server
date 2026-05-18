import React, { useState } from "react"
import { BetaTag } from "@/components/Common/Beta"
import { useFontSize } from "@/context/FontSizeProvider"
import { useMessageOption } from "@/hooks/useMessageOption"
import {
  exportPageAssistData,
  importPageAssistData
} from "@/libs/export-import"
import {
  CHAT_BACKGROUND_IMAGE_MAX_BASE64_LENGTH,
  CHAT_BACKGROUND_IMAGE_MAX_SIZE_MB,
  CHAT_BACKGROUND_IMAGE_SETTING,
  UI_MODE_SETTING
} from "@/services/settings/ui-settings"
import { useStorage } from "@plasmohq/storage/hook"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Input, Modal, Select, Switch } from "antd"
import { useTranslation } from "react-i18next"
import { Loader2, RotateCcw, Upload } from "lucide-react"
import { toBase64 } from "@/libs/to-base64"
import { PageAssistDatabase } from "@/db/dexie/chat"
import { isFireFox, isFireFoxPrivateMode } from "@/utils/is-private-mode"
import { firefoxSyncDataForPrivateMode } from "@/db/dexie/firefox-sync"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { Highlight, themes } from "prism-react-renderer"
import { useSetting } from "@/hooks/useSetting"
import {
  ACTION_ICON_CLICK_SETTING,
  CONTEXT_MENU_CLICK_SETTING
} from "@/services/action"

export const DataManagementSettings = () => {
  const { t } = useTranslation(["settings", "common"])
  const queryClient = useQueryClient()
  const { clearChat } = useMessageOption()
  const notification = useAntdNotification()
  const [resetModalOpen, setResetModalOpen] = useState(false)
  const [resetInput, setResetInput] = useState("")
  const [resetting, setResetting] = useState(false)
  const reloadTimeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)

  React.useEffect(() => {
    return () => {
      if (reloadTimeoutRef.current) {
        clearTimeout(reloadTimeoutRef.current)
      }
    }
  }, [])

  const handleResetAll = async () => {
    setResetting(true)
    try {
      const db = new PageAssistDatabase()
      await db.clearDB()
      queryClient.invalidateQueries({
        queryKey: ["fetchChatHistory"]
      })
      clearChat()
      try {
        await browser.storage.sync.clear()
        await browser.storage.local.clear()
        await browser.storage.session.clear()
      } catch (e) {
        console.error("Error clearing storage:", e)
      }
      setResetModalOpen(false)
      notification.success({
        message: t("settings:systemNotifications.resetSuccess", "All data has been reset")
      })
      setResetInput("")
      setTimeout(() => {
        window.location.reload()
      }, 1500)
    } catch (e) {
      console.error("Reset error:", e)
      notification.error({
        message: t("settings:systemNotifications.resetError", "Reset failed")
      })
    } finally {
      setResetting(false)
    }
  }

  const importDataMutation = useMutation({
    mutationFn: async (file: File) => {
      await importPageAssistData(file)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["fetchChatHistory"]
      })

      if (reloadTimeoutRef.current) {
        clearTimeout(reloadTimeoutRef.current)
      }

      const key = `import-reload-${Date.now()}`
      notification.success({
        key,
        message: (
          <span role="status" aria-live="polite">
            {t(
              "settings:systemNotifications.importSuccess",
              "Imported data successfully"
            )}
          </span>
        ),
        description: (
          <span role="status" aria-live="polite">
            {t(
              "settings:systemNotifications.importReloadNotice",
              "Page will reload in 5 seconds to apply changes..."
            )}
          </span>
        ),
        duration: 5,
        btn: (
          <button
            onClick={() => {
              if (reloadTimeoutRef.current) {
                clearTimeout(reloadTimeoutRef.current)
                reloadTimeoutRef.current = null
              }
              notification.destroy(key)
              notification.info({
                message: t(
                  "settings:systemNotifications.reloadCancelled",
                  "Reload cancelled. Some changes may require a manual refresh."
                ),
                duration: 4
              })
            }}
            className="text-primary hover:underline text-sm font-medium"
          >
            {t("common:cancel", "Cancel")}
          </button>
        )
      })

      reloadTimeoutRef.current = setTimeout(() => {
        window.location.reload()
      }, 5000)
    },
    onError: (error) => {
      console.error("Import error:", error)
      const errorMsg = error instanceof Error ? error.message : String(error)
      notification.error({
        message: t(
          "settings:systemNotifications.importError",
          "Import error"
        ),
        description: errorMsg || t(
          "settings:systemNotifications.importErrorDetail",
          "The import file may be corrupted or in an invalid format."
        )
      })
    }
  })

  const syncFirefoxData = useMutation({
    mutationFn: firefoxSyncDataForPrivateMode,
    onSuccess: () => {
      notification.success({
        message: t(
          "settings:systemNotifications.syncSuccess",
          "Firefox data synced successfully, You don't need to do this again"
        )
      })
    },
    onError: (error) => {
      console.log(error)
      notification.error({
        message: t(
          "settings:systemNotifications.syncError",
          "Firefox data sync failed"
        )
      })
    }
  })

  return (
    <div>
      <div className="mb-5">
        <h2 className="text-base font-semibold leading-7 text-text">
          {t("settings:dataManagement.title", "Data Management")}
        </h2>
        <p className="mt-1 text-xs text-text-muted">
          {t(
            "settings:dataManagement.description",
            "Export, import, sync, or reset local extension and WebUI data."
          )}
        </p>
        <div className="border-b border-border mt-3"></div>
      </div>

      {isFireFox && !isFireFoxPrivateMode && (
        <div className="flex flex-col sm:flex-row mb-3 gap-3 sm:gap-0 sm:justify-between sm:items-center">
          <span className="text-text">
            <BetaTag />
            {t("generalSettings.systemData.firefoxPrivateModeSync.label", {
              defaultValue:
                "Sync Custom Models, Prompts for Firefox Private Windows (Incognito Mode)"
            })}
          </span>
          <button
            onClick={() => {
              Modal.confirm({
                title: t("generalSettings.systemData.firefoxPrivateModeSync.confirmTitle", {
                  defaultValue: "Sync Firefox Data?"
                }),
                content: t("generalSettings.systemData.firefoxPrivateModeSync.confirmContent", {
                  defaultValue: "This will sync your custom models and prompts to Firefox private mode storage. Continue?"
                }),
                okText: t("common:continue", "Continue"),
                cancelText: t("common:cancel", "Cancel"),
                onOk: () => {
                  syncFirefoxData.mutate()
                }
              })
            }}
            disabled={syncFirefoxData.isPending}
            className="cursor-pointer rounded-md bg-primary px-4 py-2 text-white transition-colors hover:bg-primaryStrong w-full sm:w-auto">
            {syncFirefoxData.isPending ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              t("generalSettings.systemData.firefoxPrivateModeSync.button", {
                defaultValue: "Sync Data"
              })
            )}
          </button>
        </div>
      )}

      <div className="flex flex-col sm:flex-row mb-3 gap-3 sm:gap-0 sm:justify-between sm:items-center">
        <span className="text-text">
          {t("generalSettings.systemData.export.label", { defaultValue: t("generalSettings.system.export.label", { defaultValue: "Export Chat History, Knowledge Base, and Prompts" }) as string })}
        </span>
        <button
          onClick={exportPageAssistData}
          className="cursor-pointer rounded-md bg-primary px-4 py-2 text-white transition-colors hover:bg-primaryStrong w-full sm:w-auto">
          {t("generalSettings.systemData.export.button", { defaultValue: t("generalSettings.system.export.button", { defaultValue: "Export Data" }) as string })}
        </button>
      </div>

      <div className="flex flex-col sm:flex-row mb-3 gap-3 sm:gap-0 sm:justify-between sm:items-center">
        <span className="text-text">
          {t("generalSettings.systemData.import.label", { defaultValue: t("generalSettings.system.import.label", { defaultValue: "Import Chat History, Knowledge Base, and Prompts" }) as string })}
        </span>
        <label
          htmlFor="import"
          className="flex w-full cursor-pointer items-center justify-center rounded-md bg-primary px-4 py-2 text-white transition-colors hover:bg-primaryStrong sm:w-auto">
          {importDataMutation.isPending ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            </>
          ) : (
            t("generalSettings.systemData.import.button", { defaultValue: t("generalSettings.system.import.button", { defaultValue: "Import Data" }) as string })
          )}
        </label>
        <input
          type="file"
          accept=".json"
          id="import"
          className="hidden"
          disabled={importDataMutation.isPending}
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              importDataMutation.mutate(e.target.files[0])
            }
          }}
        />
      </div>

      <div className="flex flex-col sm:flex-row mb-3 gap-3 sm:gap-0 sm:justify-between sm:items-center">
        <span className="text-text">
          {t("generalSettings.systemData.deleteChatHistory.label", { defaultValue: t("generalSettings.system.deleteChatHistory.label", { defaultValue: "System Reset" }) as string })}
        </span>

        <button
          onClick={() => setResetModalOpen(true)}
          className="w-full rounded-md bg-danger px-4 py-2 text-white transition-colors hover:bg-danger sm:w-auto">
          {t("generalSettings.systemData.deleteChatHistory.button", { defaultValue: t("generalSettings.system.deleteChatHistory.button", { defaultValue: "Reset All" }) as string })}
        </button>
      </div>

      <Modal
        open={resetModalOpen}
        title={t("generalSettings.systemData.deleteChatHistory.modalTitle", { defaultValue: "System Reset" })}
        onCancel={() => {
          setResetModalOpen(false)
          setResetInput("")
        }}
        okText={t("common:reset", { defaultValue: "Reset" })}
        cancelText={t("common:cancel", { defaultValue: "Cancel" })}
        okButtonProps={{
          danger: true,
          disabled: resetInput.trim().toUpperCase() !== "RESET",
          loading: resetting
        }}
        onOk={handleResetAll}
        centered
      >
        <div className="space-y-4">
          <p className="text-text-muted">
            {t("generalSettings.systemData.deleteChatHistory.modalWarning", {
              defaultValue: "This will permanently delete ALL data including:"
            })}
          </p>
          <ul className="list-disc pl-5 text-text-muted text-sm space-y-1">
            <li>{t("generalSettings.systemData.deleteChatHistory.dataChat", { defaultValue: "Chat history and conversations" })}</li>
            <li>{t("generalSettings.systemData.deleteChatHistory.dataKnowledge", { defaultValue: "Knowledge base and documents" })}</li>
            <li>{t("generalSettings.systemData.deleteChatHistory.dataPrompts", { defaultValue: "Custom prompts and models" })}</li>
            <li>{t("generalSettings.systemData.deleteChatHistory.dataSettings", { defaultValue: "All settings and preferences" })}</li>
          </ul>
          <div className="pt-2">
            <p className="mb-2 text-sm font-medium text-text">
              {t("generalSettings.systemData.deleteChatHistory.typeToConfirm", {
                defaultValue: "Type RESET to confirm:"
              })}
            </p>
            <Input
              value={resetInput}
              onChange={(e) => setResetInput(e.target.value)}
              placeholder={t("generalSettings.systemData.deleteChatHistory.placeholder", "RESET")}
              className="font-mono"
              autoFocus
              aria-describedby="reset-hint"
            />
            <p id="reset-hint" className="text-xs text-text-subtle mt-1">
              {t("generalSettings.systemData.deleteChatHistory.caseInsensitiveHint", "Case-insensitive: 'reset', 'RESET', or 'Reset' all work")}
            </p>
          </div>
        </div>
      </Modal>
    </div>
  )
}

export const SystemSettings = () => {
  const { t } = useTranslation(["settings", "knowledge", "common"])
  const { increase, decrease, scale } = useFontSize()
  const notification = useAntdNotification()

  const quotaFriendlyMessage = t(
    "settings:storage.quotaFriendlyMessage",
    "Too many settings writes in a short period; please wait a few seconds and try again."
  )

  const showStorageError = (error: unknown) => {
    const msg =
      error instanceof Error ? error.message : typeof error === "string" ? error : ""
    const quotaHit =
      msg?.includes("MAX_WRITE_OPERATIONS_PER_MINUTE") ||
      msg?.includes("QUOTA_BYTES_PER_ITEM") ||
      msg?.includes("QUOTA_BYTES")

    notification.error({
      message: quotaHit
        ? t("settings:storage.quotaTitle", "Storage limit reached")
        : t("settings:storage.writeError", "Could not save settings"),
      description: quotaHit
        ? quotaFriendlyMessage
        : t(
            "settings:storage.writeErrorDescription",
            "We couldn't save your settings. Please try again shortly."
          )
    })
  }


  const [webuiBtnSidePanel, setWebuiBtnSidePanel] = useStorage(
    "webuiBtnSidePanel",
    false
  )

  // Default UI mode: fullscreen (webui) or sidebar (sidePanel)
  const [uiMode, setUiMode] = useSetting(UI_MODE_SETTING)

  const [actionIconClick, setActionIconClick] = useSetting(
    ACTION_ICON_CLICK_SETTING
  )

  const [contextMenuClick, setContextMenuClick] = useSetting(
    CONTEXT_MENU_CLICK_SETTING
  )
  const [chatBackgroundImage, setChatBackgroundImage] = useSetting(
    CHAT_BACKGROUND_IMAGE_SETTING
  )

  const [codeTheme, setCodeTheme] = useStorage("codeTheme", "auto")
  const sampleCode = `function hello(name: string) {
  console.log('Hello, ' + name)
}`

  const codeThemeOptions = [
    {
      label: t("generalSettings.systemBasics.codeTheme.auto", {
        defaultValue: "Follow app theme (auto)"
      }),
      value: "auto"
    },
    { label: "Dracula (dark)", value: "dracula" },
    { label: "GitHub (light)", value: "github" },
    { label: "Night Owl (dark)", value: "nightOwl" },
    { label: "Night Owl Light", value: "nightOwlLight" },
    { label: "VS Dark", value: "vsDark" }
  ]

  const resolvePreviewTheme = (key: string) => {
    switch (key) {
      case "github":
        return themes.github
      case "nightOwl":
        return themes.nightOwl
      case "nightOwlLight":
        return themes.nightOwlLight
      case "vsDark":
        return themes.vsDark
      case "dracula":
        return themes.dracula
      case "auto":
      default:
        // For preview, pick a neutral default; actual CodeBlock uses real auto logic.
        return themes.dracula
    }
  }

  const handleImageUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0]
    if (file) {
      try {
        if (!file.type.startsWith("image/")) {
          notification.error({
            message: t(
              "settings:systemNotifications.invalidImage",
              "Please select a valid image file"
            )
          })
          return
        }

        const base64String = await toBase64(file)

        if (base64String.length > CHAT_BACKGROUND_IMAGE_MAX_BASE64_LENGTH) {
          notification.error({
            message: t("settings:chatBackground.tooLargeTitle", "Image too large"),
            description: t(
              "settings:chatBackground.tooLargeDescription",
              `Please choose a smaller image (around ${CHAT_BACKGROUND_IMAGE_MAX_SIZE_MB} MB or less) for the chat background. Try compressing or resizing it and upload again.`
            )
          })
          return
        }

        try {
          await Promise.resolve(setChatBackgroundImage(base64String))
        } catch (err) {
          console.error("Storage error while saving background image:", err)
          showStorageError(err)
        }
      } catch (error) {
        console.error("Error uploading image:", error)
        showStorageError(error)
      }
    }
  }

  return (
    <div>
      <div className="mb-5">
        <h2 className="text-base font-semibold leading-7 text-text">
          {t("generalSettings.systemBasics.heading", { defaultValue: t("generalSettings.system.heading", { defaultValue: "System" }) as string })}
        </h2>
        <div className="border-b border-border mt-3"></div>
      </div>

      <div className="flex flex-col sm:flex-row mb-3 gap-3 sm:gap-0 sm:justify-between sm:items-center">
        <span className="text-text">
          {t("generalSettings.systemBasics.uiMode.label", { defaultValue: "Default UI Mode" })}
        </span>
        <Select
          options={[
            { label: t("generalSettings.systemBasics.uiMode.options.sidePanel", { defaultValue: "Sidebar" }), value: "sidePanel" },
            { label: t("generalSettings.systemBasics.uiMode.options.webui", { defaultValue: "Full Screen (Web UI)" }), value: "webui" }
          ]}
          value={uiMode}
          className="w-full sm:w-[220px]"
          onChange={async (value) => {
            await setUiMode(value)
            // Keep action/context menu behavior consistent with default mode
            await setActionIconClick(value)
            await setContextMenuClick("sidePanel")
          }}
        />
      </div>
      <div className="flex flex-col sm:flex-row mb-3 gap-3 sm:gap-0 sm:justify-between sm:items-center">
        <span className="text-text font-medium">
          <BetaTag />
          {t("generalSettings.systemBasics.fontSize.label", { defaultValue: t("generalSettings.system.fontSize.label", { defaultValue: "Font Size" }) as string })}
        </span>
        <div className="flex flex-row items-center gap-3 justify-center sm:justify-end">
          <button
            onClick={decrease}
            className="rounded-lg bg-primary px-3 py-1.5 text-sm font-medium text-white transition-colors duration-200 hover:bg-primaryStrong"
          >
            A-
          </button>
          <span className="min-w-[2rem] text-center font-medium text-text">
            {scale.toFixed(1)}x
          </span>
          <button
            onClick={increase}
            className="rounded-lg bg-primary px-3 py-1.5 text-sm font-medium text-white transition-colors duration-200 hover:bg-primaryStrong"
          >
            A+
          </button>{" "}
        </div>
      </div>

      <div className="flex flex-col sm:flex-row mb-3 gap-3 sm:gap-0 sm:justify-between sm:items-center">
        <span className="text-text">
          {t("generalSettings.systemBasics.codeTheme.label", {
            defaultValue: "Code block theme"
          })}
        </span>
        <div className="w-full sm:w-[320px] flex flex-col gap-2">
          <Select
            className="w-full"
            value={codeTheme}
            onChange={setCodeTheme}
            options={codeThemeOptions}
          />
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {codeThemeOptions.map((opt) => (
              <button
                key={opt.value}
                type="button"
                onClick={() => setCodeTheme(opt.value)}
                className={`rounded-md border px-2 py-2 text-left transition-colors ${
                  codeTheme === opt.value
                    ? "border-border-strong bg-surface2"
                    : "border-border bg-surface hover:border-border-strong"
                }`}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-text">
                    {opt.label}
                  </span>
                  {codeTheme === opt.value && (
                    <span className="rounded-full bg-surface2 px-2 py-0.5 text-[10px] text-text-subtle">
                      {t("generalSettings.systemBasics.codeTheme.current", {
                        defaultValue: "Current"
                      })}
                    </span>
                  )}
                </div>
                <Highlight
                  code={sampleCode}
                  language={"tsx" as any}
                  theme={resolvePreviewTheme(opt.value)}>
                  {({ className, style, tokens, getLineProps, getTokenProps }) => (
                    <pre
                      className={`${className} m-0 max-h-24 overflow-hidden rounded-sm text-[10px] leading-snug`}
                      style={{
                        ...style,
                        fontFamily: "var(--font-mono)"
                      }}>
                      {tokens.slice(0, 4).map((line, i) => (
                        <div
                          key={i}
                          {...getLineProps({ line })}>
                          {line.map((token, key) => (
                            <span
                              key={key}
                              {...getTokenProps({ token })}
                            />
                          ))}
                        </div>
                      ))}
                    </pre>
                  )}
                </Highlight>
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="flex flex-col sm:flex-row mb-3 gap-3 sm:gap-0 sm:justify-between sm:items-center">
        <span className="text-text">
          <BetaTag />
          {t("generalSettings.systemBasics.actionIcon.label", { defaultValue: t("generalSettings.system.actionIcon.label", { defaultValue: "Browser Action Button" }) as string })}
        </span>
        <Select
          options={[
            {
              label: "Open Web UI",
              value: "webui"
            },
            {
              label: "Open Sidebar",
              value: "sidePanel"
            }
          ]}
          value={actionIconClick}
          className="w-full sm:w-[200px]"
          onChange={(value) => {
            void setActionIconClick(value)
          }}
        />
      </div>
      <div className="flex flex-col sm:flex-row mb-3 gap-3 sm:gap-0 sm:justify-between sm:items-center">
        <span className="text-text">
          <BetaTag />
          {t("generalSettings.systemBasics.contextMenu.label", { defaultValue: t("generalSettings.system.contextMenu.label", { defaultValue: "Context Menu Action" }) as string })}
        </span>
        <Select
          options={[
            {
              label: "Open Web UI",
              value: "webui"
            },
            {
              label: "Open Sidebar",
              value: "sidePanel"
            }
          ]}
          value={contextMenuClick}
          className="w-full sm:w-[200px]"
          onChange={(value) => {
            void setContextMenuClick(value)
          }}
        />
      </div>
      <div className="flex flex-col sm:flex-row mb-3 gap-3 sm:gap-0 sm:justify-between sm:items-center">
        <span className="text-text">
          {t("generalSettings.systemBasics.webuiBtnSidePanel.label", { defaultValue: t("generalSettings.system.webuiBtnSidePanel.label", { defaultValue: "Show Web UI button in Sidebar" }) as string })}
        </span>
         <div>
          <Switch
          checked={webuiBtnSidePanel}
          onChange={(checked) => {
            setWebuiBtnSidePanel(checked)
          }}
        />
         </div>
      </div>

      <div className="flex flex-col sm:flex-row mb-3 gap-3 sm:gap-0 sm:justify-between sm:items-center">
        <span className="text-text">
          <BetaTag />
          {t("generalSettings.systemBasics.chatBackgroundImage.label", { defaultValue: t("generalSettings.system.chatBackgroundImage.label", { defaultValue: "Chat Background Image" }) as string })}
        </span>
        <div className="flex items-center gap-2 justify-center sm:justify-end">
          {chatBackgroundImage ? (
            <button
              onClick={() => {
                void setChatBackgroundImage(undefined)
              }}
              className="text-text">
              <RotateCcw className="size-4" />
            </button>
          ) : null}
          <label
            htmlFor="background-image-upload"
            className="inline-flex cursor-pointer gap-2 rounded-md bg-primary px-4 py-2 text-white transition-colors hover:bg-primaryStrong">
            <Upload className="size-4" />
            {t("knowledge:form.uploadFile.label")}
          </label>
          <input
            type="file"
            accept="image/*"
            id="background-image-upload"
            className="hidden"
            onChange={handleImageUpload}
          />
        </div>
      </div>
    </div>
  )
}
