import React from "react"
import {
  Button,
  Checkbox,
  Dropdown,
  Popover,
  Space,
  Tooltip
} from "antd"
import {
  ChevronDown,
  PaperclipIcon,
  StopCircleIcon,
  Settings2
} from "lucide-react"
import { Button as TldwButton } from "@/components/Common/Button"
import type { KnowledgeTab } from "@/components/Knowledge"

// ---------------------------------------------------------------------------
// Attachment button props
// ---------------------------------------------------------------------------

export interface PlaygroundAttachmentButtonProps {
  isProMode: boolean
  isMobileViewport: boolean
  chatMode: string
  onImageUpload: () => void
  onDocumentUpload: () => void
  onOpenKnowledgePanel: (tab: KnowledgeTab) => void
  attachmentMenuOpen: boolean
  onAttachmentMenuChange: (open: boolean) => void
  t: (key: string, defaultValue?: string, options?: any) => any
}

export const PlaygroundAttachmentButton: React.FC<PlaygroundAttachmentButtonProps> =
  React.memo(function PlaygroundAttachmentButton(props) {
    const {
      isProMode,
      isMobileViewport,
      chatMode,
      onImageUpload,
      onDocumentUpload,
      onOpenKnowledgePanel,
      attachmentMenuOpen,
      onAttachmentMenuChange,
      t
    } = props

    const imageAttachmentDisabled =
      chatMode === "rag"
        ? t(
            "playground:attachments.imageDisabledBody",
            "Disable Knowledge Search to attach images."
          )
        : null

    const menuContent = (
      <div className="flex w-56 flex-col gap-1 p-1">
        <button
          type="button"
          onClick={onDocumentUpload}
          title={t("tooltip.uploadDocuments") as string}
          className="flex w-full items-center justify-between rounded-md px-2 py-1.5 text-sm text-text transition hover:bg-surface2"
        >
          <span className="flex flex-col items-start">
            <span>
              {t("playground:actions.attachDocument", "Attach document")}
            </span>
            <span className="text-[10px] text-text-muted">
              {t(
                "playground:actions.attachDocumentHint",
                "PDF/DOCX/TXT/CSV/MD"
              )}
            </span>
          </span>
          <PaperclipIcon className="h-4 w-4" />
        </button>
        <div className="border-t border-border my-1" />
        <button
          type="button"
          onClick={() => onOpenKnowledgePanel("context")}
          className="flex w-full items-center justify-between rounded-md px-2 py-1.5 text-sm text-text transition hover:bg-surface2"
        >
          <span>
            {t(
              "playground:attachments.manageContext",
              "Manage in Knowledge Panel"
            )}
          </span>
          <Settings2 className="h-4 w-4" />
        </button>
      </div>
    )

    return (
      <div className="inline-flex items-center">
        <Tooltip title={imageAttachmentDisabled || undefined}>
          <span>
            <TldwButton
              variant="outline"
              size={isMobileViewport ? "lg" : "sm"}
              shape={isProMode ? "rounded" : "pill"}
              iconOnly={!isProMode}
              ariaLabel={
                t(
                  "playground:actions.attachImage",
                  "Attach image"
                ) as string
              }
              title={
                t(
                  "playground:actions.attachImage",
                  "Attach image"
                ) as string
              }
              disabled={chatMode === "rag"}
              data-testid="attachment-button"
              onClick={onImageUpload}
              className="rounded-r-none"
            >
              {isProMode ? (
                <span className="inline-flex items-center gap-1.5">
                  <PaperclipIcon className="h-4 w-4" aria-hidden="true" />
                  <span>{t("playground:actions.attach", "Attach")}</span>
                </span>
              ) : (
                <>
                  <PaperclipIcon className="h-4 w-4" aria-hidden="true" />
                  <span className="sr-only">
                    {t("playground:actions.attachImage", "Attach image")}
                  </span>
                </>
              )}
            </TldwButton>
          </span>
        </Tooltip>
        <Popover
          trigger="click"
          placement="topRight"
          content={menuContent}
          overlayClassName="playground-attachment-menu"
          open={attachmentMenuOpen}
          onOpenChange={onAttachmentMenuChange}
        >
          <TldwButton
            variant="outline"
            size={isMobileViewport ? "lg" : "sm"}
            shape={isProMode ? "rounded" : "pill"}
            iconOnly
            ariaLabel={
              t(
                "playground:actions.attachMore",
                "More attachments"
              ) as string
            }
            title={
              t(
                "playground:actions.attachMore",
                "More attachments"
              ) as string
            }
            className="-ml-px rounded-l-none"
          >
            <ChevronDown className="h-4 w-4" aria-hidden="true" />
          </TldwButton>
        </Popover>
      </div>
    )
  })

// ---------------------------------------------------------------------------
// Send control props
// ---------------------------------------------------------------------------

export interface PlaygroundSendControlProps {
  isProMode: boolean
  isMobileViewport: boolean
  isSending: boolean
  isConnectionReady: boolean
  sendWhenEnter: boolean
  onSendWhenEnterChange: (checked: boolean) => void
  sendLabel: string
  compareNeedsMoreModels: boolean
  onStopStreaming: () => void
  onStopListening: () => void
  onSubmitForm: () => void

  // Send menu
  sendMenuOpen: boolean
  onSendMenuChange: (open: boolean) => void

  t: (key: string, defaultValue?: string, options?: any) => any
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const PlaygroundSendControl: React.FC<PlaygroundSendControlProps> =
  React.memo(function PlaygroundSendControl(props) {
    const {
      isProMode,
      isMobileViewport,
      isSending,
      isConnectionReady,
      sendWhenEnter,
      onSendWhenEnterChange,
      sendLabel,
      compareNeedsMoreModels,
      onStopStreaming,
      onStopListening,
      onSubmitForm,
      sendMenuOpen,
      onSendMenuChange,
      t
    } = props

    const shouldQueuePrimaryAction = isSending || !isConnectionReady
    const primaryActionLabel = shouldQueuePrimaryAction
      ? t("common:queue", "Queue")
      : sendLabel
    const isMac =
      typeof navigator !== "undefined" &&
      /mac/i.test(navigator.platform || "")
    const primaryActionTitle = compareNeedsMoreModels
      ? (t(
          "playground:composer.validationCompareMinModelsInline",
          "Select at least two models for Compare mode."
        ) as string)
      : shouldQueuePrimaryAction
        ? ((isSending
            ? t(
                "playground:composer.queue.primaryWhileBusy",
                "Queue this request to run after the current response."
              )
            : t(
                "playground:composer.queue.primaryWhileOffline",
                "Queue this request until your tldw server reconnects."
              )) as string)
        : sendWhenEnter
          ? (t(
              "playground:composer.submitAriaEnter",
              "Send message (Enter)"
            ) as string)
          : (t(
              "playground:composer.submitAriaModEnter",
              isMac
                ? "Send message (\u2318+Enter)"
                : "Send message (Ctrl+Enter)"
            ) as string)

    return (
      <div className="flex items-center gap-2">
        <Space.Compact
          className={`!justify-end !w-auto ${
            isProMode ? "" : "!h-9 !rounded-full !px-3 !text-xs"
          }`}
        >
          <Button
            size={
              isMobileViewport
                ? "large"
                : isProMode
                  ? "middle"
                  : "small"
            }
            htmlType={shouldQueuePrimaryAction ? "button" : "submit"}
            onClick={
              shouldQueuePrimaryAction
                ? () => {
                    onStopListening()
                    onSubmitForm()
                  }
                : undefined
            }
            disabled={compareNeedsMoreModels}
            className={
              isMobileViewport
                ? "min-h-[44px] min-w-[44px]"
                : undefined
            }
            title={primaryActionTitle}
            aria-label={
              shouldQueuePrimaryAction
                ? (t(
                    "playground:composer.queue.primaryAria",
                    "Queue request"
                  ) as string)
                : (t(
                    "playground:composer.submitAria",
                    "Send message"
                  ) as string)
            }
          >
            <div
              className={`inline-flex items-center ${
                isProMode ? "gap-2" : "gap-1"
              }`}
            >
              {!shouldQueuePrimaryAction && sendWhenEnter ? (
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  stroke="currentColor"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  className="h-5 w-5"
                  viewBox="0 0 24 24"
                >
                  <path d="M9 10L4 15 9 20"></path>
                  <path d="M20 4v7a4 4 0 01-4 4H4"></path>
                </svg>
              ) : null}
              <span
                className={
                  isProMode
                    ? ""
                    : "text-[11px] font-semibold uppercase tracking-[0.12em]"
                }
              >
                {primaryActionLabel}
              </span>
            </div>
          </Button>
          <Dropdown
            open={sendMenuOpen}
            onOpenChange={(open) => onSendMenuChange(open)}
            disabled={compareNeedsMoreModels}
            trigger={["click"]}
            menu={{
              items: [
                {
                  key: 1,
                  label: (
                    <Checkbox
                      checked={sendWhenEnter}
                      onChange={(e) =>
                        onSendWhenEnterChange(e.target.checked)
                      }
                    >
                      {t("sendWhenEnter")}
                    </Checkbox>
                  )
                }
              ]
            }}
          >
            <Button
              size={
                isMobileViewport
                  ? "large"
                  : isProMode
                    ? "middle"
                    : "small"
              }
              disabled={compareNeedsMoreModels}
              className={
                isMobileViewport
                  ? "min-h-[44px] min-w-[44px]"
                  : undefined
              }
              aria-label={
                t(
                  "playground:composer.sendOptions",
                  "Open send options"
                ) as string
              }
              title={
                t(
                  "playground:composer.sendOptions",
                  "Open send options"
                ) as string
              }
              icon={
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth={1.5}
                  stroke="currentColor"
                  className={isProMode ? "w-5 h-5" : "w-4 h-4"}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="m19.5 8.25-7.5 7.5-7.5-7.5"
                  />
                </svg>
              }
            />
          </Dropdown>
        </Space.Compact>
        {isSending ? (
          <Tooltip title={t("tooltip.stopStreaming") as string}>
            <TldwButton
              variant="outline"
              size={isMobileViewport ? "lg" : "md"}
              iconOnly
              onClick={onStopStreaming}
              ariaLabel={t("tooltip.stopStreaming") as string}
            >
              <StopCircleIcon className="size-5 sm:size-4" />
            </TldwButton>
          </Tooltip>
        ) : null}
      </div>
    )
  })
