import { useTTS } from "@/hooks/useTTS"
import { useStorage } from "@plasmohq/storage/hook"
import React from "react"
import { useTranslation } from "react-i18next"
import { EditMessageForm } from "./EditMessageForm"
import { Image, Tooltip } from "antd"
import {
  CheckIcon,
  CopyIcon,
  Pen,
  PlayIcon,
  Square,
  CornerUpLeft,
  Trash2
} from "lucide-react"
import { HumanMessage } from "./HumanMessge"
import { ChatDocuments } from "@/models/ChatTypes"
import { DocumentChip } from "./DocumentChip"
import { DocumentFile } from "./DocumentFile"
import { buildChatTextClass } from "@/utils/chat-style"
import { IconButton } from "../IconButton"
import { useUiModeStore } from "@/store/ui-mode"
import { useStoreMessageOption } from "@/store/option"
import { EDIT_MESSAGE_EVENT } from "@/utils/timeline-actions"
import { Badge, type BadgeVariant } from "@/components/ui/primitives"

const ACTION_BUTTON_CLASS =
  "flex items-center justify-center rounded-full border border-border bg-surface2 text-text-muted hover:bg-surface hover:text-text transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-focus min-w-[44px] min-h-[44px]"

const MESSAGE_TYPE_BADGE_VARIANT: Record<string, BadgeVariant> = {
  summary: "info",
  explain: "success",
  translate: "demo",
  custom: "primary",
  rephrase: "warning",
  greeting: "demo",
  "character:greeting": "demo"
}

const getMessageTypeBadgeVariant = (messageType?: string): BadgeVariant => {
  if (!messageType) return "secondary"
  if (messageType.startsWith("compare")) return "secondary"
  return MESSAGE_TYPE_BADGE_VARIANT[messageType] ?? "info"
}

const formatMessageTypeFallback = (messageType?: string): string => {
  if (!messageType) return "Unknown"
  const words = messageType
    .split(/[:_\-\s]+/)
    .filter(Boolean)
  if (words.length === 0) return "Unknown"
  return words
    .map((word) => word.toLowerCase())
    .map((word, index) =>
      index === 0 ? `${word.charAt(0).toUpperCase()}${word.slice(1)}` : word
    )
    .join(" ")
}

type Props = {
  message: string
  message_type?: string
  hideCopy?: boolean
  botAvatar?: JSX.Element
  userAvatar?: JSX.Element
  isBot: boolean
  name: string
  role?: "user" | "assistant" | "system"
  images?: string[]
  currentMessageIndex: number
  totalMessages: number
  onRegenerate: () => void
  onEditFormSubmit: (value: string, isSend: boolean) => void
  isProcessing: boolean
  webSearch?: unknown
  isSearchingInternet?: boolean
  sources?: unknown[]
  hideEditAndRegenerate?: boolean
  onSourceClick?: (source: unknown) => void
  isTTSEnabled?: boolean
  generationInfo?: unknown
  isStreaming: boolean
  reasoningTimeTaken?: number
  openReasoning?: boolean
  modelImage?: string
  modelName?: string
  documents?: ChatDocuments
  messageId?: string
  serverMessageId?: string | null
  createdAt?: number | string
  onDelete?: () => void
}

export const PlaygroundUserMessageBubble: React.FC<Props> = (props) => {
  const [checkWideMode] = useStorage("checkWideMode", false)
  const [userTextColor] = useStorage("chatUserTextColor", "default")
  const [userTextFont] = useStorage("chatUserTextFont", "default")
  const [userTextSize] = useStorage("chatUserTextSize", "md")
  const [userDisplayName] = useStorage("chatUserDisplayName", "")
  const [isBtnPressed, setIsBtnPressed] = React.useState(false)
  const timerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const [editMode, setEditMode] = React.useState(false)
  const { t } = useTranslation(["common", "playground"])
  const { cancel, isSpeaking, speak } = useTTS()
  const uiMode = useUiModeStore((state) => state.mode)
  const isProMode = uiMode === "pro"
  const setReplyTarget = useStoreMessageOption((state) => state.setReplyTarget)
  const actionRowVisibility = isProMode
    ? "flex"
    : "hidden group-hover:flex group-focus-within:flex"
  const overflowChipVisibility = isProMode
    ? "hidden"
    : "inline-flex group-hover:hidden"
  const showInlineActions = !props.isProcessing && !editMode
  const replyId = props.messageId ?? props.serverMessageId ?? null
  const canReply =
    isProMode && Boolean(replyId) && !props.message_type?.startsWith("compare")
  const resolvedRole = props.role ?? (props.isBot ? "assistant" : "user")
  const isSystemMessage = resolvedRole === "system"
  const systemLabel = t("playground:systemPrompt", "System prompt")
  const buildReplyPreview = React.useCallback(
    (value: string) => {
      const collapsed = value.replace(/\s+/g, " ").trim()
      if (!collapsed) {
        return t("replyTargetFallback", "Message")
      }
      if (collapsed.length > 140) {
        return `${collapsed.slice(0, 137)}...`
      }
      return collapsed
    },
    [t]
  )

  React.useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current)
        timerRef.current = null
      }
    }
  }, [])

  const handleEditMessage = React.useCallback(
    (event: Event) => {
      const detail = (event as CustomEvent<{ messageId?: string }>).detail
      if (!detail?.messageId) return
      if (props.isBot) return
      const matches =
        detail.messageId === props.messageId ||
        detail.messageId === props.serverMessageId
      if (matches) {
        setEditMode(true)
      }
    },
    [props.isBot, props.messageId, props.serverMessageId]
  )

  React.useEffect(() => {
    if (typeof window === "undefined") return

    window.addEventListener(EDIT_MESSAGE_EVENT, handleEditMessage)
    return () => {
      window.removeEventListener(EDIT_MESSAGE_EVENT, handleEditMessage)
    }
  }, [handleEditMessage])

  const messageTimestamp = React.useMemo(() => {
    const raw = props.createdAt
    if (!raw) return null
    const date =
      typeof raw === "number"
        ? new Date(raw)
        : new Date(Date.parse(String(raw)))
    if (Number.isNaN(date.getTime())) return null
    return date.toLocaleTimeString([], {
      hour: "numeric",
      minute: "2-digit"
    })
  }, [props.createdAt])

  const userTextClass = React.useMemo(
    () => buildChatTextClass(userTextColor, userTextFont, userTextSize),
    [userTextColor, userTextFont, userTextSize]
  )
  const bubbleToneClass = isSystemMessage
    ? "bg-warn/10 border-warn/30 border-dashed"
    : "bg-surface2/80 border-border"

  return (
    <div
      data-testid="chat-message"
      data-role={isSystemMessage ? "system" : "user"}
      data-index={props.currentMessageIndex}
      data-message-id={props.messageId}
      data-server-message-id={props.serverMessageId}
      className={`group gap-2 relative flex w-full max-w-5xl flex-col items-end justify-center pb-2 md:px-4 text-text ${checkWideMode ? "max-w-none" : ""}`}>
      <div className="flex w-full flex-wrap items-center justify-end gap-2">
        <div className="flex flex-wrap items-center gap-2">
          {isSystemMessage ? (
            <Badge
              data-testid="playground-system-message-badge"
              variant="warning"
              size="sm"
              outline
            >
              {systemLabel}
            </Badge>
          ) : (
            <span className="text-caption font-semibold text-text">
              {userDisplayName.trim() || t("common:you", "You")}
            </span>
          )}
          {messageTimestamp && (
            <span className="text-xs text-text-muted">
              • {messageTimestamp}
            </span>
          )}
          {!editMode && props?.message_type && (
            <Badge
              data-testid="playground-message-type-badge"
              variant={getMessageTypeBadgeVariant(props.message_type)}
              size="sm"
              className="!m-0"
            >
              {t(
                `copilot.${props.message_type}`,
                formatMessageTypeFallback(props.message_type)
              )}
            </Badge>
          )}
        </div>
      </div>

      {props?.documents &&
        props?.documents.length > 0 &&
        props.documents.filter((d) => d.type === "file").length > 0 && (
          <div className="flex flex-wrap gap-2">
            {props.documents
              .filter((d) => d.type === "file")
              .map((doc, index) => (
                <DocumentFile
                  key={index}
                  document={{
                    filename: doc.filename!,
                    fileSize: doc.fileSize!
                  }}
                />
              ))}
          </div>
        )}

      {props?.documents &&
        props?.documents.length > 0 &&
        props.documents.filter((d) => d.type === "tab").length > 0 && (
          <div className="flex flex-wrap gap-2">
            {props.documents
              .filter((d) => d.type === "tab")
              .map((doc, index) => (
                <DocumentChip
                  key={index}
                  document={{
                    title: doc.title,
                    url: doc.url,
                    favIconUrl: doc.favIconUrl
                  }}
                />
              ))}
          </div>
        )}

      {!editMode && props?.message?.length > 0 && (
        <div
          dir="auto"
          data-is-not-editable={!editMode}
          className={`message-bubble ${bubbleToneClass} shadow-sm rounded-3xl prose max-w-none dark:prose-invert break-words min-h-7 prose-p:opacity-95 prose-strong:opacity-100 border max-w-[calc(100%-1.75rem)] px-4 py-3 rounded-br-lg ${userTextClass} ${
            props.message_type && !editMode ? "italic" : ""
          }`}>
          <HumanMessage message={props.message} />
        </div>
      )}

      {editMode && (
        <div
          dir="auto"
          className={`message-bubble ${bubbleToneClass} shadow-sm rounded-3xl prose max-w-none dark:prose-invert break-words min-h-7 prose-p:opacity-95 prose-strong:opacity-100 border max-w-[calc(100%-1.75rem)] px-4 py-3 rounded-br-lg ${userTextClass} ${
            props.message_type && !editMode ? "italic" : ""
          }`}>
          <div className="w-screen max-w-[100%]">
            <EditMessageForm
              value={props.message}
              onSumbit={props.onEditFormSubmit}
              onClose={() => setEditMode(false)}
              isBot={props.isBot}
            />
          </div>
        </div>
      )}

      {props.images &&
        props.images.filter((img) => img.length > 0).length > 0 && (
          <div>
            {props.images
              .filter((image) => image.length > 0)
              .map((image, index) => (
                <Image
                  key={index}
                  src={image}
                  alt="Uploaded Image"
                  width={180}
                  className="rounded-lg relative"
                />
              ))}
          </div>
        )}

      {showInlineActions && (
        <div className="flex w-full justify-end">
          <div className="flex items-center gap-1">
            <button
              type="button"
              aria-label={t("moreActions", "More actions") as string}
              title={t("moreActions", "More actions") as string}
              className={`${overflowChipVisibility} rounded-full border border-border bg-surface2 px-2 py-0.5 text-xs text-text-muted transition-colors hover:text-text`}>
              •••
            </button>
            <div
              className={`${actionRowVisibility} flex-wrap items-center gap-1`}>
              {props.isTTSEnabled && (
                <Tooltip title={t("tts")}>
                  <IconButton
                    ariaLabel={t("tts") as string}
                    onClick={() => {
                      if (isSpeaking) {
                        cancel()
                      } else {
                        speak({
                          utterance: props.message,
                          saveClip: false
                        })
                      }
                    }}
                    className={`${ACTION_BUTTON_CLASS} h-11 w-11`}>
                    {!isSpeaking ? (
                      <PlayIcon className="w-3 h-3 text-text-subtle group-hover:text-text" />
                    ) : (
                      <Square className="w-3 h-3 text-danger group-hover:text-danger" />
                    )}
                  </IconButton>
                </Tooltip>
              )}
              {!props.hideCopy && (
                <Tooltip title={t("copyToClipboard")}>
                  <IconButton
                    ariaLabel={t("copyToClipboard") as string}
                    onClick={() => {
                      navigator.clipboard.writeText(props.message)
                      setIsBtnPressed(true)
                      if (timerRef.current) {
                        clearTimeout(timerRef.current)
                      }
                      timerRef.current = setTimeout(() => {
                        setIsBtnPressed(false)
                      }, 2000)
                    }}
                    className={`${ACTION_BUTTON_CLASS} h-11 w-11`}>
                    {!isBtnPressed ? (
                      <CopyIcon className="w-3 h-3 text-text-subtle group-hover:text-text" />
                    ) : (
                      <CheckIcon className="w-3 h-3 text-success group-hover:text-success" />
                    )}
                  </IconButton>
                </Tooltip>
              )}
              {canReply && (
                <Tooltip title={t("reply", "Reply")}>
                  <IconButton
                    ariaLabel={t("reply", "Reply") as string}
                    onClick={() => {
                      if (!replyId) return
                      setReplyTarget({
                        id: replyId,
                        preview: buildReplyPreview(props.message),
                        name: props.name,
                        isBot: props.isBot
                      })
                    }}
                    className={`${ACTION_BUTTON_CLASS} h-11 w-11`}>
                    <CornerUpLeft className="w-3 h-3 text-text-subtle group-hover:text-text" />
                  </IconButton>
                </Tooltip>
              )}

              {!props.hideEditAndRegenerate && (
                <Tooltip title={t("edit")}>
                  <IconButton
                    onClick={() => setEditMode(true)}
                    ariaLabel={t("edit") as string}
                    className={`${ACTION_BUTTON_CLASS} h-11 w-11`}>
                    <Pen className="w-3 h-3 text-text-subtle group-hover:text-text" />
                  </IconButton>
                </Tooltip>
              )}
              {props.onDelete && (
                <Tooltip title={t("delete", "Delete")}>
                  <IconButton
                    onClick={props.onDelete}
                    ariaLabel={t("delete", "Delete") as string}
                    className={`${ACTION_BUTTON_CLASS} h-11 w-11`}>
                    <Trash2 className="w-3 h-3 text-danger group-hover:text-danger" />
                  </IconButton>
                </Tooltip>
              )}
            </div>
          </div>
        </div>
      )}

    </div>
  )
}
