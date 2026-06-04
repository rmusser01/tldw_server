import React from "react"
import { QuickChatMessage as QuickChatMessageType } from "@/store/quick-chat"
import { classNames } from "@/libs/class-name"
import { useTranslation } from "react-i18next"
import { useStorage } from "@plasmohq/storage/hook"
import { DEFAULT_CHAT_SETTINGS } from "@/types/chat-settings"

const Markdown = React.lazy(() => import("@/components/Common/Markdown"))

type Props = {
  message: QuickChatMessageType
  isStreaming?: boolean
  isLast?: boolean
}

export const QuickChatMessage: React.FC<Props> = ({
  message,
  isStreaming = false,
  isLast = false
}) => {
  const { t } = useTranslation("option")
  const [renderMermaidDiagrams] = useStorage(
    "renderMermaidDiagrams",
    DEFAULT_CHAT_SETTINGS.renderMermaidDiagrams
  )
  const isUser = message.role === "user"
  const showStreamingCursor = isStreaming && isLast && !isUser
  const assistantContent =
    message.content + (showStreamingCursor ? "▋" : "")

  return (
    <div
      className={classNames(
        "flex w-full mb-3",
        isUser ? "justify-end" : "justify-start"
      )}
      role="article"
      aria-label={
        isUser
          ? t("option:quickChatHelper.userMessageAria", "Your message")
          : t("option:quickChatHelper.assistantMessageAria", "Assistant message")
      }>
      <div
        className={classNames(
          "max-w-[85%] rounded-lg px-3 py-2",
          isUser
            ? "bg-[color:var(--color-primary)] text-white"
            : "bg-[color:var(--color-surface-2)] text-[color:var(--color-text)]"
        )}>
        {isUser ? (
          <p className="text-sm whitespace-pre-wrap break-words">
            {message.content}
          </p>
        ) : (
          <div className="text-sm">
            {message.content ? (
              <React.Suspense
                fallback={
                  <p className="whitespace-pre-wrap break-words">
                    {assistantContent}
                  </p>
                }
              >
                <Markdown
                  message={assistantContent}
                  className="prose prose-sm dark:prose-invert prose-p:leading-relaxed prose-pre:p-0 max-w-none"
                  enableMermaidDiagrams={
                    message.role !== "user" &&
                    renderMermaidDiagrams !== false &&
                    !isStreaming
                  }
                />
              </React.Suspense>
            ) : showStreamingCursor ? (
              <span className="inline-block animate-pulse">▋</span>
            ) : null}
          </div>
        )}
      </div>
    </div>
  )
}

export default QuickChatMessage
