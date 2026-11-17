import React from "react"
import { Select } from "antd"
import { useTranslation } from "react-i18next"
import type { ChatMessageView } from "@/services/settings/ui-settings"

type Props = {
  value: ChatMessageView
  onChange: (value: ChatMessageView) => void
}

export function MessageViewSelect({ value, onChange }: Props) {
  const { t } = useTranslation("playground")
  const options = [
    { value: "transcript" as const, label: t("messageViews.transcript", "Transcript"), description: t("messageViews.transcriptHelp", "For reading and everyday chat") },
    { value: "conversation" as const, label: t("messageViews.conversation", "Conversation"), description: t("messageViews.conversationHelp", "For roleplay and character chat") },
    { value: "research" as const, label: t("messageViews.research", "Research"), description: t("messageViews.researchHelp", "For sources and detailed review") }
  ]
  return (
    <Select<ChatMessageView>
      aria-label={t("messageViews.label", "Message view")}
      value={value}
      onChange={next => onChange(next)}
      options={options}
      virtual={false}
      variant="borderless"
      style={{ height: 44 }}
      popupMatchSelectWidth={280}
      className="min-w-[8.5rem] shrink-0"
      optionRender={({ data }) => (
        <div className="py-1">
          <div className="text-sm font-medium text-text">{data.label}</div>
          <div className="mt-0.5 text-xs text-text-muted">{data.description}</div>
        </div>
      )}
    />
  )
}
