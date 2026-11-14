import {
  MoreHorizontal,
  FileText,
  Share2,
  FileJson,
  FileCode,
  ImageIcon
} from "lucide-react"
import { Dropdown, MenuProps } from "antd"
import { IconButton } from "../Common/IconButton"
import { Message } from "@/types/message"
import { useState } from "react"
import { ShareModal } from "../Common/ShareModal"
import { createRoot } from "react-dom/client"
import { useTranslation } from "react-i18next"
import { removeModelSuffix } from "@/db/dexie/models"
import { copyToClipboard } from "@/utils/clipboard"
import { ImageExportWrapper } from "../Common/ImageExport"
import { useAntdMessage } from "@/hooks/useAntdMessage"
import { useStoreMessageOption } from "@/store/option"
import { trackCompareMetric } from "@/utils/compare-metrics"

interface MoreOptionsProps {
  messages: Message[]
  historyId: string
  shareModeEnabled: boolean
}

type CompareUserMessage = Message & {
  messageType: "compare:user"
  clusterId: string
}

type CompareReplyMessage = Message & {
  messageType: "compare:reply"
  clusterId: string
  id: string
}

const isCompareUserMessage = (message: Message): message is CompareUserMessage =>
  message.messageType === "compare:user" && typeof message.clusterId === "string"

const isCompareReplyMessage = (
  message: Message
): message is CompareReplyMessage =>
  message.messageType === "compare:reply" &&
  typeof message.clusterId === "string" &&
  typeof message.id === "string"

const buildCanonicalMessages = (
  messages: Message[],
  canonicalByCluster: Record<string, string | null>
): Message[] => {
  // Build a simple index of compare clusters
  const clusters: Record<
    string,
    {
      userIndex: number
      replyIndices: number[]
    }
  > = {}

  messages.forEach((message, index) => {
    if (!isCompareUserMessage(message)) {
      return
    }

    const clusterId = message.clusterId
    const replyIndices: number[] = []

    messages.forEach((otherMessage, otherIndex) => {
      if (
        otherIndex !== index &&
        isCompareReplyMessage(otherMessage) &&
        otherMessage.clusterId === clusterId
      ) {
        replyIndices.push(otherIndex)
      }
    })

    clusters[clusterId] = {
      userIndex: index,
      replyIndices
    }
  })

  const canonicalMessages: Message[] = []

  messages.forEach((message, index) => {
    const clusterId = message.clusterId

    if (!clusterId || !clusters[clusterId]) {
      canonicalMessages.push(message)
      return
    }

    const cluster = clusters[clusterId]

    if (isCompareUserMessage(message)) {
      canonicalMessages.push(message)
      return
    }

    if (isCompareReplyMessage(message)) {
      const replyIndices = cluster.replyIndices
      let canonicalIndex: number | null = null
      const canonicalId = canonicalByCluster?.[clusterId] ?? null

      if (canonicalId) {
        const targetIndex = replyIndices.find((replyIndex) => {
          const candidate = messages[replyIndex]
          return typeof candidate.id === "string" && candidate.id === canonicalId
        })

        if (typeof targetIndex === "number") {
          canonicalIndex = targetIndex
        }
      }

      if (canonicalIndex === null && replyIndices.length > 0) {
        canonicalIndex = replyIndices[0]
      }

      if (index === canonicalIndex) {
        canonicalMessages.push(message)
      }

      return
    }

    canonicalMessages.push(message)
  })

  return canonicalMessages
}
const formatAsText = (messages: Message[]) => {
  return messages
    .map((msg) => {
      const text = `${msg.isBot ? removeModelSuffix(`${msg.modelName || msg.name}`?.replaceAll(/accounts\/[^\/]+\/models\//g, "")) : "You"}: ${msg.message}`
      return text
    })
    .join("\n\n")
}
const formatAsMarkdown = (messages: Message[]) => {
  return messages
    .map((msg) => {
      let content = `### **${msg.isBot ? removeModelSuffix(`${msg.modelName || msg.name}`?.replaceAll(/accounts\/[^\/]+\/models\//g, "")) : "You"}**:\n\n${msg.message}`

      if (msg.images && msg.images.length > 0) {
        const imageMarkdown = msg.images
          .filter((img) => img.length > 0)
          .map((img) => `\n\n![Image](${img})`)
          .join("\n")
        content += imageMarkdown
      }

      return content
    })
    .join("\n\n")
}

const downloadFile = (content: string, filename: string) => {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" })
  const url = URL.createObjectURL(blob)
  const link = document.createElement("a")
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

const generateChatImage = async (messages: Message[]) => {
  // Lazy-load html2canvas to reduce initial bundle size (~400KB)
  const html2canvas = (await import("html2canvas")).default
  const rootStyles = getComputedStyle(document.documentElement)
  const themeBackground =
    rootStyles.getPropertyValue("--color-bg").trim() ||
    rootStyles.getPropertyValue("--color-surface").trim() ||
    null

  const root = document.createElement("div")
  document.body.appendChild(root)
  const element = <ImageExportWrapper messages={messages} />
  const reactRoot = createRoot(root)
  reactRoot.render(element)
  await new Promise((resolve) => setTimeout(resolve, 100))
  const container = document.getElementById("export-container")
  if (!container) {
    throw new Error("Export container not found")
  }
  const canvas = await html2canvas(container, {
    useCORS: true,
    backgroundColor: themeBackground,
    scale: 2
  })
  reactRoot.unmount()
  document.body.removeChild(root)

  return canvas.toDataURL("image/png")
}

export const MoreOptions = ({
  shareModeEnabled = false,
  historyId,
  messages
}: MoreOptionsProps) => {
  const { t } = useTranslation(["option", "settings"])
  const [onShareOpen, setOnShareOpen] = useState(false)
  const [open, setOpen] = useState(false)
  const message = useAntdMessage()
  const canonicalByCluster = useStoreMessageOption(
    (state) => state.compareCanonicalByCluster
  )
  // Confirm only a copy that happened; a refused write is shown, not left uncaught (G17).
  const copyAndReport = async (
    write: () => Promise<void>,
    afterCopy?: () => void
  ) => {
    try {
      await write()
    } catch {
      message.error(t("common:copyFailed", "Couldn't copy"))
      return
    }
    afterCopy?.()
    message.success(t("more.copy.success"))
  }
  const baseItems: MenuProps["items"] = [
    {
      type: "group",
      label: t("more.copy.group"),
      children: [
        {
          key: "copy-text",
          label: t("more.copy.asText"),
          icon: <FileText className="w-4 h-4" />,
          onClick: () =>
            copyAndReport(() =>
              copyToClipboard({
                text: formatAsText(messages),
                formatted: false
              })
            )
        },
        {
          key: "copy-as-formatted-text",
          label: t(
            "settings:generalSettings.settings.copyAsFormattedText.label"
          ),
          icon: <FileText className="w-4 h-4" />,
          onClick: () =>
            copyAndReport(() =>
              copyToClipboard({
                text: formatAsMarkdown(messages),
                formatted: true
              })
            )
        },
        {
          key: "copy-markdown",
          label: t("more.copy.asMarkdown"),
          icon: <FileCode className="w-4 h-4" />,
          onClick: () =>
            copyAndReport(() =>
              copyToClipboard({
                text: formatAsMarkdown(messages),
                formatted: false
              })
            )
        },
        {
          key: "copy-canonical-markdown",
          label: t("more.copy.canonicalMarkdown", {
            defaultValue: "Copy canonical transcript"
          }),
          icon: <FileCode className="w-4 h-4" />,
          onClick: () => {
            const canonicalMessages = buildCanonicalMessages(
              messages,
              canonicalByCluster || {}
            )
            return copyAndReport(
              () =>
                copyToClipboard({
                  text: formatAsMarkdown(canonicalMessages),
                  formatted: false
                }),
              () => void trackCompareMetric({ type: "export_canonical" })
            )
          }
        }
      ]
    },
    {
      type: "divider"
    },
    {
      type: "group",
      label: t("more.download.group"),
      children: [
        {
          key: "download-txt",
          label: t("more.download.text"),
          icon: <FileText className="w-4 h-4" />,
          onClick: () => {
            downloadFile(formatAsText(messages), "chat.txt")
          }
        },
        {
          key: "download-md",
          label: t("more.download.markdown"),
          icon: <FileCode className="w-4 h-4" />,
          onClick: () => {
            downloadFile(formatAsMarkdown(messages), "chat.md")
          }
        },
        {
          key: "download-canonical-md",
          label: t("more.download.canonicalMarkdown", {
            defaultValue: "Canonical transcript (Markdown)"
          }),
          icon: <FileCode className="w-4 h-4" />,
          onClick: () => {
            const canonicalMessages = buildCanonicalMessages(
              messages,
              canonicalByCluster || {}
            )
            downloadFile(
              formatAsMarkdown(canonicalMessages),
              "chat_canonical.md"
            )
            void trackCompareMetric({ type: "export_canonical" })
          }
        },
        {
          key: "download-json",
          label: t("more.download.json"),
          icon: <FileJson className="w-4 h-4" />,
          onClick: () => {
            const jsonContent = JSON.stringify(messages, null, 2)
            downloadFile(jsonContent, "chat.json")
          }
        },
        {
          key: "download-image",
          label: t("more.download.image"),
          icon: <ImageIcon className="w-4 h-4" />,
          onClick: async () => {
            try {
              const dataUrl = await generateChatImage(messages)
              const link = document.createElement("a")
              link.download = `chat_${new Date().toISOString()}.png`
              link.href = dataUrl
              link.click()
            } catch (e) {
              message.error("Failed to generate image")
            }
          }
        }
      ]
    }
  ]

  const shareItem = {
    type: "divider"
  } as const

  const shareOption = {
    key: "share",
    label: t("more.share"),
    icon: <Share2 className="w-4 h-4" />,
    onClick: () => {
      setOnShareOpen(true)
    }
  }

  const items = shareModeEnabled
    ? [...baseItems, shareItem, shareOption]
    : baseItems

  return (
    <>
      <Dropdown
        menu={{
          items
        }}
        trigger={["click"]}
        placement="bottomRight"
        open={open}
        onOpenChange={setOpen}>
        <IconButton
          className="!text-text-muted hover:text-text transition-colors h-11 w-11 sm:h-7 sm:w-7 sm:min-w-0 sm:min-h-0"
          ariaLabel={t("option:header.moreActions", "More actions")}
          hasPopup="menu"
          ariaExpanded={open}>
          <MoreHorizontal className="w-6 h-6" />
        </IconButton>
      </Dropdown>
      <ShareModal
        open={onShareOpen}
        historyId={historyId}
        messages={messages}
        setOpen={setOnShareOpen}
      />
    </>
  )
}
