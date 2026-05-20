import React, { useState, useCallback, useRef } from "react"
import {
  Card,
  Typography,
  Button,
  Space,
  Empty,
  Table,
  Tag,
  Tooltip,
  Dropdown,
  Progress,
  message
} from "antd"
import type { MenuProps } from "antd"
import { useTranslation } from "react-i18next"
import {
  Download,
  Play,
  Pause,
  Check,
  Clock,
  AlertCircle,
  Merge,
  Subtitles,
  ChevronDown
} from "lucide-react"
import { useAudiobookStudioStore, type AudioChapter } from "@/store/audiobook-studio"
import { useAudiobookGeneration } from "@/hooks/useAudiobookGeneration"
import {
  concatenateAudioChapters,
  type ChapterTiming
} from "@/utils/audio-concat"
import { downloadBlob } from "@/utils/download-blob"
import { exportSubtitles, type SubtitleFormat } from "@/utils/subtitle-generator"
import { AudiobookPlayer } from "./AudiobookPlayer"

const { Text, Title } = Typography

export const OutputPanel: React.FC = () => {
  const { t } = useTranslation(["audiobook", "common"])
  const [playingId, setPlayingId] = useState<string | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  // Combined audiobook state
  const [isCombining, setIsCombining] = useState(false)
  const [combineProgress, setCombineProgress] = useState(0)
  const [combinedAudioUrl, setCombinedAudioUrl] = useState<string | null>(null)
  const [combinedAudioBlob, setCombinedAudioBlob] = useState<Blob | null>(null)
  const [chapterTimings, setChapterTimings] = useState<ChapterTiming[]>([])

  const chapters = useAudiobookStudioStore((s) => s.chapters)
  const projectTitle = useAudiobookStudioStore((s) => s.projectTitle)
  const getTotalDuration = useAudiobookStudioStore((s) => s.getTotalDuration)

  const { downloadChapter, downloadAllChapters } = useAudiobookGeneration()

  const sortedChapters = React.useMemo(
    () => [...chapters].sort((a, b) => a.order - b.order),
    [chapters]
  )

  // Combine all chapters into a single audio file
  const handleCombineChapters = useCallback(async () => {
    const toProcess = sortedChapters.filter(
      (ch) => ch.status === "completed" && ch.audioBlob
    )

    if (toProcess.length === 0) {
      message.warning(t("audiobook:output.noChaptersToCombine", "No chapters to combine"))
      return
    }

    setIsCombining(true)
    setCombineProgress(0)

    try {
      const result = await concatenateAudioChapters(
        toProcess.map((ch) => ({
          id: ch.id,
          title: ch.title,
          blob: ch.audioBlob!
        })),
        (index, total) => {
          setCombineProgress(Math.round(((index + 1) / total) * 100))
        }
      )

      // Revoke previous URL if exists
      if (combinedAudioUrl) {
        URL.revokeObjectURL(combinedAudioUrl)
      }

      const url = URL.createObjectURL(result.blob)
      setCombinedAudioUrl(url)
      setCombinedAudioBlob(result.blob)
      setChapterTimings(result.chapterTimings)

      message.success(t("audiobook:output.combineSuccess", "Audiobook combined successfully"))
    } catch (error) {
      console.error("Failed to combine chapters:", error)
      message.error(t("audiobook:output.combineError", "Failed to combine chapters"))
    } finally {
      setIsCombining(false)
      setCombineProgress(0)
    }
  }, [sortedChapters, combinedAudioUrl, t])

  // Download combined audiobook
  const handleDownloadCombined = useCallback(() => {
    if (!combinedAudioBlob) return

    const blobType = combinedAudioBlob.type.toLowerCase()
    const extension = blobType.includes("wav")
      ? "wav"
      : blobType.includes("opus")
        ? "opus"
        : blobType.includes("aac")
          ? "aac"
          : blobType.includes("flac")
            ? "flac"
            : blobType.includes("pcm") || blobType.includes("l16")
              ? "pcm"
              : blobType.includes("ogg")
                ? "ogg"
                : "mp3"
    const filename = `${projectTitle.replace(/[^a-zA-Z0-9]/g, "_")}_complete.${extension}`
    downloadBlob(combinedAudioBlob, filename)
  }, [combinedAudioBlob, projectTitle])

  // Export subtitles
  const handleExportSubtitles = useCallback(
    (format: SubtitleFormat) => {
      if (chapterTimings.length === 0) {
        message.warning(
          t("audiobook:output.combineFirst", "Combine chapters first to export subtitles")
        )
        return
      }

      const filename = projectTitle.replace(/[^a-zA-Z0-9]/g, "_")
      exportSubtitles(chapterTimings, filename, format)
      message.success(
        t("audiobook:output.subtitleExported", "Subtitles exported as {{format}}", { format: format.toUpperCase() })
      )
    },
    [chapterTimings, projectTitle, t]
  )

  // Subtitle dropdown menu
  const subtitleMenuItems: MenuProps["items"] = [
    {
      key: "srt",
      label: t("audiobook:output.exportSrt", "Export SRT"),
      onClick: () => handleExportSubtitles("srt")
    },
    {
      key: "vtt",
      label: t("audiobook:output.exportVtt", "Export VTT"),
      onClick: () => handleExportSubtitles("vtt")
    }
  ]

  // Cleanup on unmount
  React.useEffect(() => {
    return () => {
      if (combinedAudioUrl) {
        URL.revokeObjectURL(combinedAudioUrl)
      }
    }
  }, [combinedAudioUrl])

  const completedChapters = sortedChapters.filter(
    (ch) => ch.status === "completed" && ch.audioBlob
  )
  const totalDuration = getTotalDuration()

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const mins = Math.floor((seconds % 3600) / 60)
    const secs = Math.floor(seconds % 60)
    if (hours > 0) {
      return `${hours}:${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`
    }
    return `${mins}:${secs.toString().padStart(2, "0")}`
  }

  const handlePlay = (chapter: AudioChapter) => {
    if (!chapter.audioUrl) return

    if (playingId === chapter.id) {
      // Stop playing
      if (audioRef.current) {
        audioRef.current.pause()
        audioRef.current.currentTime = 0
      }
      setPlayingId(null)
      return
    }

    // Start playing new chapter
    if (audioRef.current) {
      audioRef.current.pause()
    }

    const audio = new Audio(chapter.audioUrl)
    audioRef.current = audio
    audio.onended = () => setPlayingId(null)
    audio.onerror = () => setPlayingId(null)
    audio.play()
    setPlayingId(chapter.id)
  }

  React.useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause()
        audioRef.current = null
      }
    }
  }, [])

  const getStatusIcon = (status: AudioChapter["status"]) => {
    switch (status) {
      case "completed":
        return <Check className="h-4 w-4 text-success" />
      case "generating":
        return <Clock className="h-4 w-4 text-primary animate-pulse" />
      case "error":
        return <AlertCircle className="h-4 w-4 text-danger" />
      default:
        return <Clock className="h-4 w-4 text-text-subtle" />
    }
  }

  const columns = [
    {
      title: "#",
      dataIndex: "order",
      key: "order",
      width: 50,
      render: (_value: unknown, _record: unknown, index: number) => index + 1
    },
    {
      title: t("audiobook:output.columnTitle", "Title"),
      dataIndex: "title",
      key: "title",
      render: (title: string, record: AudioChapter) => (
        <div className="flex items-center gap-2">
          {getStatusIcon(record.status)}
          <Text>{title}</Text>
        </div>
      )
    },
    {
      title: t("audiobook:output.duration", "Duration"),
      dataIndex: "audioDuration",
      key: "duration",
      width: 100,
      render: (duration: number | undefined) =>
        duration ? formatDuration(duration) : "-"
    },
    {
      title: t("audiobook:output.status", "Status"),
      dataIndex: "status",
      key: "status",
      width: 120,
      render: (status: AudioChapter["status"]) => {
        const colors: Record<string, string> = {
          completed: "success",
          generating: "processing",
          error: "error",
          pending: "default"
        }
        const labels: Record<string, string> = {
          completed: t("audiobook:status.completed", "Completed"),
          generating: t("audiobook:status.generating", "Generating"),
          error: t("audiobook:status.error", "Error"),
          pending: t("audiobook:status.pending", "Pending")
        }
        return <Tag color={colors[status]}>{labels[status]}</Tag>
      }
    },
    {
      title: t("audiobook:output.actions", "Actions"),
      key: "actions",
      width: 150,
      render: (_value: unknown, record: AudioChapter) => (
        <Space size="small">
          {record.status === "completed" && record.audioUrl && (
            <>
              <Tooltip
                title={
                  playingId === record.id
                    ? t("audiobook:output.stop", "Stop")
                    : t("audiobook:output.play", "Play")
                }
              >
                <Button
                  size="small"
                  icon={
                    playingId === record.id ? (
                      <Pause className="h-3 w-3" />
                    ) : (
                      <Play className="h-3 w-3" />
                    )
                  }
                  onClick={() => handlePlay(record)}
                />
              </Tooltip>
              <Tooltip title={t("audiobook:output.download", "Download")}>
                <Button
                  size="small"
                  icon={<Download className="h-3 w-3" />}
                  onClick={() => downloadChapter(record)}
                />
              </Tooltip>
            </>
          )}
        </Space>
      )
    }
  ]

  if (chapters.length === 0) {
    return (
      <Card>
        <Empty
          description={
            <div className="text-center">
              <Text type="secondary" className="block">
                {t(
                  "audiobook:output.generatedChaptersAppearHere",
                  "Generated chapters appear here after generation."
                )}
              </Text>
              <Text type="secondary" className="text-xs">
                {t(
                  "audiobook:output.noChapters",
                  "Add content, split it into chapters, then generate audio."
                )}
              </Text>
            </div>
          }
        />
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      <Card>
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div>
            <Title level={5} className="!mb-1">
              {t("audiobook:output.panelTitle", "Output")}
            </Title>
            <div className="flex flex-wrap items-center gap-4 text-sm text-text-muted">
              <span>
                {t("audiobook:output.completedChapters", "{{count}} chapters ready", {
                  count: completedChapters.length
                })}
              </span>
              {totalDuration > 0 && (
                <span>
                  {t("audiobook:output.totalDuration", "Total: {{duration}}", {
                    duration: formatDuration(totalDuration)
                  })}
                </span>
              )}
            </div>
          </div>
          <Space wrap>
            <Button
              icon={<Merge className="h-4 w-4" />}
              onClick={handleCombineChapters}
              loading={isCombining}
              disabled={completedChapters.length === 0}
            >
              {isCombining
                ? t("audiobook:output.combining", "Combining... {{progress}}%", { progress: combineProgress })
                : t("audiobook:output.combine", "Combine All")}
            </Button>
            {combinedAudioBlob && (
              <Button
                type="primary"
                icon={<Download className="h-4 w-4" />}
                onClick={handleDownloadCombined}
              >
                {t("audiobook:output.downloadCombined", "Download Audiobook")}
              </Button>
            )}
            <Dropdown
              menu={{ items: subtitleMenuItems }}
              disabled={chapterTimings.length === 0}
            >
              <Button icon={<Subtitles className="h-4 w-4" />}>
                {t("audiobook:output.subtitles", "Subtitles")} <ChevronDown className="h-3 w-3 ml-1" />
              </Button>
            </Dropdown>
            <Button
              icon={<Download className="h-4 w-4" />}
              onClick={() => downloadAllChapters(sortedChapters, projectTitle)}
              disabled={completedChapters.length === 0}
            >
              {t("audiobook:output.downloadAll", "Download All Chapters")}
            </Button>
          </Space>
        </div>

        {isCombining && (
          <Progress
            percent={combineProgress}
            status="active"
            strokeColor={{ from: "rgb(var(--color-primary))", to: "rgb(var(--color-success))" }}
            className="mb-4"
          />
        )}

        <Table
          dataSource={sortedChapters}
          columns={columns}
          rowKey="id"
          size="small"
          pagination={false}
        />
      </Card>

      {/* Combined Audiobook Player */}
      {(combinedAudioUrl || completedChapters.length > 0) && (
        <AudiobookPlayer
          audioUrl={combinedAudioUrl}
          chapterTimings={chapterTimings}
        />
      )}

      <Card>
        <div className="space-y-3">
          <Title level={5} className="!mb-1">
            {t("audiobook:output.exportInfo", "Export Information")}
          </Title>
          <div className="text-sm text-text-muted space-y-1">
            <p>
              {t(
                "audiobook:output.exportInfoDesc1",
                "Each chapter is exported as a separate audio file in MP3 format."
              )}
            </p>
            <p>
              {t(
                "audiobook:output.exportInfoDesc2",
                "Files are named with the project title and chapter number for easy organization."
              )}
            </p>
            <p>
              {t(
                "audiobook:output.exportInfoDesc3",
                "Tip: Use Combine All to download a single WAV audiobook file."
              )}
            </p>
          </div>
        </div>
      </Card>
    </div>
  )
}

export default OutputPanel
