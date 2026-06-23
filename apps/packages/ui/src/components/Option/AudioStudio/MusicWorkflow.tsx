import React from "react"
import { Button, Input, Slider, Typography } from "antd"
import { useAudioStudioGenerationActions } from "./useAudioStudioGenerationActions"

const { TextArea } = Input
const { Text } = Typography

export const MusicWorkflow: React.FC = () => {
  const {
    isPending,
    musicDisabledReason,
    queueMusicGeneration
  } = useAudioStudioGenerationActions()
  const [prompt, setPrompt] = React.useState("")
  const [lyrics, setLyrics] = React.useState("")
  const [style, setStyle] = React.useState("")
  const [provider, setProvider] = React.useState("ace_step")
  const [duration, setDuration] = React.useState(45)
  const trimmedPrompt = prompt.trim()
  const disabledReason =
    musicDisabledReason ??
    (!trimmedPrompt ? "Enter a prompt before generating music." : undefined)

  const submitMusicGeneration = () => {
    if (disabledReason) return

    queueMusicGeneration(
      {
        prompt: trimmedPrompt,
        lyrics,
        style,
        duration
      },
      provider
    )
  }

  return (
    <section className="min-w-0 rounded-md border border-border bg-surface p-4">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_240px]">
        <div className="space-y-3">
          <label className="block">
            <Text strong className="mb-1 block">
              Prompt
            </Text>
            <TextArea
              aria-label="Prompt"
              placeholder="Warm documentary intro with restrained percussion"
              autoSize={{ minRows: 5, maxRows: 10 }}
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
            />
          </label>
          <label className="block">
            <Text strong className="mb-1 block">
              Lyrics
            </Text>
            <TextArea
              aria-label="Lyrics"
              placeholder="Optional lyrics or vocal direction"
              autoSize={{ minRows: 5, maxRows: 10 }}
              value={lyrics}
              onChange={(event) => setLyrics(event.target.value)}
            />
          </label>
        </div>
        <div className="space-y-4">
          <label className="block">
            <Text strong className="mb-1 block">
              Style
            </Text>
            <Input
              aria-label="Style"
              placeholder="cinematic, ambient"
              value={style}
              onChange={(event) => setStyle(event.target.value)}
            />
          </label>
          <label className="block">
            <Text strong className="mb-1 block">
              Provider
            </Text>
            <select
              aria-label="Provider"
              className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm"
              value={provider}
              onChange={(event) => setProvider(event.target.value)}
            >
              <option value="ace_step">ACE-Step</option>
            </select>
          </label>
          <div>
            <Text strong className="mb-2 block">
              Duration
            </Text>
            <Slider
              min={15}
              max={180}
              value={duration}
              onChange={(value) => {
                if (typeof value === "number") setDuration(value)
              }}
            />
          </div>
          <Button
            type="primary"
            block
            disabled={Boolean(disabledReason) || isPending}
            title={disabledReason}
            loading={isPending}
            onClick={submitMusicGeneration}
          >
            Generate music
          </Button>
        </div>
      </div>
    </section>
  )
}
