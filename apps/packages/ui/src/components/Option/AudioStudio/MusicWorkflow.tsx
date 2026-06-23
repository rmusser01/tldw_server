import React from "react"
import { Button, Input, Slider, Typography } from "antd"

const { TextArea } = Input
const { Text } = Typography

export const MusicWorkflow: React.FC = () => {
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
            />
          </label>
        </div>
        <div className="space-y-4">
          <label className="block">
            <Text strong className="mb-1 block">
              Style
            </Text>
            <Input aria-label="Style" placeholder="cinematic, ambient" />
          </label>
          <label className="block">
            <Text strong className="mb-1 block">
              Provider
            </Text>
            <select
              aria-label="Provider"
              className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm"
              defaultValue="ace_step"
            >
              <option value="ace_step">ACE-Step</option>
              <option value="server_default">Server default</option>
            </select>
          </label>
          <div>
            <Text strong className="mb-2 block">
              Duration
            </Text>
            <Slider min={15} max={180} defaultValue={45} />
          </div>
          <Button type="primary" block>
            Generate music
          </Button>
        </div>
      </div>
    </section>
  )
}
