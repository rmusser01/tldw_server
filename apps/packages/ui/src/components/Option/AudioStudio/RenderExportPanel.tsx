import React from "react"
import { Button, Divider, Typography } from "antd"
import { Download, SlidersHorizontal } from "lucide-react"
import { useAudioStudioStore } from "@/store/audio-studio"

const { Text } = Typography

export const RenderExportPanel: React.FC = () => {
  const activeProject = useAudioStudioStore((state) => state.activeProject)
  return (
    <section className="rounded-md border border-border bg-surface p-3">
      <Text strong className="block">
        Render & Export
      </Text>
      <Text type="secondary" className="mt-1 block text-xs">
        Preview render and export job wiring is server-backed; full render/export UI
        lands in TASK-2351.
      </Text>
      <Divider className="my-3" />
      <div className="space-y-2">
        <Button block icon={<SlidersHorizontal className="h-4 w-4" />} disabled={!activeProject}>
          Create preview render
        </Button>
        <Button block icon={<Download className="h-4 w-4" />} disabled={!activeProject}>
          Create export
        </Button>
      </div>
    </section>
  )
}
