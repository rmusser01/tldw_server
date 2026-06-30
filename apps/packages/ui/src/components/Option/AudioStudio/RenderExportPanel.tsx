import React from "react"
import { Button, Divider, Typography } from "antd"
import { Download, SlidersHorizontal } from "lucide-react"

const { Text } = Typography
const DEFERRED_RENDER_EXPORT_TITLE =
  "Render/export controls need a ready timeline render controls slice."

export const RenderExportPanel: React.FC = () => {
  return (
    <section className="rounded-md border border-border bg-surface p-3">
      <Text strong className="block">
        Render & Export
      </Text>
      <Text type="secondary" className="mt-1 block text-xs">
        Preview render and export job wiring is server-backed; full render/export UI
        remains a follow-up controls slice.
      </Text>
      <Divider className="my-3" />
      <div className="space-y-2">
        <Button
          block
          icon={<SlidersHorizontal className="h-4 w-4" />}
          disabled
          title={DEFERRED_RENDER_EXPORT_TITLE}
        >
          Create preview render
        </Button>
        <Button
          block
          icon={<Download className="h-4 w-4" />}
          disabled
          title={DEFERRED_RENDER_EXPORT_TITLE}
        >
          Create export
        </Button>
      </div>
    </section>
  )
}
