import React from "react"
import { Tabs } from "antd"
import { TextEditor } from "@/components/Option/AudiobookStudio/ContentInput/TextEditor"
import { ChapterList } from "@/components/Option/AudiobookStudio/ChapterEditor/ChapterList"
import { GenerationPanel as AudiobookGenerationPanel } from "@/components/Option/AudiobookStudio/Generation/GenerationPanel"
import { OutputPanel } from "@/components/Option/AudiobookStudio/Output/OutputPanel"

export const NarrationWorkflow: React.FC = () => {
  const [activeTab, setActiveTab] = React.useState("content")

  return (
    <section className="min-w-0 rounded-md border border-border bg-surface p-3">
      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={[
          {
            key: "content",
            label: "Content",
            children: <TextEditor onSplitComplete={() => setActiveTab("chapters")} />
          },
          {
            key: "chapters",
            label: "Chapters",
            children: <ChapterList />
          },
          {
            key: "voice",
            label: "Voice",
            children: <AudiobookGenerationPanel />
          },
          {
            key: "output",
            label: "Output",
            children: <OutputPanel />
          }
        ]}
      />
    </section>
  )
}
