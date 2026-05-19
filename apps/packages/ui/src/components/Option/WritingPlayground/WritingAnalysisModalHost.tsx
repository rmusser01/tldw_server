import React, { lazy, Suspense } from "react"
import { Spin } from "antd"
import { useWritingPlaygroundStore } from "@/store/writing-playground"

const LazyStoryPulse = lazy(() => import("./modals/StoryPulseModal"))
const LazyPlotTracker = lazy(() => import("./modals/PlotTrackerModal"))
const LazyEventLine = lazy(() => import("./modals/EventLineModal"))
const LazyConnectionWeb = lazy(() => import("./modals/ConnectionWebModal"))

export function WritingAnalysisModalHost() {
  const { analysisModalOpen, setAnalysisModalOpen } = useWritingPlaygroundStore()
  const onClose = () => setAnalysisModalOpen(null)

  return (
    <Suspense fallback={<Spin />}>
      {analysisModalOpen === "pulse" && <LazyStoryPulse open onClose={onClose} />}
      {analysisModalOpen === "plot" && <LazyPlotTracker open onClose={onClose} />}
      {analysisModalOpen === "timeline" && <LazyEventLine open onClose={onClose} />}
      {analysisModalOpen === "web" && <LazyConnectionWeb open onClose={onClose} />}
    </Suspense>
  )
}
