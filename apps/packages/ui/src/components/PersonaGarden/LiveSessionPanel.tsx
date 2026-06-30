import React from "react"

type LiveSessionPanelProps = {
  controls: React.ReactNode
  assistantVoice?: React.ReactNode
  diagnostics?: React.ReactNode
  error: React.ReactNode
  pendingPlan: React.ReactNode
  transcript: React.ReactNode
  composer: React.ReactNode
}

export const LiveSessionPanel: React.FC<LiveSessionPanelProps> = ({
  controls,
  assistantVoice,
  diagnostics,
  error,
  pendingPlan,
  transcript,
  composer
}) => {
  return (
    <div className="flex flex-1 flex-col gap-3">
      {controls}
      {assistantVoice}
      {diagnostics}
      {error}
      {pendingPlan}
      {transcript}
      {composer}
    </div>
  )
}
