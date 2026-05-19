import { Mic } from "lucide-react"
import type { TutorialDefinition } from "../registry"

const sttBasics: TutorialDefinition = {
  id: "stt-basics",
  routePattern: "/stt",
  labelKey: "tutorials:stt.basics.label",
  labelFallback: "STT Basics",
  descriptionKey: "tutorials:stt.basics.description",
  descriptionFallback: "Learn how to transcribe audio to text",
  icon: Mic,
  priority: 1,
  steps: [
    {
      target: '[data-testid="stt-record-strip"]',
      titleKey: "tutorials:stt.basics.recordTitle",
      titleFallback: "Record or Upload",
      contentKey: "tutorials:stt.basics.recordContent",
      contentFallback: "Press the record button to capture audio, or upload an audio file. You can also press Space to start recording.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="stt-model-selector"]',
      titleKey: "tutorials:stt.basics.modelTitle",
      titleFallback: "Select Model",
      contentKey: "tutorials:stt.basics.modelContent",
      contentFallback: "Choose a transcription model. You can select multiple models to compare results side by side.",
      placement: "bottom"
    },
    {
      target: '[data-testid="stt-transcription-output"]',
      titleKey: "tutorials:stt.basics.outputTitle",
      titleFallback: "View Results",
      contentKey: "tutorials:stt.basics.outputContent",
      contentFallback: "Transcription results appear here. You can export to Markdown or save as a note.",
      placement: "top"
    }
  ]
}

export const sttTutorials: TutorialDefinition[] = [sttBasics]
