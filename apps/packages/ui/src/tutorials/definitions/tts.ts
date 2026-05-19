import { Volume2 } from "lucide-react"
import type { TutorialDefinition } from "../registry"

const ttsBasics: TutorialDefinition = {
  id: "tts-basics",
  routePattern: "/tts",
  labelKey: "tutorials:tts.basics.label",
  labelFallback: "TTS Basics",
  descriptionKey: "tutorials:tts.basics.description",
  descriptionFallback: "Learn how to generate spoken audio from text",
  icon: Volume2,
  priority: 1,
  steps: [
    {
      target: '[data-testid="tts-provider-selector"]',
      titleKey: "tutorials:tts.basics.providerTitle",
      titleFallback: "Choose a Provider",
      contentKey: "tutorials:tts.basics.providerContent",
      contentFallback: "Select a TTS provider. Browser TTS works without any setup. TLDW providers need server configuration.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="tts-text-input"]',
      titleKey: "tutorials:tts.basics.textTitle",
      titleFallback: "Enter Text",
      contentKey: "tutorials:tts.basics.textContent",
      contentFallback: "Type or paste the text you want to convert to speech. Longer text is split into segments automatically.",
      placement: "bottom"
    },
    {
      target: '[data-testid^="tts-voice-picker-"]',
      titleKey: "tutorials:tts.basics.voiceTitle",
      titleFallback: "Pick a Voice",
      contentKey: "tutorials:tts.basics.voiceContent",
      contentFallback: "Choose from available voices. Each provider offers different voice options and styles.",
      placement: "bottom"
    },
    {
      target: '[data-testid="tts-play-button"]',
      titleKey: "tutorials:tts.basics.playTitle",
      titleFallback: "Generate Audio",
      contentKey: "tutorials:tts.basics.playContent",
      contentFallback: "Click Play to generate speech. You can listen to each segment individually or the full output.",
      placement: "top"
    }
  ]
}

export const ttsTutorials: TutorialDefinition[] = [ttsBasics]
