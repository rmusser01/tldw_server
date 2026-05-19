export type AudioRouteConcept =
  | "audio_alias"
  | "speech_combined"
  | "stt"
  | "tts"
  | "audiobook"

export type AudioRouteOwner =
  | "next_alias"
  | "shared_route"
  | "extension_route"

export type AudioRouteCapability =
  | "hosted_gate"
  | "server_capability"
  | "provider_config"
  | "model_catalog"
  | "voice_catalog"
  | "recording_source"
  | "project_state"
  | "generation_state"

export type AudioRouteFinding =
  | "F2 support"
  | "F9 support"
  | "F15 support"
  | "F18 support"
  | "F19 support"

export type AudioRouteJob = {
  route: "/audio" | "/speech" | "/stt" | "/tts" | "/audiobook-studio"
  concept: AudioRouteConcept
  label: string
  primaryJob: string
  primaryActionLabel: string
  routeOwner: AudioRouteOwner
  canonicalComponent: string
  capabilities: AudioRouteCapability[]
  routeStatePolicy: "alias" | "ready_or_recoverable" | "beta_ready_or_recoverable"
  findings: AudioRouteFinding[]
}

export const AUDIO_ROUTE_JOBS: AudioRouteJob[] = [
  {
    route: "/audio",
    concept: "audio_alias",
    label: "Audio",
    primaryJob: "Open the canonical combined speech route from old links and bookmarks.",
    primaryActionLabel: "Open Speech Playground",
    routeOwner: "next_alias",
    canonicalComponent: "RouteRedirect:/speech",
    capabilities: [],
    routeStatePolicy: "alias",
    findings: ["F2 support", "F18 support", "F19 support"]
  },
  {
    route: "/speech",
    concept: "speech_combined",
    label: "Speech Playground",
    primaryJob: "Record, transcribe, edit, and synthesize audio in one workspace.",
    primaryActionLabel: "Start audio workflow",
    routeOwner: "shared_route",
    canonicalComponent: "SpeechPlaygroundPage",
    capabilities: [
      "server_capability",
      "provider_config",
      "model_catalog",
      "voice_catalog",
      "recording_source"
    ],
    routeStatePolicy: "ready_or_recoverable",
    findings: [
      "F2 support",
      "F9 support",
      "F15 support",
      "F18 support",
      "F19 support"
    ]
  },
  {
    route: "/stt",
    concept: "stt",
    label: "STT Playground",
    primaryJob: "Transcribe audio and compare transcription results.",
    primaryActionLabel: "Start transcription",
    routeOwner: "shared_route",
    canonicalComponent: "SttPlaygroundPage",
    capabilities: [
      "hosted_gate",
      "server_capability",
      "model_catalog",
      "recording_source"
    ],
    routeStatePolicy: "ready_or_recoverable",
    findings: [
      "F2 support",
      "F9 support",
      "F15 support",
      "F18 support",
      "F19 support"
    ]
  },
  {
    route: "/tts",
    concept: "tts",
    label: "TTS Playground",
    primaryJob: "Generate audio from text with provider, voice, and model controls.",
    primaryActionLabel: "Generate speech",
    routeOwner: "shared_route",
    canonicalComponent: "SpeechPlaygroundPage:listen",
    capabilities: [
      "hosted_gate",
      "server_capability",
      "provider_config",
      "voice_catalog"
    ],
    routeStatePolicy: "ready_or_recoverable",
    findings: [
      "F2 support",
      "F9 support",
      "F15 support",
      "F18 support",
      "F19 support"
    ]
  },
  {
    route: "/audiobook-studio",
    concept: "audiobook",
    label: "Audiobook Studio",
    primaryJob: "Create long-form audiobook projects from text and generated speech.",
    primaryActionLabel: "Create project",
    routeOwner: "shared_route",
    canonicalComponent: "AudiobookStudioPage",
    capabilities: [
      "provider_config",
      "voice_catalog",
      "project_state",
      "generation_state"
    ],
    routeStatePolicy: "beta_ready_or_recoverable",
    findings: [
      "F2 support",
      "F9 support",
      "F15 support",
      "F18 support",
      "F19 support"
    ]
  }
]

export const getAudioRouteJob = (
  route: AudioRouteJob["route"]
): AudioRouteJob | undefined =>
  AUDIO_ROUTE_JOBS.find((job) => job.route === route)
