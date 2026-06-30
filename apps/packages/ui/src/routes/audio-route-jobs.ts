export type AudioRouteConcept =
  | "audio_alias"
  | "speech_combined"
  | "stt"
  | "tts"
  | "audiobook"

export type AudioRouteOwner = "shared_alias" | "shared_route"

export type AudioRouteCapability =
  | "hosted_gate"
  | "server_capability"
  | "provider_config"
  | "model_catalog"
  | "voice_catalog"
  | "recording_source"
  | "project_state"
  | "generation_state"

export const AUDIO_ROUTE_FINDINGS = [
  "F2 support",
  "F9 support",
  "F15 support",
  "F18 support",
  "F19 support"
] as const

export type AudioRouteFinding = (typeof AUDIO_ROUTE_FINDINGS)[number]
export type AudioRouteCopyField =
  | "label"
  | "primaryJob"
  | "primaryActionLabel"
export type AudioRouteCopyKey =
  `routes.audio.${AudioRouteConcept}.${AudioRouteCopyField}`

export type AudioRouteCopyText = {
  key: AudioRouteCopyKey
  fallback: string
}

export type AudioRouteCopy = {
  label: AudioRouteCopyText
  primaryJob: AudioRouteCopyText
  primaryActionLabel: AudioRouteCopyText
}

export type AudioRouteJob = {
  route: "/audio" | "/speech" | "/stt" | "/tts" | "/audiobook-studio"
  concept: AudioRouteConcept
  copy: AudioRouteCopy
  routeOwner: AudioRouteOwner
  canonicalComponent: string
  capabilities: AudioRouteCapability[]
  routeStatePolicy: "alias" | "ready_or_recoverable" | "beta_ready_or_recoverable"
  findings: AudioRouteFinding[]
}

const audioRouteCopy = (
  concept: AudioRouteConcept,
  labelFallback: string,
  primaryJobFallback: string,
  primaryActionLabelFallback: string
): AudioRouteCopy => ({
  label: {
    key: `routes.audio.${concept}.label`,
    fallback: labelFallback
  },
  primaryJob: {
    key: `routes.audio.${concept}.primaryJob`,
    fallback: primaryJobFallback
  },
  primaryActionLabel: {
    key: `routes.audio.${concept}.primaryActionLabel`,
    fallback: primaryActionLabelFallback
  }
})

export const AUDIO_ROUTE_JOBS: AudioRouteJob[] = [
  {
    route: "/audio",
    concept: "audio_alias",
    copy: audioRouteCopy(
      "audio_alias",
      "Audio",
      "Open the canonical combined speech route from old links and bookmarks.",
      "Open Speech"
    ),
    routeOwner: "shared_alias",
    canonicalComponent: "RouteAliasNavigate:/speech",
    capabilities: [],
    routeStatePolicy: "alias",
    findings: ["F2 support", "F18 support", "F19 support"]
  },
  {
    route: "/speech",
    concept: "speech_combined",
    copy: audioRouteCopy(
      "speech_combined",
      "Speech",
      "Record, transcribe, edit, and synthesize audio in one workspace.",
      "Start audio workflow"
    ),
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
    copy: audioRouteCopy(
      "stt",
      "Speech to Text",
      "Transcribe audio and compare transcription results.",
      "Start transcription"
    ),
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
    copy: audioRouteCopy(
      "tts",
      "Text to Speech",
      "Generate audio from text with provider, voice, and model controls.",
      "Generate speech"
    ),
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
    copy: audioRouteCopy(
      "audiobook",
      "Audiobook Studio",
      "Create long-form audiobook projects from text and generated speech.",
      "Create project"
    ),
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
