import type {
  PersonaAmbientMode,
  PersonaVisualStateId
} from "@/types/persona-visuals"

export type ResolveEffectiveAmbientModeInput = {
  persona?: PersonaAmbientMode | null
  global?: PersonaAmbientMode | null
  readFailed?: boolean
  surface?: string
}

export const resolveEffectiveAmbientMode = ({
  persona = null,
  global = null,
  readFailed = false,
  surface = "web"
}: ResolveEffectiveAmbientModeInput): PersonaAmbientMode => {
  if (readFailed) return "off"
  const mode = persona ?? global ?? "expressive"
  return mode === "roaming" && surface !== "web" ? "expressive" : mode
}

export type ResolveWinningPersonaVisualIntentInput = {
  error?: PersonaVisualStateId | null
  approval?: PersonaVisualStateId | null
  offline?: PersonaVisualStateId | null
  wake?: PersonaVisualStateId | null
  listening?: PersonaVisualStateId | null
  thinking?: PersonaVisualStateId | null
  speaking?: PersonaVisualStateId | null
  tool?: PersonaVisualStateId | null
  interaction?: PersonaVisualStateId | null
  ambient?: PersonaVisualStateId | null
  idle?: PersonaVisualStateId | null
}

export const resolveWinningPersonaVisualIntent = (
  input: ResolveWinningPersonaVisualIntentInput
): PersonaVisualStateId =>
  input.error ??
  input.approval ??
  input.offline ??
  input.wake ??
  input.listening ??
  input.thinking ??
  input.speaking ??
  input.tool ??
  input.interaction ??
  input.ambient ??
  input.idle ??
  "idle"
