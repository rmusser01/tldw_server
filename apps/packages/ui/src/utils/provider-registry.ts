import type { ReactNode } from "react"
import { ChromeIcon, CpuIcon } from "lucide-react"
import { AliBaBaCloudIcon } from "@/components/Icons/AliBaBaCloud"
import { ChutesIcon } from "@/components/Icons/ChutesIcon"
import { DeepSeekIcon } from "@/components/Icons/DeepSeek"
import { FireworksMonoIcon } from "@/components/Icons/Fireworks"
import { GeminiIcon } from "@/components/Icons/GeminiIcon"
import { GroqMonoIcon } from "@/components/Icons/Groq"
import { HuggingFaceIcon } from "@/components/Icons/HuggingFaceIcon"
import { InfinigenceAI } from "@/components/Icons/InfinigenceAI"
import { LLamaFile } from "@/components/Icons/Llamafile"
import { LlamaCppLogo } from "@/components/Icons/LlamacppLogo"
import { LMStudioIcon } from "@/components/Icons/LMStudio"
import { MistarlIcon } from "@/components/Icons/Mistral"
import { MoonshotIcon } from "@/components/Icons/Moonshot"
import { NovitaIcon } from "@/components/Icons/Novita"
import { OllamaIcon } from "@/components/Icons/Ollama"
import { OpenAiIcon } from "@/components/Icons/OpenAI"
import { OpenRouterIcon } from "@/components/Icons/OpenRouter"
import { SiliconFlowIcon } from "@/components/Icons/SiliconFlow"
import { TencentCloudIcon } from "@/components/Icons/TencentCloud"
import { TogtherMonoIcon } from "@/components/Icons/Togther"
import { VercelIcon } from "@/components/Icons/VercelIcon"
import { VllmLogo } from "@/components/Icons/VllmLogo"
import { VolcEngineIcon } from "@/components/Icons/VolcEngine"
import { XAIIcon } from "@/components/Icons/XAI"

export type ProviderIconKey =
  | "tldw"
  | "chrome"
  | "custom"
  | "fireworks"
  | "groq"
  | "lmstudio"
  | "openai"
  | "together"
  | "openrouter"
  | "llamafile"
  | "gemini"
  | "mistral"
  | "deepseek"
  | "siliconflow"
  | "volcengine"
  | "tencentcloud"
  | "alibabacloud"
  | "llamacpp"
  | "infinitenceai"
  | "novita"
  | "vllm"
  | "moonshot"
  | "xai"
  | "huggingface"
  | "vercel"
  | "chutes"
  | "ollama"
  | "default"

export type ProviderIconComponent = (props: {
  className?: string
}) => ReactNode

export type ProviderCapability = "llm" | "tts" | "tts-engine"

export type ProviderMeta = {
  label: string
  description?: string
  baseUrl?: string
  iconKey?: ProviderIconKey
  order?: number
  capabilities?: ProviderCapability[]
  ttsLabel?: string
}

export const PROVIDER_REGISTRY: Record<string, ProviderMeta> = {
  browser: {
    label: "Browser",
    description: "Uses your system's built-in speech synthesis",
    ttsLabel: "Browser TTS (no setup needed)",
    order: 1,
    capabilities: ["tts"]
  },
  elevenlabs: {
    label: "ElevenLabs",
    description: "ElevenLabs cloud TTS (requires API key)",
    order: 2,
    capabilities: ["tts"]
  },
  custom: {
    label: "Custom",
    baseUrl: "",
    iconKey: "custom",
    order: 1
  },
  llamacpp: {
    label: "LLaMa.cpp",
    baseUrl: "http://localhost:8080/v1",
    iconKey: "llamacpp",
    order: 2
  },
  lmstudio: {
    label: "LM Studio",
    baseUrl: "http://localhost:1234/v1",
    iconKey: "lmstudio",
    order: 3
  },
  llamafile: {
    label: "Llamafile",
    baseUrl: "http://127.0.0.1:8080/v1",
    iconKey: "llamafile",
    order: 4
  },
  openai: {
    label: "OpenAI",
    description: "OpenAI cloud TTS (requires API key)",
    baseUrl: "https://api.openai.com/v1",
    iconKey: "openai",
    order: 5,
    capabilities: ["llm", "tts"],
    ttsLabel: "OpenAI TTS"
  },
  deepseek: {
    label: "DeepSeek",
    baseUrl: "https://api.deepseek.com",
    iconKey: "deepseek",
    order: 6
  },
  fireworks: {
    label: "Fireworks",
    baseUrl: "https://api.fireworks.ai/inference/v1",
    iconKey: "fireworks",
    order: 7
  },
  novita: {
    label: "Novita AI",
    baseUrl: "https://api.novita.ai/v3/openai",
    iconKey: "novita",
    order: 8
  },
  huggingface: {
    label: "Hugging Face",
    baseUrl: "https://router.huggingface.co/v1",
    iconKey: "huggingface",
    order: 9
  },
  groq: {
    label: "Groq",
    baseUrl: "https://api.groq.com/openai/v1",
    iconKey: "groq",
    order: 10
  },
  together: {
    label: "Together",
    baseUrl: "https://api.together.xyz/v1",
    iconKey: "together",
    order: 11
  },
  openrouter: {
    label: "OpenRouter",
    baseUrl: "https://openrouter.ai/api/v1",
    iconKey: "openrouter",
    order: 12
  },
  gemini: {
    label: "Google AI",
    baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai",
    iconKey: "gemini",
    order: 13
  },
  mistral: {
    label: "Mistral",
    baseUrl: "https://api.mistral.ai/v1",
    iconKey: "mistral",
    order: 14
  },
  infinitenceai: {
    label: "Infinigence AI",
    baseUrl: "https://cloud.infini-ai.com/maas/v1",
    iconKey: "infinitenceai",
    order: 15
  },
  infinigenceai: {
    label: "Infinigence AI",
    iconKey: "infinitenceai"
  },
  siliconflow: {
    label: "SiliconFlow",
    baseUrl: "https://api.siliconflow.cn/v1",
    iconKey: "siliconflow",
    order: 16
  },
  volcengine: {
    label: "VolcEngine",
    baseUrl: "https://ark.cn-beijing.volces.com/api/v3",
    iconKey: "volcengine",
    order: 17
  },
  tencentcloud: {
    label: "TencentCloud",
    baseUrl: "https://api.lkeap.cloud.tencent.com/v1",
    iconKey: "tencentcloud",
    order: 18
  },
  alibabacloud: {
    label: "AliBaBaCloud",
    baseUrl: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    iconKey: "alibabacloud",
    order: 19
  },
  vllm: {
    label: "vLLM",
    baseUrl: "http://localhost:8000/v1",
    iconKey: "vllm",
    order: 20
  },
  moonshot: {
    label: "Moonshot",
    baseUrl: "https://api.moonshot.ai/v1",
    iconKey: "moonshot",
    order: 21
  },
  xai: {
    label: "xAI",
    baseUrl: "https://api.x.ai/v1",
    iconKey: "xai",
    order: 22
  },
  vercel: {
    label: "Vercel AI Gateway",
    baseUrl: "https://ai-gateway.vercel.sh/v1",
    iconKey: "vercel",
    order: 23
  },
  chutes: {
    label: "Chutes",
    baseUrl: "https://llm.chutes.ai/v1",
    iconKey: "chutes",
    order: 24
  },
  tldw: {
    label: "tldw Server",
    description: "Local TTS engines on your tldw server",
    iconKey: "tldw",
    order: 6,
    capabilities: ["llm", "tts"],
    ttsLabel: "tldw Server (Local)"
  },
  chrome: {
    label: "Chrome",
    iconKey: "chrome"
  },
  anthropic: {
    label: "Anthropic"
  },
  google: {
    label: "Google"
  },
  meta: {
    label: "Meta"
  },
  cohere: {
    label: "Cohere"
  },
  ollama: {
    label: "Ollama",
    iconKey: "ollama"
  },
  llama: {
    label: "LLaMa.cpp",
    iconKey: "llamacpp"
  },
  kobold: {
    label: "Kobold.cpp"
  },
  ooba: {
    label: "Oobabooga"
  },
  tabby: {
    label: "TabbyAPI"
  },
  aphrodite: {
    label: "Aphrodite"
  },
  zai: {
    label: "Z.AI"
  },
  custom_openai_api: {
    label: "Custom OpenAI API"
  },
  kokoro: {
    label: "Kokoro",
    capabilities: ["tts-engine"]
  },
  kitten_tts: {
    label: "KittenTTS",
    capabilities: ["tts-engine"]
  },
  higgs: {
    label: "Higgs",
    capabilities: ["tts-engine"]
  },
  dia: {
    label: "Dia",
    capabilities: ["tts-engine"]
  },
  chatterbox: {
    label: "Chatterbox",
    capabilities: ["tts-engine"]
  },
  vibevoice: {
    label: "VibeVoice",
    capabilities: ["tts-engine"]
  },
  neutts: {
    label: "NeuTTS",
    capabilities: ["tts-engine"]
  },
  pockettts: {
    label: "PocketTTS",
    capabilities: ["tts-engine"]
  },
  pocket_tts: {
    label: "PocketTTS",
    capabilities: ["tts-engine"]
  },
  index_tts: {
    label: "IndexTTS",
    capabilities: ["tts-engine"]
  },
  index_tts2: {
    label: "IndexTTS2",
    capabilities: ["tts-engine"]
  },
  lux_tts: {
    label: "LuxTTS",
    capabilities: ["tts-engine"]
  },
  qwen3_tts: {
    label: "Qwen3-TTS",
    capabilities: ["tts-engine"]
  },
  vibevoice_realtime: {
    label: "VibeVoice Realtime",
    capabilities: ["tts-engine"]
  },
  supertonic: {
    label: "Supertonic",
    capabilities: ["tts-engine"]
  },
  supertonic2: {
    label: "Supertonic2",
    capabilities: ["tts-engine"]
  },
  echo_tts: {
    label: "Echo TTS",
    capabilities: ["tts-engine"]
  },
  unknown: {
    label: "Unknown"
  }
}

export const PROVIDER_ICON_COMPONENTS: Record<
  ProviderIconKey,
  ProviderIconComponent
> = {
  tldw: CpuIcon,
  chrome: ChromeIcon,
  custom: CpuIcon,
  fireworks: FireworksMonoIcon,
  groq: GroqMonoIcon,
  lmstudio: LMStudioIcon,
  openai: OpenAiIcon,
  together: TogtherMonoIcon,
  openrouter: OpenRouterIcon,
  llamafile: LLamaFile,
  gemini: GeminiIcon,
  mistral: MistarlIcon,
  deepseek: DeepSeekIcon,
  siliconflow: SiliconFlowIcon,
  volcengine: VolcEngineIcon,
  tencentcloud: TencentCloudIcon,
  alibabacloud: AliBaBaCloudIcon,
  llamacpp: LlamaCppLogo,
  infinitenceai: InfinigenceAI,
  novita: NovitaIcon,
  vllm: VllmLogo,
  moonshot: MoonshotIcon,
  xai: XAIIcon,
  huggingface: HuggingFaceIcon,
  vercel: VercelIcon,
  chutes: ChutesIcon,
  ollama: OllamaIcon,
  default: CpuIcon
}

export const normalizeProviderKey = (provider?: string): string => {
  const normalized = String(provider || "unknown").toLowerCase()
  if (normalized === "kittentts" || normalized === "kitten-tts") {
    return "kitten_tts"
  }
  return normalized
}

const hasCapability = (
  meta: ProviderMeta,
  capability: ProviderCapability
): boolean => {
  if (Array.isArray(meta.capabilities) && meta.capabilities.length > 0) {
    return meta.capabilities.includes(capability)
  }
  return capability === "llm"
}

export const getProvidersByCapability = (capability: ProviderCapability) => {
  return Object.entries(PROVIDER_REGISTRY)
    .filter(([, meta]) => hasCapability(meta, capability))
    .sort(([, a], [, b]) => (a.order ?? 999) - (b.order ?? 999))
    .map(([key, meta]) => ({ key, meta }))
}

export const getProviderMeta = (provider?: string): ProviderMeta | null => {
  const key = normalizeProviderKey(provider)
  return PROVIDER_REGISTRY[key] ?? null
}

export const getProviderLabel = (
  provider?: string,
  capability?: ProviderCapability
): string => {
  const meta = getProviderMeta(provider)
  if (capability === "tts") {
    return meta?.ttsLabel || meta?.label || provider || "TTS"
  }
  return meta?.label || provider || "API"
}

export const getProviderDisplayName = (provider?: string): string => {
  return getProviderMeta(provider)?.label || provider || "API"
}

export const getProviderIconKey = (provider?: string): ProviderIconKey => {
  const meta = getProviderMeta(provider)
  return meta?.iconKey ?? "default"
}

export const getProviderIconComponent = (
  provider?: string
): ProviderIconComponent => {
  const iconKey = getProviderIconKey(provider)
  return PROVIDER_ICON_COMPONENTS[iconKey] ?? PROVIDER_ICON_COMPONENTS.default
}

export type ProviderInferenceDomain = "llm" | "tts"

type ProviderInferenceRule = {
  provider: string
  match: (value: string) => boolean
}

const LLM_PROVIDER_RULES: ProviderInferenceRule[] = [
  { provider: "openai", match: (value) => value.includes("gpt") },
  { provider: "anthropic", match: (value) => value.includes("claude") },
  { provider: "meta", match: (value) => value.includes("llama") },
  { provider: "google", match: (value) => value.includes("gemini") },
  { provider: "mistral", match: (value) => value.includes("mistral") }
]

const TTS_PROVIDER_RULES: ProviderInferenceRule[] = [
  {
    provider: "openai",
    match: (value) =>
      value === "tts-1" || value === "tts-1-hd" || value.startsWith("gpt-")
  },
  {
    provider: "kitten_tts",
    match: (value) =>
      value.startsWith("kitten_tts") ||
      value.startsWith("kitten-tts") ||
      value.startsWith("kittentts") ||
      value.startsWith("kittenml/kitten-tts")
  },
  { provider: "kokoro", match: (value) => value.startsWith("kokoro") },
  { provider: "higgs", match: (value) => value.startsWith("higgs") },
  { provider: "dia", match: (value) => value.startsWith("dia") },
  { provider: "chatterbox", match: (value) => value.startsWith("chatterbox") },
  { provider: "vibevoice", match: (value) => value.startsWith("vibevoice") },
  {
    provider: "vibevoice_realtime",
    match: (value) => value.startsWith("vibevoice_realtime") || value.startsWith("vibevoice-realtime")
  },
  { provider: "neutts", match: (value) => value.startsWith("neutts") },
  {
    provider: "pockettts",
    match: (value) =>
      value.startsWith("pockettts") ||
      value.startsWith("pocket-tts") ||
      value.startsWith("pocket_tts")
  },
  { provider: "qwen3_tts", match: (value) => value.startsWith("qwen3") || value.startsWith("qwen3_tts") },
  { provider: "lux_tts", match: (value) => value.startsWith("lux_tts") || value.startsWith("luxtts") },
  { provider: "echo_tts", match: (value) => value.startsWith("echo_tts") || value.startsWith("echo") },
  { provider: "elevenlabs", match: (value) => value.startsWith("eleven") },
  {
    provider: "index_tts",
    match: (value) =>
      value.startsWith("index_tts") || value.startsWith("indextts")
  },
  {
    provider: "index_tts2",
    match: (value) =>
      value.startsWith("index_tts2") || value.startsWith("indextts2")
  },
  {
    provider: "supertonic2",
    match: (value) => value.startsWith("supertonic2") || value.startsWith("supertonic-2")
  },
  {
    provider: "supertonic",
    match: (value) => value.startsWith("supertonic")
  }
]

const inferFromRules = (
  value: string,
  rules: ProviderInferenceRule[]
): string | null => {
  for (const rule of rules) {
    if (rule.match(value)) return rule.provider
  }
  return null
}

export const inferProviderFromModel = (
  model?: string | null,
  domain: ProviderInferenceDomain = "llm"
): string | null => {
  const normalized = String(model || "").trim().toLowerCase()
  if (!normalized) return null
  return domain === "tts"
    ? inferFromRules(normalized, TTS_PROVIDER_RULES)
    : inferFromRules(normalized, LLM_PROVIDER_RULES)
}
