export type OnboardingUatViewport = "desktop" | "mobile"

export type OnboardingUatScenario = {
  id: string
  tier: "A"
  title: string
  description: string
  specFile: string
  mockConfig: string
  viewports: OnboardingUatViewport[]
  grep: string
}

export const onboardingUatSpecFiles = [
  "e2e/onboarding-uat/setup-happy-path.spec.ts",
  "e2e/onboarding-uat/first-source.spec.ts",
  "e2e/onboarding-uat/recovery.spec.ts",
] as const

export const tierAScenarios = [
  {
    id: "hosted-openai-first-chat",
    tier: "A",
    title: "Hosted OpenAI setup to first chat",
    description:
      "Connect a first-run solo user to the backend, complete hosted OpenAI-style setup, and receive a real chat response through the mock OpenAI server.",
    specFile: "e2e/onboarding-uat/setup-happy-path.spec.ts",
    mockConfig: "hosted-success.json",
    viewports: ["desktop", "mobile"],
    grep: "hosted-openai-first-chat",
  },
  {
    id: "local-openai-first-chat",
    tier: "A",
    title: "Local OpenAI-compatible setup to first chat",
    description:
      "Exercise the peer local-provider path against the repo mock OpenAI-compatible server when the current UI exposes that flow.",
    specFile: "e2e/onboarding-uat/setup-happy-path.spec.ts",
    mockConfig: "local-success.json",
    viewports: ["desktop"],
    grep: "local-openai-first-chat",
  },
  {
    id: "first-source-after-chat",
    tier: "A",
    title: "First source after successful first chat",
    description:
      "After first chat succeeds, add the synthetic first source and ask for value from ingested content.",
    specFile: "e2e/onboarding-uat/first-source.spec.ts",
    mockConfig: "hosted-success.json",
    viewports: ["desktop"],
    grep: "first-source-after-chat",
  },
  {
    id: "provider-retry-recovery",
    tier: "A",
    title: "Provider retry recovery",
    description:
      "Exercise inline recovery when the provider returns a transient chat failure, then retry through the same real backend path.",
    specFile: "e2e/onboarding-uat/recovery.spec.ts",
    mockConfig: "chat-fail-once.json",
    viewports: ["desktop"],
    grep: "provider-retry-recovery",
  },
] as const satisfies readonly OnboardingUatScenario[]

export const onboardingUatScenarios = [...tierAScenarios] as const
