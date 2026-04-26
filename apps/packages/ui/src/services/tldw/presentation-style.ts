import type {
  PresentationVisualStyleSnapshot,
  VisualStyleRecord
} from "./TldwApiClient"

const cloneVisualStyleValue = (value: unknown): unknown => {
  if (Array.isArray(value)) {
    return value.map((entry) => cloneVisualStyleValue(entry))
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, entryValue]) => [
        key,
        cloneVisualStyleValue(entryValue)
      ])
    )
  }
  return value
}

const cloneVisualStyleObject = (
  value: Record<string, any> | null | undefined
): Record<string, any> =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (cloneVisualStyleValue(value) as Record<string, any>)
    : {}

export const clonePresentationVisualStyleSnapshot = (
  snapshot: PresentationVisualStyleSnapshot | null | undefined
): PresentationVisualStyleSnapshot | null => {
  if (!snapshot) {
    return null
  }
  return {
    id: snapshot.id,
    scope: snapshot.scope,
    name: snapshot.name,
    description: snapshot.description ?? null,
    category: snapshot.category ?? null,
    guide_number: snapshot.guide_number ?? null,
    tags: [...(snapshot.tags || [])],
    best_for: [...(snapshot.best_for || [])],
    generation_rules: cloneVisualStyleObject(snapshot.generation_rules),
    artifact_preferences: [...(snapshot.artifact_preferences || [])],
    appearance_defaults: cloneVisualStyleObject(snapshot.appearance_defaults),
    fallback_policy: cloneVisualStyleObject(snapshot.fallback_policy),
    version: snapshot.version ?? null
  }
}

export const buildPresentationVisualStyleSnapshot = (
  style: Pick<
    VisualStyleRecord,
    | "id"
    | "scope"
    | "name"
    | "description"
    | "category"
    | "guide_number"
    | "tags"
    | "best_for"
    | "generation_rules"
    | "artifact_preferences"
    | "appearance_defaults"
    | "fallback_policy"
    | "version"
  >
): PresentationVisualStyleSnapshot =>
  clonePresentationVisualStyleSnapshot({
    id: style.id,
    scope: style.scope,
    name: style.name,
    description: style.description ?? null,
    category: style.category ?? null,
    guide_number: style.guide_number ?? null,
    tags: style.tags,
    best_for: style.best_for,
    generation_rules: style.generation_rules,
    artifact_preferences: style.artifact_preferences,
    appearance_defaults: style.appearance_defaults,
    fallback_policy: style.fallback_policy,
    version: style.version ?? null
  })!
