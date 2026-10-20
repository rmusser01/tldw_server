import { create } from "zustand"

/**
 * The last send to a model that its provider did not answer (CC-N2, #3107).
 *
 * The tldw server can be healthy while the provider behind a model (Ollama,
 * a local llama.cpp, a hosted API) is down. A send that comes back as a
 * provider failure is noted here for the rest of the session, so the model
 * chip can say "Unreachable" instead of "Healthy"; the next reply from that
 * model clears it. Nothing is persisted.
 */
export type ModelProviderFailureNote = {
  /** Provider id, when the selected model names one. */
  provider?: string
  /** Model id the send went to. */
  model?: string
  /** When the failure was seen (ms since epoch). */
  at: number
}

type ModelProviderFailureState = {
  failures: Record<string, ModelProviderFailureNote>
}

const keyOf = (selectedModel: string | null | undefined) =>
  typeof selectedModel === "string" ? selectedModel.trim() : ""

export const useModelProviderFailureStore = create<ModelProviderFailureState>(
  () => ({ failures: {} })
)

export const recordModelProviderFailure = (
  selectedModel: string | null | undefined,
  note: Omit<ModelProviderFailureNote, "at">
) => {
  const key = keyOf(selectedModel)
  if (!key) return
  useModelProviderFailureStore.setState((state) => ({
    failures: { ...state.failures, [key]: { ...note, at: Date.now() } }
  }))
}

export const clearModelProviderFailure = (selectedModel: string | null | undefined) => {
  const key = keyOf(selectedModel)
  if (!key || !useModelProviderFailureStore.getState().failures[key]) return
  useModelProviderFailureStore.setState((state) => {
    const { [key]: _cleared, ...failures } = state.failures
    return { failures }
  })
}

export const resetModelProviderFailures = () => {
  useModelProviderFailureStore.setState({ failures: {} })
}

/** The provider failure last seen for this model in this session, if any. */
export const useModelProviderFailure = (selectedModel: string | null | undefined) => {
  const key = keyOf(selectedModel)
  return useModelProviderFailureStore((state) =>
    key ? state.failures[key] : undefined
  )
}
