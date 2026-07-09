import {
  createCharacterEmoteStreamParser,
  type CharacterEmoteEvent
} from "@/utils/character-emotes"

type StreamChunkState = {
  fullText: string
  contentToSave: string
  token: string
}

type StreamTextState = {
  fullText: string
  contentToSave: string
}

const replaceTrailingToken = (
  text: string,
  token: string,
  visibleText: string
): string => {
  if (!token) return text
  return `${text.slice(0, Math.max(0, text.length - token.length))}${visibleText}`
}

export const createCharacterEmoteStream = () => {
  const parser = createCharacterEmoteStreamParser()
  const events: CharacterEmoteEvent[] = []

  const remember = (newEvents: CharacterEmoteEvent[]) => {
    if (newEvents.length > 0) events.push(...newEvents)
    return newEvents
  }

  return {
    events,
    sanitizeChunk<T extends StreamChunkState>(chunkState: T) {
      if (!chunkState.token) {
        return { ...chunkState, visibleText: "", emoteEvents: [] }
      }
      const parsed = parser.push(chunkState.token)
      return {
        ...chunkState,
        fullText: replaceTrailingToken(
          chunkState.fullText,
          chunkState.token,
          parsed.visibleText
        ),
        contentToSave: replaceTrailingToken(
          chunkState.contentToSave,
          chunkState.token,
          parsed.visibleText
        ),
        visibleText: parsed.visibleText,
        emoteEvents: remember(parsed.events)
      }
    },
    flush(state: StreamTextState) {
      const parsed = parser.flush()
      if (!parsed.visibleText) {
        return { ...state, visibleText: "", emoteEvents: remember(parsed.events) }
      }
      return {
        fullText: `${state.fullText}${parsed.visibleText}`,
        contentToSave: `${state.contentToSave}${parsed.visibleText}`,
        visibleText: parsed.visibleText,
        emoteEvents: remember(parsed.events)
      }
    }
  }
}
