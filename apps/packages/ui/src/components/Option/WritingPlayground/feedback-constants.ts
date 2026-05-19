import type { Mood } from "./hooks/useWritingFeedback"

export const MOOD_COLORS: Record<Exclude<Mood, null>, string> = {
  tense: "#ff4d4f",
  romantic: "#ff85c0",
  melancholic: "#597ef7",
  action: "#fa8c16",
  calm: "#52c41a",
  mysterious: "#722ed1",
  humorous: "#fadb14",
}
