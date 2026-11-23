/**
 * The first `end` UTF-16 code units of `text`, one fewer when the cut would
 * leave the high half of a surrogate pair at the end (an emoji, say).
 */
export const sliceCodeUnitsSafely = (text: string, end: number): string => {
  if (end >= text.length) return text
  const cut = Math.max(0, end)
  const last = text.charCodeAt(cut - 1)
  return text.slice(0, last >= 0xd800 && last <= 0xdbff ? cut - 1 : cut)
}
