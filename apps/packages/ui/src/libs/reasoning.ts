const tags = ["think", "reason", "reasoning", "thought"];
type ReasoningPart = {
  type: "reasoning" | "text";
  content: string;
  reasoning_running?: boolean;
};

function splitReasoning(text: string): ReasoningPart[] {
  const result: ReasoningPart[] = [];
  const tagPattern = new RegExp(`<(/?)(${tags.join("|")})>`, "gi");
  const closingTagPattern = new RegExp(`</(${tags.join("|")})>`, "gi");
  let currentIndex = 0;
  let depth = 0;

  for (const match of text.matchAll(tagPattern)) {
    if (!match[1]) {
      if (depth === 0) {
        const content = text
          .slice(currentIndex, match.index)
          .replace(closingTagPattern, "");
        if (content) result.push({ type: "text", content });
        currentIndex = match.index! + match[0].length;
      }
      depth += 1;
    } else if (depth > 0) {
      depth -= 1;
      if (depth === 0) {
        const content = text.slice(currentIndex, match.index);
        if (content) result.push({ type: "reasoning", content });
        currentIndex = match.index! + match[0].length;
      }
    }
  }

  if (currentIndex < text.length) {
    result.push({
      type: depth > 0 ? "reasoning" : "text",
      content:
        depth > 0
          ? text.slice(currentIndex)
          : text.slice(currentIndex).replace(closingTagPattern, ""),
      reasoning_running: depth > 0,
    });
  }
  return result;
}

export function parseReasoning(text: string): ReasoningPart[] {
  try {
    return splitReasoning(text)
      .map((part) => ({ ...part, content: part.content.trim() }))
      .filter((part) => part.content || part.reasoning_running !== undefined);
  } catch (e) {
    console.error(`Error parsing reasoning: ${e}`);
    return [{ type: "text", content: text }];
  }
}

export function isReasoningStarted(text: string): boolean {
  const tagPattern = new RegExp(`<(${tags.join("|")})>`, "i")
  return tagPattern.test(text)
}

export function isReasoningEnded(text: string): boolean {
  const closeTagPattern = new RegExp(`</(${tags.join("|")})>`, "i")
  return closeTagPattern.test(text)
}

export function removeReasoning(text: string): string {
  return splitReasoning(text)
    .filter((part) => part.type === "text")
    .map((part) => part.content)
    .join("")
    .trim();
}

export function mergeReasoningContent(
  originalText: string,
  reasoning: string
): string {
  const reasoningTag = "<think>"

  originalText = originalText.replace(reasoningTag, "")

  return `${reasoningTag}${originalText + reasoning}`
}

export function replaceThinkTagToEM(text: string): string {
  const tagPattern = new RegExp(`<\\/?(${tags.join("|")})>`, "gi");
  const emStyle = "font-style: italic; font-size: 0.9em; margin-bottom: 1em;";
  return splitReasoning(text)
    .map((part) =>
      part.type === "reasoning"
        ? `<em style="${emStyle}">${part.content.replace(tagPattern, "")}</em>\n\n`
        : part.content,
    )
    .join("")
    .replaceAll("<think>", "")
    .replaceAll("</think>", "");
}

export const MISSING_FINAL_ANSWER_MESSAGE =
  "No final answer was generated. Retry or continue to request an answer."

/** A settled reasoning trace without any user-facing answer text. */
export function isReasoningOnlyResponse(text: string): boolean {
  const parts = parseReasoning(text);
  return (
    (isReasoningStarted(text) || isReasoningEnded(text)) &&
    !parts.some(
      (part) => part.type === "text" && part.content.trim().length > 0,
    )
  );
}
