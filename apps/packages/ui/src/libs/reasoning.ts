const tags = ["think", "reason", "reasoning", "thought"];
type ReasoningPart = {
  type: "reasoning" | "text";
  content: string;
  reasoning_running?: boolean;
  range?: { start: number; end: number };
};

// Reserve distinct escapes so typed reasoning can include literal tags and Markdown code.
export function encodeReasoningText(text: string): string {
  return text.replaceAll("&", "&#x000026;").replaceAll("<", "&#x00003c;");
}

export function decodeReasoningText(text: string): string {
  return text.replace(/&#x0000(26|3c);/g, (_, code) => code === "26" ? "&" : "<");
}

function splitReasoning(text: string, forHistory = false): ReasoningPart[] {
  const result: ReasoningPart[] = [];
  const tagPattern = new RegExp(`<(/?)(${tags.join("|")})>`, "gi");
  const closingTagPattern = new RegExp(`</(${tags.join("|")})>`, "gi");
  let currentIndex = 0;
  const openTags: string[] = [];
  let closedThinking = false;

  for (const match of text.matchAll(tagPattern)) {
    const name = match[2].toLowerCase();
    if (!match[1]) {
      if (!openTags.length) {
        const content = text
          .slice(currentIndex, match.index)
          .replace(closingTagPattern, "");
        if (content) result.push({ type: "text", content });
        currentIndex = match.index! + match[0].length;
      }
      openTags.push(name);
    } else if (!openTags.length && forHistory && closedThinking) {
      // An old trace may have ended at a literal, unescaped closing tag.
      currentIndex = match.index! + match[0].length;
    } else if (openTags.length && (!forHistory || openTags.at(-1) === name)) {
      openTags.pop();
      if (!openTags.length) {
        closedThinking = true;
        const content = text.slice(currentIndex, match.index);
        if (content) result.push({
          type: "reasoning",
          content,
          range: { start: currentIndex, end: match.index },
        });
        currentIndex = match.index! + match[0].length;
      }
    }
  }

  if (currentIndex < text.length) {
    result.push({
      type: openTags.length > 0 ? "reasoning" : "text",
      content:
        openTags.length > 0
          ? text.slice(currentIndex)
          : text.slice(currentIndex).replace(closingTagPattern, ""),
      reasoning_running: openTags.length > 0,
      ...(openTags.length > 0 && { range: { start: currentIndex, end: text.length } }),
    });
  }
  return result;
}

export function parseReasoning(text: string): ReasoningPart[] {
  try {
    return splitReasoning(text)
      .map((part) => ({
        ...part,
        content: (part.type === "reasoning" ? decodeReasoningText(part.content) : part.content).trim(),
      }))
      .filter((part) => part.content || part.reasoning_running !== undefined);
  } catch (e) {
    console.error(`Error parsing reasoning: ${e}`);
    return [{ type: "text", content: text }];
  }
}

export function isEditableReasoningContent(content: string): boolean {
  const [part] = splitReasoning(`<think>${content}</think>`);
  return Boolean(
    content.trim() &&
    !part?.reasoning_running &&
    part?.range?.end === "<think>".length + content.length
  );
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
  return splitReasoning(text, true)
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
