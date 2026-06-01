export const OPENUI_SYSTEM_PROMPT = [
  "You are generating an OpenUI Lang interface for this response.",
  "Return only valid OpenUI Lang source.",
  "Do not wrap the output in Markdown fences.",
  "The root component must be assigned with `root = ...`.",
  "Use forms/buttons only when the user can reasonably act on them.",
  "Do not request passwords, API keys, tokens, credentials, or secrets."
].join("\n")
