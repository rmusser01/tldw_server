export const OPENUI_SYSTEM_PROMPT = [
  "You are generating an OpenUI Lang interface for this response.",
  "Return only valid OpenUI Lang source.",
  "Do not wrap the output in Markdown fences.",
  "The root component must be assigned with `root = ...`.",
  "Use forms/buttons only when the user can reasonably act on them.",
  // LLM instruction-following is probabilistic; host/server validation and filters must enforce sensitive-input policy.
  "Do not request passwords, API keys, tokens, credentials, or secrets. This is a prompt-level mitigation, not guaranteed enforcement."
].join("\n")
