Prompts Folder
===============

Purpose
- Centralize module prompts for easy modification without editing Python code.
- Each module has its own prompt file. Update these to change system or helper prompts.

Conventions
- Use simple sections with headings and fenced code blocks.
- Keep prompts concise and deterministic when possible.
- Files are plaintext Markdown for easy reading and editing.

Modules and Files
- embeddings.prompts.md: Contextualization and embedding-related prompts (situate context, outline, etc.)
- rag.prompts.md: Retrieval and reranking prompts.
- chunking.prompts.md: Structure-aware headers, summarization during chunking, proposition extraction templates.
- audio.prompts.md: Transcription/analysis helper prompts.
- chat.prompts.md: Chat assistant system prompts and tools.
- evals.prompts.md: Evaluation tasks and rubric prompts.
- ingestion.prompts.md: Ingestion-time analysis prompts (e.g., claims).
- mcp.prompts.md: MCP tools/system prompts.
- slides.prompts.md: Standalone HTML presentation generation contract.
- summarization.prompts.yaml: Summarization system prompt defaults.

Example Loader (future)
- The codebase currently embeds some prompts directly. A future iteration can load these files at startup and pass them into modules.
- Suggested format: YAML/TOML or Markdown with headings. For now, Markdown is used.

Editing Tips
- Keep a changelog at the bottom of each file.
- If you need environment-specific variants, create files like embeddings.prompts.local.md and reference them in config.
- Standalone Slides loads `slides.standalone_html_system` strictly with a 128 KiB ceiling. Set
  `TLDW_PROMPT_FILE_SLIDES__STANDALONE_HTML_SYSTEM` to a reviewed regular UTF-8 file for an
  override. A configured blank, missing, unreadable, malformed, or oversized override disables
  standalone generation; it never falls back to the packaged contract.
