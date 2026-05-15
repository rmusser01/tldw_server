# Page Review Checklist (WebUI + Extension)

## Standard checks (apply to every page)
- Page loads without blank screen or infinite spinner.
- Root container renders (`#__next` for WebUI, `#root` for extension).
- Header/navigation visible and aligned.
- No error boundary UI or "Something went wrong" message.
- No uncaught console errors (note any warnings).
- Primary action/interaction visible (button, input, or main CTA).
- Empty/error states render cleanly when no data is available.

## WebUI pages (tldw-frontend)

Chat
- /chat
- /chat/agent
- /chat/settings

Media
- /media
- /media-multi
- /media-trash
- /media/123/view (redirect)

Settings
- /settings
- /settings/tldw
- /settings/model
- /settings/chat
- /settings/prompt
- /settings/knowledge
- /settings/rag
- /settings/speech
- /settings/evaluations
- /settings/chatbooks
- /settings/characters
- /settings/world-books
- /settings/chat-dictionaries
- /settings/health
- /settings/processed
- /settings/about
- /settings/share
- /settings/quick-ingest
- /settings/prompt-studio

Admin
- /admin
- /admin/server
- /admin/llamacpp
- /admin/mlx
- /admin/orgs
- /admin/data-ops
- /admin/watchlists-items
- /admin/watchlists-runs
- /admin/maintenance

Workspace / tools
- /flashcards
- /quiz
- /moderation-playground
- /kanban
- /data-tables
- /content-review
- /claims-review
- /watchlists
- /chatbooks
- /notes
- /collections
- /evaluations
- /search
- /review
- /reading
- /items
- /chunking-playground

Knowledge
- /knowledge
- /world-books
- /dictionaries
- /characters
- /prompts
- /prompt-studio

Audio
- /tts
- /stt
- /speech
- /audio

Connectors
- /connectors
- /connectors/browse
- /connectors/jobs
- /connectors/sources

Other
- /
- /login
- /config
- /documentation
- /profile
- /privileges
- /quick-chat-popout
- /onboarding-test
- /for/journalists
- /for/osint
- /for/researchers
- /__debug__/authz.spec
- /__debug__/sidepanel-error-boundary

## Extension options routes (options.html#/path)
- #/
- #/onboarding-test
- #/settings
- #/settings/tldw
- #/settings/model
- #/settings/prompt
- #/settings/evaluations
- #/settings/chat
- #/settings/ui
- #/settings/quick-ingest
- #/settings/speech
- #/settings/image-generation
- #/settings/share
- #/settings/processed
- #/settings/health
- #/settings/prompt-studio
- #/settings/knowledge
- #/settings/chatbooks
- #/settings/characters
- #/settings/world-books
- #/settings/chat-dictionaries
- #/settings/rag
- #/settings/about
- #/chunking-playground
- #/documentation
- #/review
- #/flashcards
- #/quiz
- #/writing-playground
- #/model-playground
- #/chatbooks
- #/watchlists
- #/kanban
- #/data-tables
- #/collections
- #/media
- #/media-trash
- #/media-multi
- #/content-review
- #/notes
- #/knowledge
- #/world-books
- #/dictionaries
- #/characters
- #/prompts
- #/prompt-studio
- #/tts
- #/stt
- #/speech
- #/evaluations
- #/audiobook-studio
- #/workflow-editor
- #/research-studio
- #/moderation-playground
- #/admin/server
- #/admin/llamacpp
- #/admin/mlx
- #/quick-chat-popout

## Extension sidepanel routes (sidepanel.html#/path)
- #/
- #/agent
- #/settings
- #/error-boundary-test

## Automation
WebUI (all pages)
- `cd tldw-frontend && bun run e2e:smoke -- --workers=1`

Extension (options + sidepanel sweep)
- `cd extension && bun run test:e2e -- tests/e2e/page-review.spec.ts`
- Optional artifacts: `TLDW_PAGE_REVIEW_CAPTURE=1` to save screenshots to `playwright-mcp-artifacts/extension-page-review`.
- Strict console errors: `TLDW_PAGE_REVIEW_STRICT=1`.
