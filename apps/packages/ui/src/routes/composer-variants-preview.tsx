import React from "react"
import { TerminalStackV1 } from "@/components/Chat/composer/variants/TerminalStackV1"
import { SplitBriefV3 } from "@/components/Chat/composer/variants/SplitBriefV3"
import { RadialCommandV5 } from "@/components/Chat/composer/variants/RadialCommandV5"
import {
  ChatComposer,
} from "@/components/Chat/composer/ChatComposer"
import type { ChatComposerVariant } from "@/components/Chat/composer/types"

/**
 * Dev-only preview route for the Primer composer redesign variants.
 * Mount via `/composer-variants-preview` in the Next.js frontend.
 *
 * Each variant is rendered twice:
 *   - Desktop density (normal `/chat` context)
 *   - Compact density (~360px extension sidepanel context)
 *
 * State is mocked with `useState` — no wiring into Playground/Sidepanel
 * yet. This harness is the visual review target while we iterate on the
 * three variants before they land in production.
 */

const V1_DEMO_TOP_CHIPS = [
  { id: "add-source", label: "+ source" },
  { id: "web", label: "☼ Web search", active: true },
  { id: "mcp", label: "⌘ MCP · 3" },
  { id: "persona", label: "✦ Persona · Dr. Hoffman" },
  { id: "rag", label: "RAG", variant: "accent" as const },
]

const V1_DEMO_BOTTOM_CHIPS = [
  { id: "model", label: "haiku-4-5" },
  { id: "temp", label: "⌁ temp 0.7" },
  { id: "ocr", label: "◐ OCR" },
]

const V1_DEMO_ICON_BUTTONS = [
  { id: "attach", label: "Attach file", icon: "⎙" },
  { id: "voice", label: "Voice", icon: "◉" },
  { id: "prompts", label: "Prompt library", icon: "✿" },
  { id: "slash", label: "Slash commands", icon: "/" },
]

const SP_DEMO_TOP_CHIPS = [
  { id: "web", label: "☼ Web", active: true },
  { id: "model", label: "haiku-4-5" },
  { id: "temp", label: "⌁ 0.7" },
]

const SP_DEMO_ICON_BUTTONS = [
  { id: "attach", label: "Attach", icon: "⎙" },
  { id: "voice", label: "Voice", icon: "◉" },
  { id: "prompts", label: "Prompts", icon: "✿" },
  { id: "slash", label: "Slash", icon: "/" },
]

const V3_BRIEF_SECTIONS = [
  {
    id: "brief",
    label: "Brief",
    fields: [
      { id: "src", fieldKey: "src", value: "▣ irb-archive · 14", active: true },
      { id: "mdl", fieldKey: "mdl", value: "haiku-4-5" },
      { id: "tmp", fieldKey: "tmp", value: "0.7 · balanced" },
      { id: "per", fieldKey: "per", value: "Dr. Hoffman", active: true },
      { id: "mcp", fieldKey: "mcp", value: "fs · web · git" },
      { id: "web", fieldKey: "web", value: "on · strict", active: true },
    ],
  },
  {
    id: "prompts",
    label: "Prompts",
    fields: [
      {
        id: "slash-prompts",
        fieldKey: "✿",
        value: "/cite · /table · /critique",
      },
    ],
  },
]

const V3_SP_BRIEF_SECTIONS = [
  {
    id: "brief",
    fields: [
      { id: "src", value: "▣ 14", active: true },
      { id: "mdl", value: "haiku-4-5" },
      { id: "tmp", value: "0.7" },
      { id: "per", value: "✦ Hoffman", active: true },
      { id: "web", value: "☼ web", active: true },
      { id: "mcp", value: "⌘ 3" },
    ],
  },
]

const V3_DEMO_ICON_BUTTONS = [
  { id: "attach", label: "Attach", icon: "⎙" },
  { id: "voice", label: "Voice", icon: "◉" },
  { id: "slash", label: "Slash", icon: "/" },
  { id: "mention", label: "Mention", icon: "@" },
]

const V3_SP_ICON_BUTTONS = [
  { id: "attach", label: "Attach", icon: "⎙" },
  { id: "voice", label: "Voice", icon: "◉" },
  { id: "slash", label: "Slash", icon: "/" },
]

const V5_FACETS = [
  { id: "src", fieldKey: "src", value: "▣ irb-archive · 14", active: true },
  { id: "mdl", fieldKey: "mdl", value: "haiku-4-5" },
  { id: "tmp", fieldKey: "tmp", value: "0.7" },
  { id: "per", fieldKey: "per", value: "Hoffman", active: true },
  { id: "web", fieldKey: "web", value: "on", active: true },
  { id: "mcp", fieldKey: "mcp", value: "3 tools" },
]

const V5_SP_FACETS = [
  { id: "src", value: "▣ 14", active: true },
  { id: "mdl", value: "haiku-4-5" },
  { id: "web", value: "☼", active: true },
]

const V5_ICON_BUTTONS = [
  { id: "attach", label: "Attach", icon: "⎙" },
  { id: "voice", label: "Voice", icon: "◉" },
]

const V5_SP_ICON_BUTTONS = [
  { id: "attach", label: "Attach", icon: "⎙" },
]

const V5_DEMO_PALETTE_GROUPS = [
  {
    id: "models",
    label: "Models · 4 results",
    rows: [
      {
        id: "haiku",
        icon: "☀",
        command: "/model haiku-4-5",
        hint: "current · Anthropic · 200k ctx · local proxy",
        kbd: "↩",
      },
      {
        id: "opus",
        icon: "♦",
        command: "/model opus-4-1",
        hint: "deep reasoning · 1M ctx · remote",
      },
      {
        id: "gpt5",
        icon: "◉",
        command: "/model gpt-5",
        hint: "OpenAI · via gateway",
      },
      {
        id: "llama",
        icon: "●",
        command: "/model llama-3.3-70B",
        hint: "self-hosted · ollama · air-gap OK",
      },
    ],
  },
  {
    id: "related",
    label: "Related commands",
    rows: [
      {
        id: "temp",
        icon: "⌁",
        command: "/temp 0.7",
        hint: "temperature · balanced",
      },
      {
        id: "persona",
        icon: "✦",
        command: "/persona Hoffman",
        hint: "switch voice · 12 personas",
      },
    ],
  },
]

const DEFAULT_DESKTOP_PROMPT =
  "Show me the strongest counter-argument from the Anthropic hosted-inference thread, with the exact passage and page number."
const DEFAULT_COMPACT_PROMPT = "Strongest counter-argument?"
const V3_DESKTOP_PROMPT =
  "Show me the strongest counter-argument from the Anthropic hosted-inference thread, with the exact passage and page number. Compare it to the NIST SP 800-218A position on model egress."

const Frame: React.FC<{
  label: string
  width?: number
  children: React.ReactNode
}> = ({ label, width, children }) => (
  <div
    className="border border-border rounded-md overflow-hidden bg-bg shadow-md"
    style={width ? { width } : undefined}
  >
    <div className="px-3.5 py-2 border-b border-border bg-surface/50 font-mono text-[10px] text-text-subtle uppercase tracking-wider">
      {label}
    </div>
    {children}
  </div>
)

const ComposerVariantsPreview: React.FC = () => {
  const [desktopMessage, setDesktopMessage] = React.useState(
    DEFAULT_DESKTOP_PROMPT
  )
  const [compactMessage, setCompactMessage] = React.useState(
    DEFAULT_COMPACT_PROMPT
  )
  const [v3DesktopMessage, setV3DesktopMessage] = React.useState(
    V3_DESKTOP_PROMPT
  )
  const [v3CompactMessage, setV3CompactMessage] = React.useState(
    DEFAULT_COMPACT_PROMPT
  )
  const [v5DesktopMessage, setV5DesktopMessage] = React.useState("/model ")
  const [v5CompactMessage, setV5CompactMessage] = React.useState("")
  const [v5PaletteOpen, setV5PaletteOpen] = React.useState(true)
  const [v5PaletteIdx, setV5PaletteIdx] = React.useState(0)
  const [sendCount, setSendCount] = React.useState(0)
  const [previewVariant, setPreviewVariant] =
    React.useState<ChatComposerVariant>("v1")
  const [liveMessage, setLiveMessage] = React.useState("")

  const onSend = React.useCallback(() => {
    setSendCount((n) => n + 1)
  }, [])

  return (
    <div className="min-h-screen bg-bg text-text px-10 pt-10 pb-20 max-w-[1640px] mx-auto">
      <header className="flex items-baseline gap-4 mb-2 flex-wrap">
        <span className="font-mono text-[11px] text-primary tracking-wider">
          § Composer · preview
        </span>
        <h1 className="font-display font-semibold text-2xl tracking-tight">
          Primer composer variants
        </h1>
        <span className="ml-auto font-mono text-[11px] text-text-subtle uppercase tracking-wider">
          dev harness · {sendCount} sends
        </span>
      </header>
      <p className="font-serif italic text-text-muted text-base max-w-[780px] leading-relaxed mb-8">
        Visual review target for the Primer composer redesign. State is
        mocked; each variant is rendered at both desktop and extension-
        sidepanel (~360px) widths. Wire-in to <code className="font-mono text-primary">Playground</code>{" "}
        and <code className="font-mono text-primary">sidepanel-chat</code> is
        pending.
      </p>

      {/* Live dispatcher demo */}
      <section className="mb-14">
        <div className="flex items-baseline gap-3.5 px-4 py-3 border-b border-border bg-surface rounded-t-md">
          <span className="font-mono text-[11px] text-primary tracking-wider">
            &lt;ChatComposer&gt;
          </span>
          <span className="font-display font-semibold text-[17px]">
            Live dispatcher
          </span>
          <span className="font-mono text-[10px] px-2 py-0.5 rounded-full border border-primary/40 text-primary uppercase tracking-wider">
            isolated preview state
          </span>
          <span className="font-serif italic text-text-muted text-sm ml-2">
            Pick a variant without mutating the real composer preference
            used by Playground or the sidepanel.
          </span>
          <div className="ml-auto flex items-center gap-1.5">
            {(["v1", "v3", "v5"] as ChatComposerVariant[]).map((v) => {
              const active = v === previewVariant
              return (
                <button
                  key={v}
                  type="button"
                  onClick={() => setPreviewVariant(v)}
                  aria-pressed={active}
                  className={
                    "px-2.5 py-1 rounded-md font-mono text-[11px] border transition-colors " +
                    (active
                      ? "bg-primary text-bg border-primary"
                      : "bg-surface2 text-text-muted border-border hover:text-text")
                  }
                >
                  {v.toUpperCase()}
                </button>
              )
            })}
          </div>
        </div>
        <div className="p-6 border border-t-0 border-border rounded-b-md bg-surface/40">
          <Frame label={`previewVariant=${previewVariant}`}>
            {previewVariant === "v1" && (
              <ChatComposer
                variant="v1"
                message={liveMessage}
                onMessageChange={setLiveMessage}
                onSend={onSend}
                sourceChip={{ count: 14, label: "irb-archive" }}
                topChips={V1_DEMO_TOP_CHIPS}
                bottomChips={V1_DEMO_BOTTOM_CHIPS}
                iconButtons={V1_DEMO_ICON_BUTTONS}
                tokens={{ used: liveMessage.length, max: 8000 }}
              />
            )}
            {previewVariant === "v3" && (
              <ChatComposer
                variant="v3"
                message={liveMessage}
                onMessageChange={setLiveMessage}
                onSend={onSend}
                briefSections={V3_BRIEF_SECTIONS}
                iconButtons={V3_DEMO_ICON_BUTTONS}
                tokens={{ used: liveMessage.length, max: 8000 }}
                costLabel="≈ $0.001"
              />
            )}
            {previewVariant === "v5" && (
              <ChatComposer
                variant="v5"
                message={liveMessage}
                onMessageChange={setLiveMessage}
                onSend={onSend}
                facets={V5_FACETS}
                iconButtons={V5_ICON_BUTTONS}
                tokens={{ used: liveMessage.length, max: 8000 }}
                onPaletteTrigger={() => setV5PaletteOpen((o) => !o)}
              />
            )}
          </Frame>
        </div>
      </section>

      <section className="mb-14">
        <div className="flex items-baseline gap-3.5 px-4 py-3 border-b border-border bg-surface rounded-t-md">
          <span className="font-mono text-[11px] text-primary tracking-wider">
            V1
          </span>
          <span className="font-display font-semibold text-[17px]">
            Terminal Stack
          </span>
          <span className="font-mono text-[10px] px-2 py-0.5 rounded-full border border-primary/40 text-primary uppercase tracking-wider">
            safe · refinement
          </span>
          <span className="font-serif italic text-text-muted text-sm ml-2">
            A cleaned-up take on today's composer — cyan caret, visible
            source chip above, controls docked below.
          </span>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-[1fr_420px] gap-0 border border-t-0 border-border rounded-b-md bg-surface/40">
          <div className="p-7 border-r-0 xl:border-r border-dashed border-border">
            <div className="font-mono text-[10px] text-text-subtle uppercase tracking-wider mb-2">
              Desktop · /chat
            </div>
            <Frame label="options.html#/chat · Chapter III · an interview with the engine">
              <TerminalStackV1
                message={desktopMessage}
                onMessageChange={setDesktopMessage}
                onSend={onSend}
                sourceChip={{ count: 14, label: "irb-archive" }}
                topChips={V1_DEMO_TOP_CHIPS}
                bottomChips={V1_DEMO_BOTTOM_CHIPS}
                iconButtons={V1_DEMO_ICON_BUTTONS}
                tokens={{ used: 127, max: 8000 }}
                density="desktop"
              />
            </Frame>
          </div>

          <div className="p-7">
            <div className="font-mono text-[10px] text-text-subtle uppercase tracking-wider mb-2">
              Extension sidepanel · 360px
            </div>
            <Frame label="sidepanel" width={360}>
              <TerminalStackV1
                message={compactMessage}
                onMessageChange={setCompactMessage}
                onSend={onSend}
                sourceChip={{ count: 14, label: "irb-archive" }}
                topChips={SP_DEMO_TOP_CHIPS}
                iconButtons={SP_DEMO_ICON_BUTTONS}
                tokens={{ used: 84, max: 8000 }}
                density="compact"
              />
            </Frame>
          </div>
        </div>
      </section>

      <section className="mb-14">
        <div className="flex items-baseline gap-3.5 px-4 py-3 border-b border-border bg-surface rounded-t-md">
          <span className="font-mono text-[11px] text-primary tracking-wider">
            V3
          </span>
          <span className="font-display font-semibold text-[17px]">
            Split Brief
          </span>
          <span className="font-mono text-[10px] px-2 py-0.5 rounded-full border border-primary/40 text-primary uppercase tracking-wider">
            safe · structured
          </span>
          <span className="font-serif italic text-text-muted text-sm ml-2">
            Left pane is the brief (persona, sources, model, temp) as
            labelled field chips. Right pane is the question.
          </span>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-[1fr_420px] gap-0 border border-t-0 border-border rounded-b-md bg-surface/40">
          <div className="p-7 border-r-0 xl:border-r border-dashed border-border">
            <div className="font-mono text-[10px] text-text-subtle uppercase tracking-wider mb-2">
              Desktop · /chat
            </div>
            <Frame label="options.html#/chat">
              <SplitBriefV3
                message={v3DesktopMessage}
                onMessageChange={setV3DesktopMessage}
                onSend={onSend}
                briefSections={V3_BRIEF_SECTIONS}
                iconButtons={V3_DEMO_ICON_BUTTONS}
                tokens={{ used: 284, max: 8000 }}
                costLabel="≈ $0.003"
                density="desktop"
              />
            </Frame>
          </div>

          <div className="p-7">
            <div className="font-mono text-[10px] text-text-subtle uppercase tracking-wider mb-2">
              Extension sidepanel · 360px
            </div>
            <Frame label="sidepanel" width={360}>
              <SplitBriefV3
                message={v3CompactMessage}
                onMessageChange={setV3CompactMessage}
                onSend={onSend}
                briefSections={V3_SP_BRIEF_SECTIONS}
                iconButtons={V3_SP_ICON_BUTTONS}
                tokens={{ used: 84, max: 8000 }}
                density="compact"
              />
            </Frame>
          </div>
        </div>
      </section>

      <section className="mb-14">
        <div className="flex items-baseline gap-3.5 px-4 py-3 border-b border-border bg-surface rounded-t-md">
          <span className="font-mono text-[11px] text-primary tracking-wider">
            V5
          </span>
          <span className="font-display font-semibold text-[17px]">
            Radial Command
          </span>
          <span className="font-mono text-[10px] px-2 py-0.5 rounded-full border border-accent/40 text-accent uppercase tracking-wider">
            explore · palette-first
          </span>
          <span className="font-serif italic text-text-muted text-sm ml-2">
            Everything collapses to a single line; typing <code className="font-mono text-primary">/</code> opens an inline palette with every composer capability.
          </span>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-[1fr_420px] gap-0 border border-t-0 border-border rounded-b-md bg-surface/40">
          <div className="p-7 border-r-0 xl:border-r border-dashed border-border">
            <div className="font-mono text-[10px] text-text-subtle uppercase tracking-wider mb-2">
              Desktop · /chat · slash menu open
            </div>
            <Frame label="options.html#/chat · command mode">
              <RadialCommandV5
                message={v5DesktopMessage}
                onMessageChange={setV5DesktopMessage}
                onSend={onSend}
                facets={V5_FACETS}
                tokens={{ used: 284, max: 8000 }}
                iconButtons={V5_ICON_BUTTONS}
                paletteOpen={v5PaletteOpen}
                paletteGroups={V5_DEMO_PALETTE_GROUPS}
                paletteActiveIndex={v5PaletteIdx}
                onPaletteActiveIndexChange={setV5PaletteIdx}
                paletteQuery="model"
                paletteMatchCountLabel="14 commands matched"
                onPaletteSelect={() => setV5PaletteOpen(false)}
                onPaletteTrigger={() => setV5PaletteOpen((open) => !open)}
                density="desktop"
              />
            </Frame>
          </div>

          <div className="p-7">
            <div className="font-mono text-[10px] text-text-subtle uppercase tracking-wider mb-2">
              Extension sidepanel · 360px · empty
            </div>
            <Frame label="sidepanel · new chat" width={360}>
              <RadialCommandV5
                message={v5CompactMessage}
                onMessageChange={setV5CompactMessage}
                onSend={onSend}
                placeholder="Ask anything · / for commands"
                facets={V5_SP_FACETS}
                iconButtons={V5_SP_ICON_BUTTONS}
                onPaletteTrigger={() => {}}
                density="compact"
              />
            </Frame>
          </div>
        </div>
      </section>

      <footer className="text-center text-text-subtle font-mono text-xs">
        All three variants rendered. Next: a top-level ChatComposer that
        picks one based on <code className="text-primary">preferences.ui.composer_variant</code>.
      </footer>
    </div>
  )
}

export default ComposerVariantsPreview
