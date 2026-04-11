/**
 * BasicSettings - Common RAG options (always visible in Basic mode)
 */

import React from "react"
import { Tooltip } from "antd"
import { HelpCircle } from "lucide-react"
import { useKnowledgeQA } from "../KnowledgeQAProvider"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { cn } from "@/libs/utils"

export function BasicSettings() {
  const { settings, updateSetting, preset } = useKnowledgeQA()
  const { capabilities, loading: capsLoading } = useServerCapabilities()
  const webFallbackHelpId = React.useId()
  const webFallbackHelpText =
    "Requires a configured web search provider on the server (e.g., DuckDuckGo/Brave/Bing/Google/Tavily)."

  return (
    <div className="space-y-6">
      {/* Search Mode */}
      <div className="space-y-2">
        <label className="text-sm font-medium">Search Mode</label>
        <select
          value={settings.search_mode}
          onChange={(e) => updateSetting("search_mode", e.target.value as typeof settings.search_mode)}
          className="w-full px-3 py-2 rounded-md border border-border bg-surface focus:outline-none focus:ring-2 focus:ring-primary"
        >
          <option value="hybrid">Hybrid (Recommended)</option>
          <option value="vector">Vector Only</option>
          <option value="fts">Full-Text Only</option>
        </select>
        <p className="text-xs text-text-muted">
          Hybrid combines full-text and semantic search for best results
        </p>
      </div>

      {/* Top-K Results */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium">Number of Sources</label>
          <span className="text-sm text-text-muted">{settings.top_k}</span>
        </div>
        <input
          type="range"
          min={1}
          max={50}
          value={settings.top_k}
          onChange={(e) => updateSetting("top_k", parseInt(e.target.value, 10))}
          className="w-full accent-primary"
        />
        <p className="text-xs text-text-muted">
          How many documents to retrieve (5-10 for quick, 20+ for thorough)
        </p>
        {settings.top_k > 30 && preset === "thorough" && (
          <p className="text-xs text-warn">
            High source count with Deep preset may cause slow responses.
          </p>
        )}
      </div>

      {/* Source Types */}
      <div className="space-y-2">
        <label className="text-sm font-medium">Search Sources</label>
        <div className="space-y-2">
          {[
            { value: "media_db", label: "Your Documents" },
            { value: "notes", label: "Your Notes" },
            { value: "characters", label: "Characters & Profiles" },
            { value: "chats", label: "Conversations" },
            { value: "kanban", label: "Boards" },
          ].map((source) => (
            <label key={source.value} className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={settings.sources.includes(source.value as typeof settings.sources[number])}
                onChange={(e) => {
                  const newSources = e.target.checked
                    ? [...settings.sources, source.value as typeof settings.sources[number]]
                    : settings.sources.filter((s) => s !== source.value)
                  updateSetting("sources", newSources)
                }}
                className="rounded border-border"
              />
              <span className="text-sm">{source.label}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Web Search Fallback */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <label id="web-fallback-label" className="text-sm font-medium">Web Search Fallback</label>
            <Tooltip title={webFallbackHelpText}>
              <button
                type="button"
                aria-label="Web search fallback requirements"
                aria-describedby={webFallbackHelpId}
                className="inline-flex items-center"
              >
                <HelpCircle className="h-4 w-4 text-text-muted" aria-hidden="true" />
              </button>
            </Tooltip>
            <span id={webFallbackHelpId} className="sr-only">
              {webFallbackHelpText}
            </span>
          </div>
          <button
            role="switch"
            aria-checked={settings.enable_web_fallback}
            aria-labelledby="web-fallback-label"
            onClick={() => updateSetting("enable_web_fallback", !settings.enable_web_fallback)}
            className={cn(
              "relative inline-flex h-6 w-11 items-center rounded-full transition-colors",
              settings.enable_web_fallback ? "bg-primary" : "bg-muted"
            )}
          >
            <span
              className={cn(
                "inline-block h-4 w-4 rounded-full bg-white transition-transform",
                settings.enable_web_fallback ? "translate-x-6" : "translate-x-1"
              )}
            />
          </button>
        </div>
        <p className="text-xs text-text-muted">
          When local sources are weak, Knowledge QA can pull web results.
        </p>
        {settings.enable_web_fallback && !capsLoading && capabilities && !capabilities.hasWebSearch && (
          <div className="text-xs text-warn">
            Web search isn’t available on this server, so fallback may not return results.
          </div>
        )}
      </div>

      {/* Generation Toggle */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label id="generate-answer-label" className="text-sm font-medium">Generate Answer</label>
          <button
            role="switch"
            aria-checked={settings.enable_generation}
            aria-labelledby="generate-answer-label"
            onClick={() => updateSetting("enable_generation", !settings.enable_generation)}
            className={cn(
              "relative inline-flex h-6 w-11 items-center rounded-full transition-colors",
              settings.enable_generation ? "bg-primary" : "bg-muted"
            )}
          >
            <span
              className={cn(
                "inline-block h-4 w-4 rounded-full bg-white transition-transform",
                settings.enable_generation ? "translate-x-6" : "translate-x-1"
              )}
            />
          </button>
        </div>
        <p className="text-xs text-text-muted">
          Use AI to synthesize an answer from retrieved documents
        </p>
      </div>

      {/* Citations Toggle */}
      {settings.enable_generation && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label id="include-citations-label" className="text-sm font-medium">Include Citations</label>
            <button
              role="switch"
              aria-checked={settings.enable_citations}
              aria-labelledby="include-citations-label"
              onClick={() => updateSetting("enable_citations", !settings.enable_citations)}
              className={cn(
                "relative inline-flex h-6 w-11 items-center rounded-full transition-colors",
                settings.enable_citations ? "bg-primary" : "bg-muted"
              )}
            >
              <span
                className={cn(
                  "inline-block h-4 w-4 rounded-full bg-white transition-transform",
                  settings.enable_citations ? "translate-x-6" : "translate-x-1"
                )}
              />
            </button>
          </div>
          <p className="text-xs text-text-muted">
            Add inline citations [1], [2] to reference sources
          </p>
        </div>
      )}

      {/* Reranking Toggle */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label id="enable-reranking-label" className="text-sm font-medium">Enable Reranking</label>
          <button
            role="switch"
            aria-checked={settings.enable_reranking}
            aria-labelledby="enable-reranking-label"
            onClick={() => updateSetting("enable_reranking", !settings.enable_reranking)}
            className={cn(
              "relative inline-flex h-6 w-11 items-center rounded-full transition-colors",
              settings.enable_reranking ? "bg-primary" : "bg-muted"
            )}
          >
            <span
              className={cn(
                "inline-block h-4 w-4 rounded-full bg-white transition-transform",
                settings.enable_reranking ? "translate-x-6" : "translate-x-1"
              )}
            />
          </button>
        </div>
        <p className="text-xs text-text-muted">
          Improve relevance by reranking results (slightly slower)
        </p>
      </div>

      {/* Max Tokens */}
      {settings.enable_generation && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">Answer Length</label>
            <span className="text-sm text-text-muted">
              {settings.max_generation_tokens} tokens
            </span>
          </div>
          <input
            type="range"
            min={50}
            max={2000}
            step={50}
            value={settings.max_generation_tokens}
            onChange={(e) => updateSetting("max_generation_tokens", parseInt(e.target.value, 10))}
            className="w-full accent-primary"
          />
          <div className="flex justify-between text-xs text-text-muted">
            <span>Brief</span>
            <span>Detailed</span>
          </div>
        </div>
      )}
    </div>
  )
}
