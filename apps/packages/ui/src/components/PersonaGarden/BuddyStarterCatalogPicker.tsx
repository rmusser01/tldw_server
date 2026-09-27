import React from "react"
import { Button } from "antd"
import { Copy } from "lucide-react"
import { useTranslation } from "react-i18next"
import { Badge } from "@/components/ui/primitives"
import type { PersonaVisualStarterPackSummary } from "@/types/persona-visuals"
import { groupBuddyStarterPacksByTier } from "./buddyBuilderState"
import { BuddyStarterArtwork } from "./BuddyStarterArtwork"

export type BuddyStarterCatalogPickerProps = {
  starterPacks: PersonaVisualStarterPackSummary[]
  copyingStarterId?: string | null
  loading?: boolean
  error?: string | null
  onRetry?: () => void
  onCopyStarterPack: (starterPackId: string) => void
}

const StarterChoice = ({ starter, copyingStarterId, onCopyStarterPack }: {
  starter: PersonaVisualStarterPackSummary
  copyingStarterId: string | null
  onCopyStarterPack: (id: string) => void
}) => {
  const { t } = useTranslation(["sidepanel", "common"])
  const [previewReady, setPreviewReady] = React.useState(false)
  const hasArtwork = starter.production_status === "art_ready"
  return (
    <article data-testid={`buddy-builder-starter-${starter.id}`} className="min-w-0 overflow-hidden rounded-xl border border-border bg-surface">
      {hasArtwork ? <BuddyStarterArtwork starterId={starter.id} title={starter.title} onReadyChange={setPreviewReady} /> : null}
      <div className="space-y-2 p-3">
        <h3 data-testid="buddy-builder-starter-title" className="text-sm font-semibold text-text">{starter.title}</h3>
        <p className="text-sm leading-5 text-text-muted">{starter.description}</p>
        {!hasArtwork ? <p className="text-sm text-text-muted">{t("sidepanel:personaGarden.visuals.builder.needsArtwork", { defaultValue: "Template only. Add artwork before activation." })}</p> : null}
        <div className="flex flex-wrap items-center gap-2 text-xs text-text-muted">
          <Badge variant="secondary" size="sm">
            {hasArtwork ? t("sidepanel:personaGarden.visuals.builder.readyArtwork", { defaultValue: "Artwork included" }) : t("sidepanel:personaGarden.visuals.builder.template", { defaultValue: "Custom template" })}
          </Badge>
          {starter.license_label ? <span>{starter.license_label}</span> : null}
        </div>
        <Button
          size="small"
          icon={<Copy className="h-3.5 w-3.5" />}
          loading={copyingStarterId === starter.id}
          disabled={Boolean(copyingStarterId) || (hasArtwork && !previewReady)}
          onClick={() => onCopyStarterPack(starter.id)}
        >
          {hasArtwork ? t("sidepanel:personaGarden.visuals.builder.copyRecommended", { defaultValue: "Copy as draft" }) : t("sidepanel:personaGarden.visuals.builder.copyTemplate", { defaultValue: "Copy template as draft" })}
        </Button>
      </div>
    </article>
  )
}

export const BuddyStarterCatalogPicker: React.FC<BuddyStarterCatalogPickerProps> = ({
  starterPacks, copyingStarterId = null, loading = false, error = null, onRetry, onCopyStarterPack
}) => {
  const { t } = useTranslation(["sidepanel", "common"])
  const grouped = groupBuddyStarterPacksByTier(starterPacks)
  const ordered = [...grouped.basic, ...grouped.intermediate, ...grouped.intricate]
  const templates = ordered.filter((starter) => starter.production_status !== "art_ready")
  const choice = (starter: PersonaVisualStarterPackSummary) => <StarterChoice key={starter.id} starter={starter} copyingStarterId={copyingStarterId} onCopyStarterPack={onCopyStarterPack} />
  return (
    <section data-testid="buddy-builder-starter-catalog" className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-base font-semibold text-text">{t("sidepanel:personaGarden.visuals.builder.defaultCatalog", { defaultValue: "Choose a ready-made Buddy" })}</h2>
        {onRetry ? <Button size="small" disabled={loading} onClick={onRetry}>
          {t(error ? "sidepanel:personaGarden.visuals.builder.retryCatalog" : "sidepanel:personaGarden.visuals.builder.refreshCatalog", { defaultValue: error ? "Retry catalog" : "Refresh catalog" })}
        </Button> : null}
      </div>
      <p className="text-sm text-text-muted">{t("sidepanel:personaGarden.visuals.builder.galleryHelp", { defaultValue: "Preview the artwork, then copy a Buddy to review and activate it." })}</p>
      {loading ? <p role="status" className="text-sm text-text-muted">{t("sidepanel:personaGarden.visuals.builder.loading", { defaultValue: "Loading Buddies…" })}</p> : null}
      {error ? <p role="alert" className="text-sm text-text">{error}</p> : null}
      {(["basic", "intermediate", "intricate"] as const).map((tier) => {
        const ready = grouped[tier].filter((starter) => starter.production_status === "art_ready")
        return ready.length ? <div key={tier} data-testid={`buddy-builder-tier-${tier}`} className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">{ready.map(choice)}</div> : null
      })}
      {!loading && !error && !ordered.length ? <p className="text-sm text-text-muted">{t("sidepanel:personaGarden.visuals.builder.catalogEmpty", { defaultValue: "No ready-made Buddies are available on this server. Import a pack or create your own below." })}</p> : null}
      {templates.length ? <details className="border-t border-border pt-3">
        <summary className="cursor-pointer rounded text-sm font-medium text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary">{t("sidepanel:personaGarden.visuals.builder.customTemplates", { defaultValue: "Custom artwork templates" })}</summary>
        <div className="mt-3 grid gap-3 sm:grid-cols-2">{templates.map(choice)}</div>
      </details> : null}
    </section>
  )
}

export default BuddyStarterCatalogPicker
