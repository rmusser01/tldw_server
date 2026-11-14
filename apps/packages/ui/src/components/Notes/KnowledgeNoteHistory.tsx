import { useTranslation } from "react-i18next";
import { resolveKnowledgeNoteProvenance } from "@/utils/knowledge-note-provenance";
import {
  getKnowledgeAnswerTrustLabel,
  getKnowledgeTrustReasonMessage,
  isKnowledgeAnswerTrustState,
} from "@/components/Option/KnowledgeQA/trustState";

/** Retained references are descriptive text, never authority to fetch a source. */
export function KnowledgeNoteHistory({ note }: { note: unknown }) {
  const { t } = useTranslation("option");
  const { provenance } = resolveKnowledgeNoteProvenance(note);
  if (!provenance) return null;
  const sources =
    provenance.sources ??
    provenance.research?.sources.flatMap((source) => source.evidence.sources) ??
    [];
  return (
    <details className="mt-2 text-xs text-text-muted">
      <summary className="cursor-pointer">
        {t("notesSearch.sourceHistory", "Original source history")}
      </summary>
      <div
        className="mt-2 max-h-60 space-y-2 overflow-auto whitespace-pre-wrap break-words"
        tabIndex={0}
        role="region"
        aria-label={t("notesSearch.sourceHistory", "Original source history")}
      >
        <p>
          {t(
            "notesSearch.historyExplanation",
            "This history describes the original answer. Later edits do not verify it again.",
          )}
        </p>
        {provenance.question && <p>{provenance.question}</p>}
        {isKnowledgeAnswerTrustState(provenance.trust_state) && (
          <p>{getKnowledgeAnswerTrustLabel(provenance.trust_state)}</p>
        )}
        {provenance.trust_reason_codes?.map((reason, index) => (
          <p key={index}>
            {getKnowledgeTrustReasonMessage(
              reason as Parameters<typeof getKnowledgeTrustReasonMessage>[0],
            ) || reason}
          </p>
        ))}
        {provenance.scope?.sources?.length ? (
          <p>
            {t("notesSearch.historyLibraries", "Library sources:")}{" "}
            {provenance.scope.sources.join(", ")}
          </p>
        ) : null}
        {provenance.scope?.include_media_ids?.length ? (
          <p>
            {t("notesSearch.historyMedia", "Selected media:")}{" "}
            {provenance.scope.include_media_ids.join(", ")}
          </p>
        ) : null}
        {provenance.scope?.include_note_ids?.length ? (
          <p>
            {t("notesSearch.historyNotes", "Selected notes:")}{" "}
            {provenance.scope.include_note_ids.join(", ")}
          </p>
        ) : null}
        {provenance.scope?.keyword_filter ? (
          <p>
            {t("notesSearch.historyKeywords", "Keyword filter:")}{" "}
            {Array.isArray(provenance.scope.keyword_filter)
              ? provenance.scope.keyword_filter.join(", ")
              : provenance.scope.keyword_filter}
          </p>
        ) : null}
        {provenance.scope?.enable_web_fallback != null && (
          <p>
            {provenance.scope.enable_web_fallback
              ? t("notesSearch.historyWebAllowed", "Web fallback was allowed.")
              : t("notesSearch.historyWebOff", "Web fallback was off.")}
          </p>
        )}
        {sources.map((source, index) => (
          <blockquote key={index} className="border-l border-border pl-2">
            <p className="font-medium">{source.title}</p>
            <p>{source.excerpt}</p>
            {source.url && <p>{source.url}</p>}
          </blockquote>
        ))}
      </div>
    </details>
  );
}
