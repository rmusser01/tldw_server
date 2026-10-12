import React from "react";
import { Button, Select } from "antd";
import { useTranslation } from "react-i18next";
import type { Diagnostic } from "@/services/document-editing/types";
import { scalarToUtf16, logicalPosition } from "./document-positions";

const modes = ["text", "markdown", "yaml", "json", "jsonl"];
export function DocumentDiagnostics(props: {
  language: string | null | undefined;
  persistence: "portable" | "temporary" | "unavailable";
  disabled: boolean;
  languageDisabled?: boolean;
  formatDisabled: boolean;
  status: string;
  announce?: boolean;
  diagnostics: readonly Diagnostic[];
  complete?: boolean;
  truncated?: boolean;
  source: string;
  onLanguageChange: (value: string | null) => void;
  onCheck: () => void;
  onFormat: () => void;
  onNavigate: (start: number, end: number) => void;
}) {
  const { t } = useTranslation("option");
  const label = (name: string, fallback: string) =>
    t(`notesDocument.${name}`, { defaultValue: fallback });
  const unsupported =
    props.language !== undefined &&
    props.language !== null &&
    !modes.includes(props.language);
  const statusLabels: Record<string, string> = {
    idle: label("idle", "Checking unavailable"),
    pending: label("pending", "Check pending"),
    checked:
      props.complete === false || props.truncated
        ? label("incomplete", "Check incomplete; showing limited Problems")
        : props.diagnostics.length
          ? label("problemsFound", "Check complete; review Problems")
          : label("checked", "Checked; no problems found"),
    unsupported: label("unsupported", "This mode or source cannot be checked"),
    resource_limited: label(
      "resourceLimited",
      "Check reached a resource limit; source kept",
    ),
    failed: label("failed", "Check failed; retry Check"),
    formatted: label("formatted", "Formatted"),
    unchanged: label("unchanged", "Already formatted"),
    blocked: label(
      "blocked",
      "Formatting not applied; review Problems or focus the source and retry",
    ),
  };
  return (
    <div className="mt-3 space-y-2" data-testid="notes-document-tools">
      <div className="flex flex-wrap items-center gap-2">
        <label htmlFor="notes-document-language" className="text-sm text-text">
          {label("language", "Language")}
        </label>
        <Select
          id="notes-document-language"
          aria-label={label("language", "Language")}
          className="min-w-32"
          value={props.language === undefined ? -2 : (props.language ?? -1)}
          disabled={
            props.languageDisabled || props.persistence === "unavailable"
          }
          onChange={(value) => {
            if (value === -1 || typeof value === "string")
              props.onLanguageChange(value === -1 ? null : value);
          }}
          options={[
            { value: -1, label: label("default", "Default (Markdown)") },
            ...modes.map((value) => ({
              value,
              label: label(
                value,
                value === "text" ? "Plain text" : value.toUpperCase(),
              ),
            })),
            ...(props.language === undefined
              ? [{ value: -2, label: label("unknown", "Unspecified") }]
              : []),
            ...(unsupported
              ? [
                  {
                    value: props.language!,
                    label: t("notesDocument.future", {
                      defaultValue: "Unsupported: {{language}}",
                      language: props.language,
                    }),
                  },
                ]
              : []),
          ]}
        />
        <Button size="small" disabled={props.disabled} onClick={props.onCheck}>
          {label("check", "Check document")}
        </Button>
        <Button
          size="small"
          disabled={props.formatDisabled}
          onMouseDown={(event) => event.preventDefault()}
          onClick={props.onFormat}
        >
          {label("format", "Format document")}
        </Button>
      </div>
      {props.persistence !== "portable" ? (
        <p className="text-xs text-text-muted">
          {props.persistence === "temporary"
            ? label(
                "temporary",
                "Temporary display mode for this note/session; this server cannot save language metadata.",
              )
            : label(
                "metadataUnavailable",
                "Language metadata support is unavailable; reconnect to select a mode.",
              )}
        </p>
      ) : null}
      <p
        role="status"
        aria-live={props.announce ? "polite" : "off"}
        aria-atomic="true"
        className="text-xs text-text-muted"
        data-testid="notes-document-status"
      >
        {statusLabels[props.status] ?? statusLabels.failed}
      </p>
      {props.diagnostics.length > 0 ? (
        <div className="max-h-40 overflow-auto">
          <h3 className="text-sm font-medium text-text">
            {label("problems", "Problems")}
          </h3>
          <ul aria-label={label("problems", "Problems")} className="space-y-1">
            {props.diagnostics.map((item, index) => {
              const start = scalarToUtf16(props.source, item.start);
              const end = scalarToUtf16(props.source, item.end);
              const position = logicalPosition(props.source, start);
              return (
                <li key={`${item.code}:${item.start}:${index}`}>
                  <button
                    type="button"
                    className="text-left text-sm text-text hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                    onClick={() => props.onNavigate(start, end)}
                  >{`${position.line}:${position.column} — ${label(`diagnostic.${item.code}`, item.message)}`}</button>
                </li>
              );
            })}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
