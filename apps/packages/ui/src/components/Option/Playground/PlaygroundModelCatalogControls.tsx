import { Input, Select } from "antd";

import type { ModelSortMode } from "@/hooks/playground";

type PlaygroundTranslate = (
  key: string,
  defaultValueOrOptions?: unknown,
  options?: unknown,
) => string;

export type PlaygroundModelListScope = "configured" | "catalog";

export type PlaygroundModelCatalogControlsProps = {
  t: PlaygroundTranslate;
  modelListScope: PlaygroundModelListScope;
  setModelListScope: (scope: PlaygroundModelListScope) => void;
  modelSearchQuery: string;
  setModelSearchQuery: (query: string) => void;
  modelSortMode: ModelSortMode;
  setModelSortMode: (mode: ModelSortMode) => void;
};

export const PlaygroundModelCatalogControls = ({
  t,
  modelListScope,
  setModelListScope,
  modelSearchQuery,
  setModelSearchQuery,
  modelSortMode,
  setModelSortMode,
}: PlaygroundModelCatalogControlsProps) => (
  <div className="p-2 border-b border-border flex flex-col gap-2">
    <div className="flex items-center gap-2">
      <button
        type="button"
        data-testid="model-list-scope-toggle"
        aria-pressed={modelListScope === "catalog"}
        className={`rounded border px-2 py-1 text-xs transition-colors ${
          modelListScope === "catalog"
            ? "border-primary bg-primary/10 text-primary"
            : "border-border bg-surface2 text-text-muted hover:text-text"
        }`}
        onClick={(event) => {
          event.preventDefault();
          setModelListScope(
            modelListScope === "catalog" ? "configured" : "catalog",
          );
        }}
      >
        {modelListScope === "catalog"
          ? t("playground:composer.configuredModels", "Configured")
          : t("playground:composer.searchAllModels", "Search all models")}
      </button>
      <span className="text-[11px] text-text-subtle">
        {modelListScope === "catalog"
          ? t("playground:composer.catalogScopeHint", "All known models")
          : t(
              "playground:composer.configuredScopeHint",
              "Usable configured models",
            )}
      </span>
    </div>
    <div className="flex items-center gap-2">
      <Input
        size="small"
        placeholder={t(
          modelListScope === "catalog"
            ? "playground:composer.modelCatalogSearchPlaceholder"
            : "playground:composer.modelSearchPlaceholder",
          modelListScope === "catalog"
            ? "Search all known models"
            : "Search models",
        )}
        aria-label={t(
          "playground:composer.modelSearchLabel",
          "Search models",
        )}
        value={modelSearchQuery}
        allowClear
        className="flex-1"
        onChange={(event) => setModelSearchQuery(event.target.value)}
        onKeyDown={(event) => event.stopPropagation()}
      />
      <Select
        size="small"
        value={modelSortMode}
        onChange={(value) => setModelSortMode(value as ModelSortMode)}
        options={[
          {
            value: "favorites",
            label: t("playground:composer.sort.favorites", "Favorites"),
          },
          { value: "az", label: t("playground:composer.sort.az", "A-Z") },
          {
            value: "provider",
            label: t("playground:composer.sort.provider", "Provider"),
          },
          {
            value: "localFirst",
            label: t("playground:composer.sort.localFirst", "Local-first"),
          },
        ]}
        className="min-w-[120px]"
        onKeyDown={(event) => event.stopPropagation()}
      />
    </div>
  </div>
);
