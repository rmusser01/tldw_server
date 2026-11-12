import { Input, Segmented, Select } from "antd";

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
  <div className="p-2 border-b border-border flex shrink-0 flex-col gap-2">
    {/* A labelled two-way switch (CC-06): it always shows both choices. */}
    <div
      className="min-w-0"
      onKeyDown={(event) => {
        if (event.key !== "Escape") event.stopPropagation();
      }}
    >
      <Segmented<PlaygroundModelListScope>
        block
        size="small"
        data-testid="model-list-scope-toggle"
        aria-label={t("playground:composer.modelScopeLabel", "Models to show")}
        value={modelListScope}
        onChange={(value) => setModelListScope(value)}
        options={[
          {
            value: "configured",
            label: t("playground:composer.modelScopeReady", "Ready to use"),
          },
          {
            value: "catalog",
            label: t("playground:composer.modelScopeAll", "All models"),
          },
        ]}
      />
    </div>
    <div className="flex flex-wrap items-center gap-2">
      <Input
        size="small"
        placeholder={t(
          modelListScope === "catalog"
            ? "playground:composer.modelCatalogSearchPlaceholder"
            : "playground:composer.modelSearchPlaceholder",
          modelListScope === "catalog"
            ? "Search all models"
            : "Search models",
        )}
        aria-label={t(
          "playground:composer.modelSearchLabel",
          "Search models",
        )}
        value={modelSearchQuery}
        allowClear
        className="min-w-0 flex-1 basis-36"
        onChange={(event) => setModelSearchQuery(event.target.value)}
        onKeyDown={(event) => {
          if (event.key !== "Escape") event.stopPropagation();
        }}
      />
      <Select
        size="small"
        aria-label={t("playground:composer.modelSortLabel", "Sort models")}
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
        className="min-w-0 flex-1 basis-32"
        onKeyDown={(event) => {
          if (event.key !== "Escape") event.stopPropagation();
        }}
      />
    </div>
  </div>
);
