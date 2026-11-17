import { useTranslation } from "react-i18next"

type AdvancedFiltersProps = {
  filterText: string
  onFilterTextChange: (value: string) => void
  totalCount: number
  selectedCount: number
}

export function AdvancedFilters({
  filterText,
  onFilterTextChange,
  totalCount,
  selectedCount
}: AdvancedFiltersProps) {
  const { t } = useTranslation(["option"])

  return (
    <section className="space-y-2">
      <header className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">
          {t("option:repo2txt.filters", { defaultValue: "Filters" })}
        </h3>
        <span className="text-xs text-text-subtle">
          {t("option:repo2txt.selectedCount", {
            defaultValue: "{{selected}}/{{total}} selected",
            selected: selectedCount,
            total: totalCount
          })}
        </span>
      </header>
      <p className="text-xs text-text-subtle">
        {t("option:repo2txt.selectionPolicy", {
          defaultValue:
            "Only checked files are exported. Search filters this list only; .gitignore is not applied. UTF-8 text only: 1 MiB per file, 10 MiB selected text/ZIP, 5,000 entries. Binary files and symbolic links are rejected."
        })}
      </p>
      <input
        type="search"
        value={filterText}
        onChange={(event) => onFilterTextChange(event.target.value)}
        placeholder={t("option:repo2txt.filterFiles", {
          defaultValue: "Filter files"
        })}
        className="w-full rounded border px-3 py-1.5 text-sm"
      />
    </section>
  )
}
