import { Search, X, FilterX } from 'lucide-react'
import { Popover, Tooltip } from 'antd'
import { useTranslation } from 'react-i18next'
import type { Ref } from 'react'

interface SearchBarProps {
  value: string
  onChange: (value: string) => void
  placeholder?: string
  inputRef?: Ref<HTMLInputElement>
  onClearAll?: () => void // Optional callback to reset filters too
  hasActiveFilters?: boolean // Whether there are active filters to clear
}

export function SearchBar({
  value,
  onChange,
  placeholder = 'Search media (title/content)',
  inputRef,
  onClearAll,
  hasActiveFilters = false
}: SearchBarProps) {
  const { t } = useTranslation(['review'])

  const showClearSearch = value.length > 0
  const showClearAll = Boolean(hasActiveFilters && onClearAll)
  const inputPaddingClass = showClearSearch && showClearAll ? 'pr-16' : 'pr-10'

  const handleClearSearch = () => {
    onChange('')
  }

  const handleClearAll = () => {
    onChange('')
    if (onClearAll) {
      onClearAll()
    }
  }

  return (
    <div>
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-text-subtle" />
        <input
          ref={inputRef}
          type="text"
          data-testid="media-search-input"
          placeholder={placeholder}
          aria-label={placeholder}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className={`w-full pl-10 ${inputPaddingClass} py-2.5 border border-border bg-surface text-text placeholder:text-text-muted rounded-lg focus:outline-none focus:ring-2 focus:ring-focus focus:border-transparent`}
        />
        {/* Clear search button */}
        {showClearSearch && (
          <button
            onClick={handleClearSearch}
            data-testid="media-search-clear"
            className={`absolute top-1/2 -translate-y-1/2 text-text-subtle hover:text-text ${
              showClearAll ? 'right-10' : 'right-3'
            }`}
            aria-label={t('mediaPage.clearSearch', 'Clear search')}
            title={t('mediaPage.clearSearch', 'Clear search')}
            type="button"
          >
            <X className="w-5 h-5" />
          </button>
        )}
        {/* Clear all (search + filters) button when filters active */}
        {showClearAll && (
          <Tooltip title={t('mediaPage.clearAllFilters', 'Clear search and filters')}>
            <button
              onClick={handleClearAll}
              data-testid="media-search-clear-all"
              className="absolute right-3 top-1/2 -translate-y-1/2 text-warn hover:text-warn"
              aria-label={t('mediaPage.clearAllFilters', 'Clear search and filters')}
              title={t('mediaPage.clearAllFilters', 'Clear search and filters')}
              type="button"
            >
              <FilterX className="w-5 h-5" />
            </button>
          </Tooltip>
        )}
      </div>

      <div className="mt-1 flex justify-end">
        <Popover
          trigger="click"
          placement="bottomRight"
          title={t('mediaPage.searchSyntaxHelp', 'Search syntax help')}
          content={
            <div className="max-w-xs space-y-2 text-sm">
              <p>
                {t(
                  'mediaPage.searchSyntaxHelpText',
                  'Advanced syntax: use quotes for exact phrases, AND/OR/NOT, and -term to exclude.'
                )}
              </p>
              <ul className="list-none space-y-1 text-xs text-text-muted">
                <li><code className="bg-surface2 px-1 rounded">&quot;exact phrase&quot;</code> — {t('mediaPage.searchSyntaxExact', 'match exact phrase')}</li>
                <li><code className="bg-surface2 px-1 rounded">term1 AND term2</code> — {t('mediaPage.searchSyntaxAnd', 'both terms required')}</li>
                <li><code className="bg-surface2 px-1 rounded">term1 OR term2</code> — {t('mediaPage.searchSyntaxOr', 'either term matches')}</li>
                <li><code className="bg-surface2 px-1 rounded">NOT term</code> / <code className="bg-surface2 px-1 rounded">-term</code> — {t('mediaPage.searchSyntaxNot', 'exclude term')}</li>
              </ul>
            </div>
          }
        >
          <button
            type="button"
            className="h-5 w-5 rounded-full border border-border text-[11px] font-semibold text-text-muted hover:text-text hover:border-text-muted transition-colors cursor-pointer"
            aria-label={t('mediaPage.searchSyntaxHelp', 'Search syntax help')}
            title={t('mediaPage.searchSyntaxHelp', 'Search syntax help')}
          >
            ?
          </button>
        </Popover>
      </div>
    </div>
  )
}
