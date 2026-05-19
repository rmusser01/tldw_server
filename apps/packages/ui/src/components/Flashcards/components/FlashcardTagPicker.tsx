import React from "react"
import { Select } from "antd"
import { useTranslation } from "react-i18next"

import { useDebounce } from "@/hooks/useDebounce"
import { useGlobalFlashcardTagSuggestionsQuery } from "../hooks"
import { normalizeFlashcardTags } from "../utils/tag-normalization"

type FlashcardTagPickerProps = {
  value?: string[]
  onChange?: (value: string[]) => void
  active?: boolean
  disabled?: boolean
  placeholder?: string
  className?: string
  dataTestId?: string
}

const DEFAULT_WRAPPER_TEST_ID = "flashcard-tag-picker"
const SEARCH_DEBOUNCE_MS = 250

export const FlashcardTagPicker: React.FC<FlashcardTagPickerProps> = ({
  value,
  onChange,
  active = true,
  disabled = false,
  placeholder,
  className,
  dataTestId
}) => {
  const { t } = useTranslation(["option"])
  const [dropdownOpen, setDropdownOpen] = React.useState(false)
  const [searchText, setSearchText] = React.useState("")
  const wrapperRef = React.useRef<HTMLDivElement>(null)
  const wrapperTestId = dataTestId ?? DEFAULT_WRAPPER_TEST_ID
  const searchInputTestId = `${wrapperTestId}-search-input`

  const debouncedSearchText = useDebounce(searchText, SEARCH_DEBOUNCE_MS)
  const normalizedValue = React.useMemo(() => normalizeFlashcardTags(value), [value])
  const queryEnabled = active && dropdownOpen

  const tagSuggestionsQuery = useGlobalFlashcardTagSuggestionsQuery(debouncedSearchText, {
    enabled: queryEnabled,
    limit: 50
  })

  const options = React.useMemo(
    () =>
      (tagSuggestionsQuery.data?.items ?? []).map((item) => ({
        label: item.tag,
        value: item.tag
      })),
    [tagSuggestionsQuery.data]
  )

  React.useEffect(() => {
    if (!dropdownOpen) return

    const setSearchInputTestId = () => {
      const searchInput = wrapperRef.current?.querySelector<HTMLInputElement>(
        "input.ant-select-selection-search-input, input.ant-select-input"
      )
      if (searchInput && searchInput.getAttribute("data-testid") !== searchInputTestId) {
        searchInput.setAttribute("data-testid", searchInputTestId)
      }
    }

    if (typeof window.requestAnimationFrame === "function") {
      const frameId = window.requestAnimationFrame(setSearchInputTestId)
      return () => window.cancelAnimationFrame(frameId)
    }

    const timeoutId = window.setTimeout(setSearchInputTestId, 0)
    return () => window.clearTimeout(timeoutId)
  }, [dropdownOpen, searchInputTestId])

  const handleChange = React.useCallback(
    (nextValue: unknown) => {
      const nextTags = Array.isArray(nextValue) ? nextValue.map((tag) => String(tag)) : []
      onChange?.(normalizeFlashcardTags(nextTags))
    },
    [onChange]
  )

  return (
    <div ref={wrapperRef} data-testid={wrapperTestId} className={className}>
      <Select
        mode="tags"
        value={normalizedValue}
        onChange={handleChange}
        open={dropdownOpen}
        onOpenChange={setDropdownOpen}
        showSearch
        searchValue={searchText}
        onSearch={setSearchText}
        filterOption={false}
        allowClear
        disabled={disabled}
        placeholder={
          placeholder ??
          t("option:flashcards.tagsPlaceholder", {
            defaultValue: "Add tags..."
          })
        }
        options={options}
        loading={tagSuggestionsQuery.isFetching}
        notFoundContent={tagSuggestionsQuery.isError ? null : undefined}
      />
    </div>
  )
}

export default FlashcardTagPicker
