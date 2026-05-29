import React from "react"
import { Button, Card, Checkbox, Input, InputNumber, Tag, Typography } from "antd"
import type {
  DatasetSample,
  EmbeddingRecipeCandidateHint,
  RecipeManifest
} from "@/services/evaluations"
import { Alert as DsAlert } from "@/components/ui/primitives"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useTranslation } from "react-i18next"
import { useEmbeddingRecipeCandidates } from "../../hooks/useRecipes"

const { Text } = Typography

type Props = {
  datasetSource: "inline" | "saved"
  dataset: DatasetSample[]
  runConfig: Record<string, any>
  manifest?: RecipeManifest | null
  onDatasetChange: (next: DatasetSample[]) => void
  onRunConfigChange: (next: Record<string, any>) => void
}

type CandidateConfig = {
  provider: string
  model: string
}

type MediaSearchResult = {
  id: string
  title: string
  url?: string
}

type EmbeddingDatasetSample = DatasetSample & {
  query_id: string
  input: string
  expected_ids: string[]
}

const DEFAULT_RUN_CONFIG = {
  comparison_mode: "embedding_only",
  candidates: [],
  media_ids: [],
  top_k: 10,
  hybrid_alpha: 0.7,
  guided_source_labeling: true,
  source_id_contract: "media_id"
}

const parseInteger = (value: unknown, fallback: number): number => {
  const next = Number.parseInt(String(value), 10)
  return Number.isFinite(next) ? next : fallback
}

const parseNumeric = (value: unknown, fallback: number): number => {
  const next = Number(value)
  return Number.isFinite(next) ? next : fallback
}

const isIntegerId = (value: unknown): boolean => /^\d+$/.test(String(value ?? "").trim())

const splitMediaIds = (value: string): string[] =>
  Array.from(
    new Set(
      value
        .split(/[,\s]+/)
        .map((entry) => entry.trim())
        .filter(isIntegerId)
    )
  )

const joinMediaIds = (value: unknown): string =>
  Array.isArray(value)
    ? value
        .map((entry) => String(entry ?? "").trim())
        .filter(isIntegerId)
        .join(", ")
    : ""

const normalizeMediaIdArray = (value: unknown): string[] =>
  Array.isArray(value)
    ? Array.from(new Set(value.map((entry) => String(entry ?? "").trim()).filter(isIntegerId)))
    : []

const normalizeCandidates = (value: unknown): CandidateConfig[] =>
  Array.isArray(value)
    ? value
        .map((candidate) => {
          const record =
            candidate && typeof candidate === "object"
              ? (candidate as Record<string, any>)
              : {}
          const provider = String(record.provider ?? "").trim()
          const model = String(record.model ?? "").trim()
          if (!provider || !model) return null
          return { provider, model }
        })
        .filter((candidate): candidate is CandidateConfig => candidate !== null)
    : []

const candidateKey = (candidate: CandidateConfig | EmbeddingRecipeCandidateHint): string =>
  `${candidate.provider}:${candidate.model}`

const candidateFromHint = (
  hint: EmbeddingRecipeCandidateHint
): CandidateConfig => ({
  provider: hint.provider,
  model: hint.model
})

const normalizeRunConfig = (
  runConfig: Record<string, any>,
  manifest?: RecipeManifest | null
): Record<string, any> => {
  const manifestDefaults = manifest?.default_run_config || {}
  const manifestSourceIdContract =
    manifest?.capabilities?.source_labeling?.source_id_contract
  const sourceIdContract =
    typeof runConfig.source_id_contract === "string" &&
    runConfig.source_id_contract.trim()
      ? runConfig.source_id_contract.trim()
      : typeof manifestSourceIdContract?.kind === "string" &&
          manifestSourceIdContract.kind.trim()
        ? manifestSourceIdContract.kind.trim()
        : DEFAULT_RUN_CONFIG.source_id_contract

  return {
    ...DEFAULT_RUN_CONFIG,
    ...manifestDefaults,
    ...runConfig,
    comparison_mode: String(
      runConfig.comparison_mode ||
        manifestDefaults.comparison_mode ||
        DEFAULT_RUN_CONFIG.comparison_mode
    ),
    candidates: normalizeCandidates(runConfig.candidates),
    media_ids: normalizeMediaIdArray(runConfig.media_ids).map((id) =>
      parseInteger(id, 0)
    ).filter((id) => id > 0),
    top_k: Math.max(1, parseInteger(runConfig.top_k ?? manifestDefaults.top_k, 10)),
    hybrid_alpha: Math.max(
      0,
      Math.min(1, parseNumeric(runConfig.hybrid_alpha ?? manifestDefaults.hybrid_alpha, 0.7))
    ),
    guided_source_labeling: true,
    source_id_contract: sourceIdContract
  }
}

const normalizeDataset = (dataset: DatasetSample[]): EmbeddingDatasetSample[] => {
  const baseDataset = Array.isArray(dataset) ? dataset : []
  return baseDataset.map((sample, index) => {
    const record = sample && typeof sample === "object" ? (sample as Record<string, any>) : {}
    return {
      query_id:
        String(record.query_id ?? record.sample_id ?? `q-${index + 1}`).trim() ||
        `q-${index + 1}`,
      input: String(record.input ?? record.query ?? ""),
      expected_ids: normalizeMediaIdArray(record.expected_ids)
    }
  })
}

const extractMediaSearchItems = (payload: unknown): Record<string, any>[] => {
  const record = payload && typeof payload === "object" ? (payload as Record<string, any>) : {}
  const candidates = [record.media, record.results, record.items, record.data]
  const firstArray = candidates.find(Array.isArray)
  return Array.isArray(firstArray) ? (firstArray as Record<string, any>[]) : []
}

const normalizeMediaSearchResults = (payload: unknown): MediaSearchResult[] =>
  extractMediaSearchItems(payload)
    .map((item) => {
      const id = String(item?.id ?? item?.media_id ?? "").trim()
      if (!isIntegerId(id)) return null
      const url =
        typeof item?.url === "string"
          ? item.url
          : typeof item?.source_url === "string"
            ? item.source_url
            : undefined
      const result: MediaSearchResult = {
        id,
        title:
          String(item?.title ?? item?.name ?? item?.filename ?? `Media ${id}`).trim() ||
          `Media ${id}`
      }
      if (url) result.url = url
      return result
    })
    .filter((item): item is MediaSearchResult => item !== null)

export const EmbeddingsModelSelectionConfig: React.FC<Props> = ({
  datasetSource,
  dataset,
  runConfig,
  manifest,
  onDatasetChange,
  onRunConfigChange
}) => {
  const { t } = useTranslation(["evaluations", "common"])
  const normalizedRunConfig = React.useMemo(
    () => normalizeRunConfig(runConfig, manifest),
    [runConfig, manifest]
  )
  const normalizedDataset = React.useMemo(() => normalizeDataset(dataset), [dataset])
  const editableDataset = React.useMemo<EmbeddingDatasetSample[]>(
    () =>
      normalizedDataset.length > 0
        ? normalizedDataset
        : [{ query_id: "q-1", input: "", expected_ids: [] }],
    [normalizedDataset]
  )
  const candidateQuery = useEmbeddingRecipeCandidates(true)
  const candidateHints = React.useMemo<EmbeddingRecipeCandidateHint[]>(() => {
    const response = candidateQuery.data as any
    const candidates = response?.data?.candidates
    return Array.isArray(candidates) ? candidates : []
  }, [candidateQuery.data])
  const readyCandidates = React.useMemo(
    () => candidateHints.filter((candidate) => candidate.status === "ready"),
    [candidateHints]
  )
  const sourceContractDisplay =
    manifest?.capabilities?.source_labeling?.source_id_contract &&
    typeof manifest.capabilities.source_labeling.source_id_contract === "object"
      ? manifest.capabilities.source_labeling.source_id_contract
      : { kind: normalizedRunConfig.source_id_contract, type: "integer" }
  const didPrefillCandidates = React.useRef(false)
  const [searchQueries, setSearchQueries] = React.useState<Record<number, string>>({})
  const [searchResults, setSearchResults] = React.useState<
    Record<number, MediaSearchResult[]>
  >({})
  const [searchErrors, setSearchErrors] = React.useState<Record<number, string | null>>({})
  const [searchingRows, setSearchingRows] = React.useState<Record<number, boolean>>({})

  const applyRunConfig = (updater: (current: Record<string, any>) => Record<string, any>) => {
    onRunConfigChange(normalizeRunConfig(updater(normalizedRunConfig), manifest))
  }

  const applyDataset = (
    updater: (current: EmbeddingDatasetSample[]) => EmbeddingDatasetSample[]
  ) => {
    onDatasetChange(
      updater(datasetSource === "inline" ? editableDataset : normalizedDataset)
    )
  }

  React.useEffect(() => {
    if (didPrefillCandidates.current) return
    if (normalizedRunConfig.candidates.length > 0 || readyCandidates.length === 0) return

    didPrefillCandidates.current = true
    onRunConfigChange(
      normalizeRunConfig(
        {
          ...normalizedRunConfig,
          candidates: readyCandidates.map(candidateFromHint)
        },
        manifest
      )
    )
  }, [manifest, normalizedRunConfig, onRunConfigChange, readyCandidates])

  const updateDatasetSample = (
    sampleIndex: number,
    updater: (sample: EmbeddingDatasetSample) => EmbeddingDatasetSample
  ) => {
    applyDataset((current) =>
      current.map((sample, index) =>
        index === sampleIndex
          ? updater(sample)
          : sample
      )
    )
  }

  const addQuery = () => {
    applyDataset((current) => [
      ...current,
      {
        query_id: `q-${current.length + 1}`,
        input: "",
        expected_ids: []
      }
    ])
  }

  const removeQuery = (sampleIndex: number) => {
    applyDataset((current) =>
      current.length <= 1 ? current : current.filter((_, index) => index !== sampleIndex)
    )
  }

  const toggleCandidate = (candidate: EmbeddingRecipeCandidateHint, checked: boolean) => {
    if (candidate.status !== "ready") return
    applyRunConfig((current) => {
      const currentCandidates = normalizeCandidates(current.candidates)
      const key = candidateKey(candidate)
      const nextCandidates = checked
        ? [
            ...currentCandidates,
            candidateFromHint(candidate)
          ].filter(
            (item, index, all) =>
              all.findIndex((other) => candidateKey(other) === candidateKey(item)) === index
          )
        : currentCandidates.filter((item) => candidateKey(item) !== key)
      return {
        ...current,
        candidates: nextCandidates
      }
    })
  }

  const searchMediaForRow = async (sampleIndex: number, query: string) => {
    const trimmedQuery = query.trim()
    setSearchQueries((current) => ({ ...current, [sampleIndex]: query }))
    setSearchErrors((current) => ({ ...current, [sampleIndex]: null }))
    if (!trimmedQuery) {
      setSearchResults((current) => ({ ...current, [sampleIndex]: [] }))
      return
    }

    setSearchingRows((current) => ({ ...current, [sampleIndex]: true }))
    try {
      const response = await tldwClient.searchMedia(
        { query: trimmedQuery },
        { page: 1, results_per_page: 8 }
      )
      setSearchResults((current) => ({
        ...current,
        [sampleIndex]: normalizeMediaSearchResults(response)
      }))
    } catch (error: any) {
      setSearchResults((current) => ({ ...current, [sampleIndex]: [] }))
      setSearchErrors((current) => ({
        ...current,
        [sampleIndex]:
          error?.message ||
          t("evaluations:embeddingRecipeMediaSearchError", {
            defaultValue: "Media search failed."
          })
      }))
    } finally {
      setSearchingRows((current) => ({ ...current, [sampleIndex]: false }))
    }
  }

  const toggleExpectedId = (
    sampleIndex: number,
    mediaId: string,
    checked: boolean
  ) => {
    if (!isIntegerId(mediaId)) return
    updateDatasetSample(sampleIndex, (sample) => {
      const currentIds = normalizeMediaIdArray(sample.expected_ids)
      const nextIds = checked
        ? Array.from(new Set([...currentIds, mediaId]))
        : currentIds.filter((id) => id !== mediaId)
      return {
        ...sample,
        expected_ids: nextIds
      }
    })
  }

  const selectedCandidateKeys = new Set(
    normalizeCandidates(normalizedRunConfig.candidates).map(candidateKey)
  )

  return (
    <div className="space-y-4">
      <Card size="small" title="Corpus">
        <div className="grid gap-3 md:grid-cols-3">
          <div>
            <Text strong>Comparison mode</Text>
            <Input
              aria-label="Comparison mode"
              className="mt-2"
              value={String(normalizedRunConfig.comparison_mode || "")}
              onChange={(event) =>
                applyRunConfig((current) => ({
                  ...current,
                  comparison_mode: event.target.value
                }))
              }
            />
          </div>
          <div>
            <Text strong>Media IDs</Text>
            <Input
              aria-label="Media IDs"
              className="mt-2"
              value={joinMediaIds(normalizedRunConfig.media_ids)}
              onChange={(event) =>
                applyRunConfig((current) => ({
                  ...current,
                  media_ids: splitMediaIds(event.target.value).map((id) =>
                    parseInteger(id, 0)
                  )
                }))
              }
            />
          </div>
          <div>
            <Text strong>Source ID contract</Text>
            <div className="mt-2">
              <Tag>{String(sourceContractDisplay.kind || "media_id")}</Tag>
              <Tag>{String(sourceContractDisplay.type || "integer")}</Tag>
            </div>
          </div>
        </div>
      </Card>

      <Card size="small" title="Queries">
        <div className="space-y-3">
          {editableDataset.map((sample, index) => {
            return (
              <div key={`${sample.query_id}-${index}`} className="space-y-2 border-b border-border-subtle pb-3 last:border-b-0 last:pb-0">
                <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_160px]">
                  <div>
                    <Text strong>Query text {index + 1}</Text>
                    <Input
                      aria-label={`Query text ${index + 1}`}
                      className="mt-2"
                      value={sample.input}
                      onChange={(event) =>
                        updateDatasetSample(index, (current) => ({
                          ...current,
                          input: event.target.value
                        }))
                      }
                    />
                  </div>
                  <div className="flex items-end justify-end">
                    <Button
                      size="small"
                      disabled={editableDataset.length <= 1}
                      onClick={() => removeQuery(index)}
                    >
                      Remove
                    </Button>
                  </div>
                </div>
              </div>
            )
          })}
          <Button size="small" onClick={addQuery}>
            Add query
          </Button>
        </div>
      </Card>

      <Card size="small" title="Expected sources">
        <div className="space-y-4">
          {editableDataset.map((sample, index) => {
            const expectedIds = normalizeMediaIdArray(sample.expected_ids)
            const rowResults = searchResults[index] || []
            return (
              <div key={`sources-${sample.query_id}-${index}`} className="space-y-3 border-b border-border-subtle pb-4 last:border-b-0 last:pb-0">
                <div>
                  <Text strong>Expected media IDs {index + 1}</Text>
                  <Input
                    aria-label={`Expected media IDs ${index + 1}`}
                    className="mt-2"
                    value={joinMediaIds(expectedIds)}
                    onChange={(event) =>
                      updateDatasetSample(index, (current) => ({
                        ...current,
                        expected_ids: splitMediaIds(event.target.value)
                      }))
                    }
                  />
                </div>
                <div>
                  <Text strong>Find expected sources for query {index + 1}</Text>
                  <div className="mt-2 flex gap-2">
                    <Input
                      aria-label={`Find expected sources for query ${index + 1}`}
                      value={searchQueries[index] ?? ""}
                      onChange={(event) =>
                        void searchMediaForRow(index, event.target.value)
                      }
                    />
                    <Button
                      size="small"
                      loading={Boolean(searchingRows[index])}
                      onClick={() =>
                        void searchMediaForRow(index, searchQueries[index] ?? "")
                      }
                    >
                      Search
                    </Button>
                  </div>
                  {searchErrors[index] && (
                    <DsAlert className="mt-2" variant="error">
                      {searchErrors[index]}
                    </DsAlert>
                  )}
                  {rowResults.length > 0 && (
                    <div className="mt-2 grid gap-2 md:grid-cols-2">
                      {rowResults.map((result) => (
                        <Checkbox
                          key={`${index}-${result.id}`}
                          checked={expectedIds.includes(result.id)}
                          onChange={(event) =>
                            toggleExpectedId(index, result.id, event.target.checked)
                          }
                        >
                          <span>{result.title}</span>
                          <span className="ml-2 text-xs text-text-muted">#{result.id}</span>
                        </Checkbox>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      </Card>

      <Card size="small" title="Models">
        <div className="space-y-3">
          {candidateHints.length === 0 ? (
            <Text type="secondary">No candidate hints loaded.</Text>
          ) : (
            candidateHints.map((candidate) => {
              const ready = candidate.status === "ready"
              const key = candidateKey(candidate)
              return (
                <div key={key} className="grid gap-2 rounded border border-border-subtle p-2 md:grid-cols-[minmax(0,1fr)_180px]">
                  <Checkbox
                    disabled={!ready}
                    checked={selectedCandidateKeys.has(key)}
                    onChange={(event) => toggleCandidate(candidate, event.target.checked)}
                  >
                    {candidate.provider} {candidate.model}
                  </Checkbox>
                  <div className="text-sm md:text-right">
                    <Tag color={ready ? "green" : "default"}>{candidate.status}</Tag>
                    {candidate.status_reason && (
                      <div className="mt-1 text-xs text-text-muted">
                        {candidate.status_reason}
                      </div>
                    )}
                  </div>
                </div>
              )
            })
          )}
        </div>
      </Card>

      <Card size="small" title="Run review">
        <div className="grid gap-3 md:grid-cols-3">
          <div>
            <Text strong>Top K</Text>
            <InputNumber
              aria-label="Top K"
              className="mt-2 w-full"
              min={1}
              value={normalizedRunConfig.top_k}
              onChange={(value) =>
                applyRunConfig((current) => ({
                  ...current,
                  top_k: parseInteger(value, normalizedRunConfig.top_k)
                }))
              }
            />
          </div>
          <div>
            <Text strong>Hybrid alpha</Text>
            <InputNumber
              aria-label="Hybrid alpha"
              className="mt-2 w-full"
              min={0}
              max={1}
              step={0.05}
              value={normalizedRunConfig.hybrid_alpha}
              onChange={(value) =>
                applyRunConfig((current) => ({
                  ...current,
                  hybrid_alpha: parseNumeric(value, normalizedRunConfig.hybrid_alpha)
                }))
              }
            />
          </div>
          <div>
            <Text strong>Guided source labeling</Text>
            <div className="mt-2">
              <Tag color="blue">
                {normalizedRunConfig.guided_source_labeling ? "enabled" : "disabled"}
              </Tag>
            </div>
          </div>
        </div>
        <div className="mt-3 text-xs text-text-muted">
          {normalizedRunConfig.candidates.length} candidates, {normalizedRunConfig.media_ids.length} corpus media IDs, {editableDataset.length} queries
        </div>
      </Card>
    </div>
  )
}
