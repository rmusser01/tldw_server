type UnknownRecord = Record<string, unknown>

const isRecord = (value: unknown): value is UnknownRecord =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const firstNonEmptyString = (...values: unknown[]): string => {
  for (const value of values) {
    if (typeof value === "string" && value.trim().length > 0) {
      return value.trim()
    }
  }
  return ""
}

const extractNestedContent = (value: unknown): string => {
  if (typeof value === "string") return value.trim()
  if (!isRecord(value)) return ""

  return firstNonEmptyString(
    value.text,
    value.content,
    value.raw_text,
    value.rawText,
    value.transcript,
    value.summary
  )
}

const extractAnalysisEntry = (value: unknown): string => {
  if (typeof value === "string") return value.trim()
  if (!isRecord(value)) return ""

  return firstNonEmptyString(
    value.content,
    value.text,
    value.summary,
    value.analysis_content,
    value.analysisContent,
    value.analysis
  )
}

const getVersionNumber = (value: UnknownRecord): number => {
  const raw = value.version_number ?? value.versionNumber ?? value.version ?? value.id
  const parsed = Number(raw)
  return Number.isFinite(parsed) ? parsed : Number.NEGATIVE_INFINITY
}

const findLatestVersion = (detail: UnknownRecord): UnknownRecord | null => {
  if (isRecord(detail.latest_version)) return detail.latest_version
  if (isRecord(detail.latestVersion)) return detail.latestVersion

  const versions = Array.isArray(detail.versions)
    ? detail.versions.filter(isRecord)
    : []
  if (versions.length === 0) return null

  return versions.reduce((latest, candidate) =>
    getVersionNumber(candidate) > getVersionNumber(latest) ? candidate : latest
  )
}

export const extractMediaDetailContent = (detail: unknown): string => {
  if (typeof detail === "string") return detail.trim()
  if (!isRecord(detail)) return ""

  const fromContentObject = extractNestedContent(detail.content)
  if (fromContentObject) return fromContentObject

  const fromRoot = firstNonEmptyString(
    detail.text,
    detail.transcript,
    detail.raw_text,
    detail.rawText,
    detail.raw_content,
    detail.rawContent,
    detail.summary
  )
  if (fromRoot) return fromRoot

  const latestVersion = isRecord(detail.latest_version)
    ? detail.latest_version
    : isRecord(detail.latestVersion)
      ? detail.latestVersion
      : null
  if (latestVersion) {
    const fromLatestContent = extractNestedContent(latestVersion.content)
    if (fromLatestContent) return fromLatestContent

    const fromLatest = firstNonEmptyString(
      latestVersion.text,
      latestVersion.transcript,
      latestVersion.raw_text,
      latestVersion.rawText,
      latestVersion.summary
    )
    if (fromLatest) return fromLatest
  }

  const data = isRecord(detail.data) ? detail.data : null
  if (data) {
    const fromDataContent = extractNestedContent(data.content)
    if (fromDataContent) return fromDataContent

    const fromData = firstNonEmptyString(
      data.text,
      data.transcript,
      data.raw_text,
      data.rawText,
      data.summary
    )
    if (fromData) return fromData
  }

  return ""
}

export const extractMediaDetailAnalysis = (detail: unknown): string => {
  if (!isRecord(detail)) return ""

  const processing = isRecord(detail.processing) ? detail.processing : null
  const fromRoot = firstNonEmptyString(
    processing?.analysis,
    detail.analysis,
    detail.analysis_content,
    detail.analysisContent
  )
  if (fromRoot) return fromRoot

  if (Array.isArray(detail.analyses)) {
    for (const entry of detail.analyses) {
      const analysis = extractAnalysisEntry(entry)
      if (analysis) return analysis
    }
  }

  const latestVersion = findLatestVersion(detail)
  if (latestVersion) {
    const fromVersion = firstNonEmptyString(
      latestVersion.analysis_content,
      latestVersion.analysisContent,
      latestVersion.analysis
    )
    if (fromVersion) return fromVersion
  }

  return firstNonEmptyString(detail.summary)
}
