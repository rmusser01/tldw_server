export type InspectorRailProps = {
  scopeLabel: string
  stagedSourceCount: number
  stagedSourceTitles: string[]
  selectedModelLabel: string
  selectedPersonaLabel: string | null
  backendAvailable: boolean
  streaming: boolean
}

const panelClass = "rounded-md border border-border bg-surface px-3 py-2"
const headingClass = "text-[11px] font-semibold text-text-muted"
const valueClass = "mt-1 text-sm font-medium text-text"
const mutedClass = "mt-1 text-xs text-text-muted"

const getRuntimeLabel = (backendAvailable: boolean, streaming: boolean) => {
  if (!backendAvailable) {
    return "Server unavailable"
  }

  return streaming ? "Streaming" : "Ready"
}

export const InspectorRail = ({
  scopeLabel,
  stagedSourceCount,
  stagedSourceTitles,
  selectedModelLabel,
  selectedPersonaLabel,
  backendAvailable,
  streaming
}: InspectorRailProps) => {
  const runtimeLabel = getRuntimeLabel(backendAvailable, streaming)

  return (
    <aside
      aria-label="Chat workspace inspector"
      className="flex min-w-0 flex-col gap-2 text-sm"
    >
      <section className={panelClass}>
        <h2 className={headingClass}>Scope</h2>
        <p className={valueClass}>{scopeLabel}</p>
      </section>

      <section className={panelClass}>
        <h2 className={headingClass}>Sources</h2>
        <p className={valueClass}>
          {stagedSourceCount} source{stagedSourceCount === 1 ? "" : "s"} staged
        </p>
        {stagedSourceTitles.length > 0 ? (
          <ul className="mt-2 space-y-1">
            {stagedSourceTitles.map((title) => (
              <li key={title} className="min-w-0 break-words text-xs text-text">
                {title}
              </li>
            ))}
          </ul>
        ) : (
          <p className={mutedClass}>No sources staged</p>
        )}
      </section>

      <section className={panelClass}>
        <h2 className={headingClass}>Model / Persona</h2>
        <p className={valueClass}>{selectedModelLabel}</p>
        <p className={mutedClass}>{selectedPersonaLabel ?? "No persona selected"}</p>
      </section>

      <section className={panelClass}>
        <h2 className={headingClass}>Approvals</h2>
        <p className={valueClass}>Not configured</p>
      </section>

      <section className={panelClass}>
        <h2 className={headingClass}>Task Progress</h2>
        <p className={valueClass}>No active task</p>
      </section>

      <section className={panelClass}>
        <h2 className={headingClass}>Runtime</h2>
        <p className={valueClass}>{runtimeLabel}</p>
      </section>
    </aside>
  )
}
