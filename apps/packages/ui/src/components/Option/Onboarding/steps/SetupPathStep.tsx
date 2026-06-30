import React from "react"
import { Box, Laptop, Users } from "lucide-react"

type SetupPath = "docker" | "local" | "multi_user"

type SetupPathStepProps = {
  onSelect: (path: SetupPath) => void
}

const pathButtonClass =
  "flex min-h-28 w-full flex-col items-start gap-3 rounded-md border border-border bg-surface px-4 py-4 text-left text-text transition-colors hover:border-primary/60 hover:bg-surface2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60"

export function SetupPathStep({ onSelect }: SetupPathStepProps) {
  return (
    <section aria-labelledby="setup-path-title" className="space-y-4">
      <div>
        <h2 id="setup-path-title" className="text-lg font-semibold text-text">
          Choose your setup path
        </h2>
        <p className="mt-1 text-sm text-text-muted">
          Pick the path that matches how this server will run.
        </p>
      </div>
      <div className="grid gap-3 md:grid-cols-3">
        <button
          type="button"
          onClick={() => onSelect("docker")}
          className={pathButtonClass}
        >
          <Box className="size-5 text-primary" aria-hidden="true" />
          <span className="text-sm font-semibold">Solo, Docker</span>
          <span className="text-xs text-text-muted">
            Single-user server from the container profile.
          </span>
        </button>
        <button
          type="button"
          onClick={() => onSelect("local")}
          className={pathButtonClass}
        >
          <Laptop className="size-5 text-primary" aria-hidden="true" />
          <span className="text-sm font-semibold">Solo, local install</span>
          <span className="text-xs text-text-muted">
            Single-user server from a local Python environment.
          </span>
        </button>
        <button
          type="button"
          onClick={() => onSelect("multi_user")}
          className={pathButtonClass}
        >
          <Users className="size-5 text-primary" aria-hidden="true" />
          <span className="text-sm font-semibold">Multi-user</span>
          <span className="text-xs text-text-muted">
            Shared server with user accounts and admin setup.
          </span>
        </button>
      </div>
    </section>
  )
}
