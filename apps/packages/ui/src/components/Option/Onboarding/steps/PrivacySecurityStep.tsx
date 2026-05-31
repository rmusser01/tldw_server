import React from "react"
import { AlertTriangle, CheckCircle2, Shield } from "lucide-react"

import type { FirstRunMetadata } from "@/types/setup-onboarding"

type PrivacySecurityStepProps = {
  metadata: FirstRunMetadata | null
  onBack: () => void
  onContinue: () => void
  saving?: boolean
}

const accessLabel = (value?: string | null): string => {
  switch (value) {
    case "local":
      return "Local browser access"
    case "lan":
      return "LAN browser access"
    case "remote":
      return "Remote browser access"
    default:
      return "Unknown browser access"
  }
}

export function PrivacySecurityStep({
  metadata,
  onBack,
  onContinue,
  saving = false
}: PrivacySecurityStepProps) {
  const [acknowledged, setAcknowledged] = React.useState(false)
  const browserAccess = metadata?.connection?.browser_access
  const shouldWarnRemote =
    metadata?.remote_setup_enabled || browserAccess === "lan" || browserAccess === "remote"

  return (
    <section aria-labelledby="privacy-security-title" className="space-y-5">
      <div className="flex items-start gap-3">
        <span className="inline-flex size-10 items-center justify-center rounded-md bg-surface2 text-primary">
          <Shield className="size-5" aria-hidden="true" />
        </span>
        <div>
          <h2
            id="privacy-security-title"
            className="text-lg font-semibold text-text"
          >
            Privacy and security
          </h2>
          <p className="mt-1 text-sm text-text-muted">
            Confirm the access and secret-storage behavior for this setup.
          </p>
        </div>
      </div>

      {shouldWarnRemote ? (
        <div className="flex gap-3 rounded-md border border-warning/40 bg-warning/10 px-4 py-3 text-sm text-text">
          <AlertTriangle className="mt-0.5 size-4 shrink-0 text-warning" aria-hidden="true" />
          <p>
            This setup request appears reachable beyond localhost. Continue only
            if you intentionally enabled remote setup access.
          </p>
        </div>
      ) : null}

      <dl className="grid gap-3 md:grid-cols-2">
        <div className="rounded-md border border-border bg-surface px-4 py-3">
          <dt className="text-xs font-medium uppercase tracking-normal text-text-muted">
            Auth mode
          </dt>
          <dd className="mt-1 text-sm font-medium text-text">
            {metadata?.auth_mode || "single_user"}
          </dd>
          <p className="mt-2 text-xs text-text-muted">
            This flow assumes solo single-user setup. Multi-user mode should use
            the operator guide.
          </p>
        </div>
        <div className="rounded-md border border-border bg-surface px-4 py-3">
          <dt className="text-xs font-medium uppercase tracking-normal text-text-muted">
            Browser access
          </dt>
          <dd className="mt-1 text-sm font-medium text-text">
            {accessLabel(browserAccess)}
          </dd>
          <p className="mt-2 text-xs text-text-muted">
            Backend metadata is used for this classification.
          </p>
        </div>
        <div className="rounded-md border border-border bg-surface px-4 py-3">
          <dt className="text-xs font-medium uppercase tracking-normal text-text-muted">
            Single-user auth
          </dt>
          <dd className="mt-1 text-sm font-medium text-text">
            {metadata?.bundled_single_user_auth_available
              ? "Bundled auth available"
              : "Manual API key required"}
          </dd>
          <p className="mt-2 text-xs text-text-muted">
            The backend decides whether the WebUI can use the bundled local
            single-user key.
          </p>
        </div>
        <div className="rounded-md border border-border bg-surface px-4 py-3">
          <dt className="text-xs font-medium uppercase tracking-normal text-text-muted">
            Provider secrets
          </dt>
          <dd className="mt-1 flex items-center gap-2 text-sm font-medium text-text">
            <CheckCircle2 className="size-4 text-success" aria-hidden="true" />
            Stored by backend, returned masked
          </dd>
          <p className="mt-2 text-xs text-text-muted">
            Raw API keys are submitted only to the setup endpoint and are not
            displayed again by this UI.
          </p>
        </div>
      </dl>

      <label className="flex items-start gap-3 rounded-md border border-border bg-surface px-4 py-3 text-sm text-text">
        <input
          type="checkbox"
          checked={acknowledged}
          onChange={(event) => setAcknowledged(event.currentTarget.checked)}
          className="mt-1"
        />
        <span>
          I understand local or remote setup access and provider secret storage.
        </span>
      </label>

      <div className="flex flex-wrap justify-between gap-2">
        <button
          type="button"
          onClick={onBack}
          disabled={saving}
          className="rounded-md border border-border bg-surface px-3 py-2 text-sm font-medium text-text hover:bg-surface2"
        >
          Back
        </button>
        <button
          type="button"
          disabled={!acknowledged || saving}
          onClick={onContinue}
          className="rounded-md bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground disabled:cursor-not-allowed disabled:opacity-50"
        >
          Continue
        </button>
      </div>
    </section>
  )
}
