import React from "react"
import { useTranslation } from "react-i18next"
import { Button, Checkbox, Input, Select, Tag, Typography } from "antd"

import { tldwClient } from "@/services/tldw/TldwApiClient"
import { toAllowedPath } from "@/services/tldw/path-utils"

type PersonaScopeRuleType =
  | "conversation_id"
  | "character_id"
  | "media_id"
  | "media_tag"
  | "note_id"

type PersonaScopeRule = {
  rule_type: PersonaScopeRuleType
  rule_value: string
  include: boolean
}

type ScopesPanelProps = {
  selectedPersonaId?: string
  selectedPersonaName: string
}

const SCOPE_RULE_TYPE_OPTIONS: Array<{
  value: PersonaScopeRuleType
  label: string
}> = [
  { value: "conversation_id", label: "Conversation" },
  { value: "character_id", label: "Character" },
  { value: "media_id", label: "Media ID" },
  { value: "media_tag", label: "Media tag" },
  { value: "note_id", label: "Note" }
]

const emptyScopeRule = (): PersonaScopeRule => ({
  rule_type: "media_tag",
  rule_value: "",
  include: true
})

export const ScopesPanel: React.FC<ScopesPanelProps> = ({
  selectedPersonaId = "",
  selectedPersonaName
}) => {
  const { t } = useTranslation(["sidepanel", "common"])
  const [rules, setRules] = React.useState<PersonaScopeRule[]>([])
  const [loading, setLoading] = React.useState(false)
  const [saving, setSaving] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [notice, setNotice] = React.useState<string | null>(null)

  const normalizedPersonaId = String(selectedPersonaId || "").trim()

  React.useEffect(() => {
    let cancelled = false
    setError(null)
    setNotice(null)
    if (!normalizedPersonaId) {
      setRules([])
      setLoading(false)
      return
    }

    const loadRules = async () => {
      setLoading(true)
      try {
        const response = await tldwClient.fetchWithAuth(
          toAllowedPath(
            `/api/v1/persona/profiles/${encodeURIComponent(
              normalizedPersonaId
            )}/scope-rules`
          )
        )
        if (!response.ok) {
          throw new Error(response.error || "Failed to load scope rules")
        }
        const payload = await response.json()
        if (!cancelled) {
          setRules(Array.isArray(payload?.rules) ? payload.rules : [])
        }
      } catch (err: any) {
        if (!cancelled) {
          setError(String(err?.message || "Failed to load scope rules"))
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void loadRules()
    return () => {
      cancelled = true
    }
  }, [normalizedPersonaId])

  const updateRule = (index: number, patch: Partial<PersonaScopeRule>) => {
    setRules((current) =>
      current.map((rule, ruleIndex) =>
        ruleIndex === index ? { ...rule, ...patch } : rule
      )
    )
  }

  const removeRule = (index: number) => {
    setRules((current) => current.filter((_, ruleIndex) => ruleIndex !== index))
  }

  const saveRules = async () => {
    setError(null)
    setNotice(null)
    if (loading || saving) return
    if (!normalizedPersonaId) {
      setError("Select a persona before editing scope rules.")
      return
    }
    const normalizedRules = rules.map((rule) => ({
      rule_type: rule.rule_type,
      rule_value: String(rule.rule_value || "").trim(),
      include: Boolean(rule.include)
    }))
    if (normalizedRules.some((rule) => !rule.rule_value)) {
      setError("Rule value is required.")
      return
    }

    setSaving(true)
    try {
      const response = await tldwClient.fetchWithAuth(
        toAllowedPath(
          `/api/v1/persona/profiles/${encodeURIComponent(
            normalizedPersonaId
          )}/scope-rules`
        ),
        {
          method: "PUT",
          body: { rules: normalizedRules }
        }
      )
      if (!response.ok) {
        throw new Error(response.error || "Failed to save scope rules")
      }
      const payload = await response.json()
      setRules(Array.isArray(payload?.rules) ? payload.rules : normalizedRules)
      setNotice("Scope rules saved.")
    } catch (err: any) {
      setError(String(err?.message || "Failed to save scope rules"))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex items-center justify-between gap-2">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
            {t("sidepanel:personaGarden.scopes.heading", {
              defaultValue: "Scoped Access"
            })}
          </div>
          <div className="mt-1 text-sm font-medium text-text">
            {selectedPersonaName ||
              t("sidepanel:personaGarden.scopes.selectedPersonaFallback", {
                defaultValue: "Selected persona"
              })}
          </div>
        </div>
        <Button
          data-testid="persona-scope-save-button"
          size="small"
          type="primary"
          loading={saving}
          disabled={!normalizedPersonaId || loading || saving}
          onClick={() => {
            void saveRules()
          }}
        >
          {t("common:save", { defaultValue: "Save" })}
        </Button>
      </div>

      <p className="mt-2 text-xs text-text-muted">
        {t("sidepanel:personaGarden.scopes.description", {
          defaultValue:
            "Limit persona retrieval and session context to explicitly allowed IDs or tags. These rules narrow scope; they do not grant new server permissions."
        })}
      </p>

      {!normalizedPersonaId ? (
        <Typography.Text type="secondary" className="text-xs">
          {t("sidepanel:personaGarden.scopes.noneSelected", {
            defaultValue: "Select a persona to edit scope rules."
          })}
        </Typography.Text>
      ) : null}

      {error ? (
        <div className="mt-2 text-xs text-danger" role="alert">
          {error}
        </div>
      ) : null}
      {notice ? <div className="mt-2 text-xs text-success">{notice}</div> : null}

      <div className="mt-3 space-y-2">
        {loading ? (
          <Typography.Text type="secondary" className="text-xs">
            {t("common:loading", { defaultValue: "Loading..." })}
          </Typography.Text>
        ) : null}
        {!loading && normalizedPersonaId && rules.length === 0 ? (
          <Typography.Text type="secondary" className="text-xs">
            {t("sidepanel:personaGarden.scopes.empty", {
              defaultValue: "No scope rules yet."
            })}
          </Typography.Text>
        ) : null}
        {rules.map((rule, index) => (
          <div
            key={`${rule.rule_type}-${index}`}
            className="rounded-md border border-border bg-background p-2"
          >
            <div className="grid grid-cols-1 gap-2 md:grid-cols-[150px_1fr_auto_auto]">
              <Select
                size="small"
                value={rule.rule_type}
                options={SCOPE_RULE_TYPE_OPTIONS}
                onChange={(value) =>
                  updateRule(index, { rule_type: value as PersonaScopeRuleType })
                }
              />
              <Input
                data-testid={`persona-scope-rule-value-${index}`}
                size="small"
                value={rule.rule_value}
                placeholder={t("sidepanel:personaGarden.scopes.valuePlaceholder", {
                  defaultValue: "ID or tag"
                })}
                onChange={(event) =>
                  updateRule(index, { rule_value: event.target.value })
                }
              />
              <Checkbox
                data-testid={`persona-scope-rule-include-${index}`}
                checked={rule.include}
                onChange={(event) =>
                  updateRule(index, { include: event.target.checked })
                }
              >
                {rule.include ? <Tag color="green">include</Tag> : <Tag>exclude</Tag>}
              </Checkbox>
              <Button size="small" onClick={() => removeRule(index)}>
                {t("common:remove", { defaultValue: "Remove" })}
              </Button>
            </div>
          </div>
        ))}
      </div>

      <Button
        data-testid="persona-scope-add-rule"
        className="mt-3"
        size="small"
        disabled={!normalizedPersonaId}
        onClick={() => setRules((current) => [...current, emptyScopeRule()])}
      >
        {t("sidepanel:personaGarden.scopes.addRule", {
          defaultValue: "Add scope rule"
        })}
      </Button>
    </div>
  )
}
