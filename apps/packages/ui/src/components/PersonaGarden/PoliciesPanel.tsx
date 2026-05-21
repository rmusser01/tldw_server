import React from "react"
import { useTranslation } from "react-i18next"
import { Button, Checkbox, Input, Select, Tag, Typography } from "antd"

import { tldwClient } from "@/services/tldw/TldwApiClient"
import { toAllowedPath } from "@/services/tldw/path-utils"

import { McpToolPicker } from "./McpToolPicker"

type PersonaPolicyRuleKind = "mcp_tool" | "skill"

type PersonaPolicyRule = {
  rule_kind: PersonaPolicyRuleKind
  rule_name: string
  allowed: boolean
  require_confirmation: boolean
  max_calls_per_turn: number | null
}

type PoliciesPanelProps = {
  selectedPersonaId?: string
  personaCapabilities?: string[]
  personaDefaultTools?: string[]
  hasPendingPlan: boolean
}

const POLICY_RULE_KIND_OPTIONS: Array<{
  value: PersonaPolicyRuleKind
  label: string
}> = [
  { value: "mcp_tool", label: "MCP tool" },
  { value: "skill", label: "Skill" }
]

const emptyPolicyRule = (): PersonaPolicyRule => ({
  rule_kind: "mcp_tool",
  rule_name: "",
  allowed: true,
  require_confirmation: true,
  max_calls_per_turn: null
})

export const PoliciesPanel: React.FC<PoliciesPanelProps> = ({
  selectedPersonaId = "",
  personaCapabilities = [],
  personaDefaultTools = [],
  hasPendingPlan
}) => {
  const { t } = useTranslation(["sidepanel", "common"])
  const [rules, setRules] = React.useState<PersonaPolicyRule[]>([])
  const [loading, setLoading] = React.useState(false)
  const [saving, setSaving] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [notice, setNotice] = React.useState<string | null>(null)

  const normalizedPersonaId = String(selectedPersonaId || "").trim()

  React.useEffect(() => {
    let cancelled = false
    setRules([])
    setError(null)
    setNotice(null)
    if (!normalizedPersonaId) return

    const loadRules = async () => {
      setLoading(true)
      try {
        const response = await tldwClient.fetchWithAuth(
          toAllowedPath(
            `/api/v1/persona/profiles/${encodeURIComponent(
              normalizedPersonaId
            )}/policy-rules`
          )
        )
        if (!response.ok) {
          throw new Error(response.error || "Failed to load policy rules")
        }
        const payload = await response.json()
        if (!cancelled) {
          setRules(Array.isArray(payload?.rules) ? payload.rules : [])
        }
      } catch (err: any) {
        if (!cancelled) {
          setError(String(err?.message || "Failed to load policy rules"))
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

  const updateRule = (index: number, patch: Partial<PersonaPolicyRule>) => {
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
    if (!normalizedPersonaId) {
      setError("Select a persona before editing policy rules.")
      return
    }

    const normalizedRules = rules.map((rule) => {
      const parsedMaxCalls = Number(rule.max_calls_per_turn || 0)
      return {
        rule_kind: rule.rule_kind,
        rule_name: String(rule.rule_name || "").trim(),
        allowed: Boolean(rule.allowed),
        require_confirmation: Boolean(rule.require_confirmation),
        max_calls_per_turn:
          Number.isFinite(parsedMaxCalls) && parsedMaxCalls > 0
            ? parsedMaxCalls
            : null
      }
    })
    if (normalizedRules.some((rule) => !rule.rule_name)) {
      setError("Rule name is required.")
      return
    }

    setSaving(true)
    try {
      const response = await tldwClient.fetchWithAuth(
        toAllowedPath(
          `/api/v1/persona/profiles/${encodeURIComponent(
            normalizedPersonaId
          )}/policy-rules`
        ),
        {
          method: "PUT",
          body: { rules: normalizedRules }
        }
      )
      if (!response.ok) {
        throw new Error(response.error || "Failed to save policy rules")
      }
      const payload = await response.json()
      setRules(Array.isArray(payload?.rules) ? payload.rules : normalizedRules)
      setNotice("Policy rules saved.")
    } catch (err: any) {
      setError(String(err?.message || "Failed to save policy rules"))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
          {t("sidepanel:personaGarden.policies.heading", {
            defaultValue: "Tool Policies"
          })}
        </div>
        <Button
          data-testid="persona-policy-save-button"
          size="small"
          type="primary"
          loading={saving}
          disabled={!normalizedPersonaId}
          onClick={() => {
            void saveRules()
          }}
        >
          {t("common:save", { defaultValue: "Save" })}
        </Button>
      </div>

      <p className="mt-2 text-xs text-text-muted">
        {t("sidepanel:personaGarden.policies.description", {
          defaultValue:
            "Choose which already-authorized tools or skills this persona may use, and whether a live turn must ask before executing them."
        })}
      </p>
      <div className="text-xs text-text-muted">
        {hasPendingPlan
          ? t("sidepanel:personaGarden.policies.pendingPlan", {
              defaultValue: "A pending tool plan is available on the Live Session tab."
            })
          : t("sidepanel:personaGarden.policies.noPendingPlan", {
              defaultValue: "No pending tool plan right now."
            })}
      </div>

      {personaCapabilities.length > 0 || personaDefaultTools.length > 0 ? (
        <div
          data-testid="persona-policy-catalog-context"
          className="mt-3 rounded-md border border-border bg-background px-3 py-2"
        >
          {personaDefaultTools.length > 0 ? (
            <div className="space-y-1">
              <div className="text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
                {t("sidepanel:personaGarden.policies.defaultTools", {
                  defaultValue: "Persona defaults"
                })}
              </div>
              <div className="flex flex-wrap gap-1">
                {personaDefaultTools.map((toolName) => (
                  <Tag key={toolName} color="blue">
                    {toolName}
                  </Tag>
                ))}
              </div>
            </div>
          ) : null}
          {personaCapabilities.length > 0 ? (
            <div className={personaDefaultTools.length > 0 ? "mt-2 space-y-1" : "space-y-1"}>
              <div className="text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
                {t("sidepanel:personaGarden.policies.capabilities", {
                  defaultValue: "Capabilities"
                })}
              </div>
              <div className="flex flex-wrap gap-1">
                {personaCapabilities.map((capability) => (
                  <Tag key={capability}>{capability}</Tag>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      ) : null}

      {!normalizedPersonaId ? (
        <Typography.Text type="secondary" className="text-xs">
          {t("sidepanel:personaGarden.policies.noneSelected", {
            defaultValue: "Select a persona to edit policy rules."
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
            {t("sidepanel:personaGarden.policies.empty", {
              defaultValue: "No policy rules yet."
            })}
          </Typography.Text>
        ) : null}
        {rules.map((rule, index) => (
          <div
            key={`${rule.rule_kind}-${index}`}
            className="rounded-md border border-border bg-background p-2"
          >
            <div className="grid grid-cols-1 gap-2 md:grid-cols-[130px_1fr_120px_150px_110px_auto]">
              <Select
                size="small"
                value={rule.rule_kind}
                options={POLICY_RULE_KIND_OPTIONS}
                onChange={(value) =>
                  updateRule(index, { rule_kind: value as PersonaPolicyRuleKind })
                }
              />
              {rule.rule_kind === "mcp_tool" ? (
                <McpToolPicker
                  value={rule.rule_name}
                  onChange={(value) => updateRule(index, { rule_name: value })}
                />
              ) : (
                <Input
                  data-testid={`persona-policy-rule-name-${index}`}
                  size="small"
                  value={rule.rule_name}
                  placeholder={t("sidepanel:personaGarden.policies.namePlaceholder", {
                    defaultValue: "Tool or skill name"
                  })}
                  onChange={(event) =>
                    updateRule(index, { rule_name: event.target.value })
                  }
                />
              )}
              <Checkbox
                data-testid={`persona-policy-rule-allowed-${index}`}
                checked={rule.allowed}
                onChange={(event) =>
                  updateRule(index, { allowed: event.target.checked })
                }
              >
                {rule.allowed ? <Tag color="green">allow</Tag> : <Tag color="red">deny</Tag>}
              </Checkbox>
              <Checkbox
                data-testid={`persona-policy-rule-confirm-${index}`}
                checked={rule.require_confirmation}
                onChange={(event) =>
                  updateRule(index, { require_confirmation: event.target.checked })
                }
              >
                {t("sidepanel:personaGarden.policies.confirm", {
                  defaultValue: "Confirm"
                })}
              </Checkbox>
              <Input
                data-testid={`persona-policy-rule-max-calls-${index}`}
                size="small"
                type="number"
                min={1}
                value={rule.max_calls_per_turn ?? ""}
                placeholder={t("sidepanel:personaGarden.policies.maxCalls", {
                  defaultValue: "Max calls"
                })}
                onChange={(event) =>
                  updateRule(index, {
                    max_calls_per_turn: event.target.value
                      ? Number(event.target.value)
                      : null
                  })
                }
              />
              <Button size="small" onClick={() => removeRule(index)}>
                {t("common:remove", { defaultValue: "Remove" })}
              </Button>
            </div>
          </div>
        ))}
      </div>

      <Button
        data-testid="persona-policy-add-rule"
        className="mt-3"
        size="small"
        disabled={!normalizedPersonaId}
        onClick={() => setRules((current) => [...current, emptyPolicyRule()])}
      >
        {t("sidepanel:personaGarden.policies.addRule", {
          defaultValue: "Add policy rule"
        })}
      </Button>
    </div>
  )
}
