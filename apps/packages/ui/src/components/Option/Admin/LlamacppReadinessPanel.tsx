import React from "react"
import { Alert, Card, Descriptions, Space, Tag, Typography } from "antd"
import type { LlamacppConfigResponse } from "@/types/llamacpp-admin"

const { Text } = Typography

interface LlamacppReadinessPanelProps {
  config: LlamacppConfigResponse | null
  loading?: boolean
}

const formatValue = (value: string | number | boolean | null | undefined) =>
  value === undefined || value === null || value === "" ? "Not set" : String(value)

export const LlamacppReadinessPanel: React.FC<LlamacppReadinessPanelProps> = ({
  config,
  loading = false
}) => {
  const saved = config?.saved_config
  const active = config?.active_config
  const envOverrides = Object.entries(config?.env_overrides || {}).filter(
    ([, overridden]) => overridden
  )

  return (
    <Card title="Readiness" loading={loading}>
      {!config ? (
        <Text type="secondary">Saved llama.cpp configuration is unavailable.</Text>
      ) : (
        <Space orientation="vertical" size="middle" className="w-full">
          <Space wrap>
            <Tag color={saved?.enabled ? "green" : "default"}>
              {saved?.enabled ? "Saved enabled" : "Saved disabled"}
            </Tag>
            <Tag color={active?.handler_configured ? "green" : "orange"}>
              {active?.handler_configured
                ? "Active handler configured"
                : "Active handler not configured"}
            </Tag>
            {config.restart_required && <Tag color="orange">Restart required</Tag>}
          </Space>

          <Descriptions size="small" bordered column={{ xs: 1, md: 2 }}>
            <Descriptions.Item label="Saved models directory">
              <Text code>{formatValue(saved?.models_dir)}</Text>
            </Descriptions.Item>
            <Descriptions.Item label="Saved executable">
              <Text code>{formatValue(saved?.executable_path)}</Text>
            </Descriptions.Item>
            <Descriptions.Item label="Saved host">
              <Text code>{formatValue(saved?.default_host)}</Text>
            </Descriptions.Item>
            <Descriptions.Item label="Saved port">
              <Text code>{formatValue(saved?.default_port)}</Text>
            </Descriptions.Item>
            <Descriptions.Item label="Active runtime">
              {active?.handler_configured ? (
                <Space wrap>
                  <Tag color={active.enabled ? "green" : "default"}>
                    {active.enabled ? "enabled" : "disabled"}
                  </Tag>
                  {active.active_pid && <Tag>pid {active.active_pid}</Tag>}
                </Space>
              ) : (
                <Text type="secondary">No active handler configuration.</Text>
              )}
            </Descriptions.Item>
            <Descriptions.Item label="Active model">
              <Text code>{formatValue(active?.active_model)}</Text>
            </Descriptions.Item>
          </Descriptions>

          {config.restart_required && (
            <Alert
              type="warning"
              showIcon
              title="API server restart required"
              description={
                <Space wrap size="small">
                  {config.restart_reasons.length > 0 ? (
                    config.restart_reasons.map((reason) => (
                      <Tag key={reason} color="orange">
                        {reason}
                      </Tag>
                    ))
                  ) : (
                    <Text type="secondary">Saved config differs from active runtime.</Text>
                  )}
                </Space>
              }
            />
          )}

          {envOverrides.length > 0 && (
            <Alert
              type="warning"
              showIcon
              title="Environment overrides are active"
              description={
                <Space wrap size="small">
                  {envOverrides.map(([key]) => (
                    <Tag key={key} color="gold">
                      {key} override
                    </Tag>
                  ))}
                </Space>
              }
            />
          )}

          {config.warnings.map((warning) => (
            <Alert key={warning} type="info" showIcon title={warning} />
          ))}
        </Space>
      )}
    </Card>
  )
}

export default LlamacppReadinessPanel
