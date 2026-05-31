import React, { useState, useRef, useCallback, useEffect } from "react"
import {
  Card,
  Button,
  Input,
  Form,
  Switch,
  Space,
  Spin,
  message,
  InputNumber,
  Descriptions
} from "antd"
import {
  deriveAdminGuardFromError,
  sanitizeAdminErrorMessage
} from "./admin-error-utils"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"

// ---------------------------------------------------------------------------
// RuntimeConfigPage -- load / edit / save cleanup & registration settings
// ---------------------------------------------------------------------------

const RuntimeConfigPage: React.FC = () => {
  const [adminGuard, setAdminGuard] = useState<"forbidden" | "notFound" | null>(null)

  // Cleanup settings
  const [cleanupData, setCleanupData] = useState<Record<string, any> | null>(null)
  const [cleanupLoading, setCleanupLoading] = useState(false)
  const [cleanupSaving, setCleanupSaving] = useState(false)
  const [cleanupForm] = Form.useForm()

  // Registration settings
  const [registrationData, setRegistrationData] = useState<Record<string, any> | null>(null)
  const [registrationLoading, setRegistrationLoading] = useState(false)
  const [registrationSaving, setRegistrationSaving] = useState(false)
  const [registrationForm] = Form.useForm()

  const initialLoadRef = useRef(false)

  const markAdminGuardFromError = useCallback((err: any) => {
    const guardState = deriveAdminGuardFromError(err)
    if (guardState) setAdminGuard(guardState)
  }, [])

  // ── Cleanup Settings ──

  const loadCleanupSettings = useCallback(async () => {
    setCleanupLoading(true)
    try {
      const data = await tldwClient.getCleanupSettings()
      setCleanupData(data)
      cleanupForm.setFieldsValue(data)
    } catch (err: any) {
      markAdminGuardFromError(err)
      message.error(
        sanitizeAdminErrorMessage(err, "Failed to load cleanup settings")
      )
    } finally {
      setCleanupLoading(false)
    }
  }, [cleanupForm, markAdminGuardFromError])

  const handleSaveCleanup = async () => {
    setCleanupSaving(true)
    try {
      const values = cleanupForm.getFieldsValue()
      const updated = await tldwClient.updateCleanupSettings(values)
      setCleanupData(updated)
      message.success("Cleanup settings saved")
    } catch (err: any) {
      markAdminGuardFromError(err)
      message.error(
        sanitizeAdminErrorMessage(err, "Failed to save cleanup settings")
      )
    } finally {
      setCleanupSaving(false)
    }
  }

  // ── Registration Settings ──

  const loadRegistrationSettings = useCallback(async () => {
    setRegistrationLoading(true)
    try {
      const data = await tldwClient.getRegistrationSettings()
      setRegistrationData(data)
      registrationForm.setFieldsValue(data)
    } catch (err: any) {
      markAdminGuardFromError(err)
      message.error(
        sanitizeAdminErrorMessage(err, "Failed to load registration settings")
      )
    } finally {
      setRegistrationLoading(false)
    }
  }, [registrationForm, markAdminGuardFromError])

  const handleSaveRegistration = async () => {
    setRegistrationSaving(true)
    try {
      const values = registrationForm.getFieldsValue()
      const updated = await tldwClient.updateRegistrationSettings(values)
      setRegistrationData(updated)
      message.success("Registration settings saved")
    } catch (err: any) {
      markAdminGuardFromError(err)
      message.error(
        sanitizeAdminErrorMessage(err, "Failed to save registration settings")
      )
    } finally {
      setRegistrationSaving(false)
    }
  }

  // ── Initial Load ──

  useEffect(() => {
    if (initialLoadRef.current) return
    initialLoadRef.current = true
    loadCleanupSettings()
    loadRegistrationSettings()
  }, [loadCleanupSettings, loadRegistrationSettings])

  // ── Guard Rendering ──

  if (adminGuard === "forbidden") {
    return (
      <DesignSystemAlert variant="error" title="Forbidden">
        You do not have permission to view runtime configuration.
      </DesignSystemAlert>
    )
  }
  if (adminGuard === "notFound") {
    return (
      <DesignSystemAlert variant="warning" title="Not Available">
        The runtime configuration endpoints are not available on this server.
      </DesignSystemAlert>
    )
  }

  return (
    <Space direction="vertical" size="large" style={{ width: "100%" }}>
      {/* Cleanup Settings */}
      <Card
        title="Cleanup Settings"
        extra={
          <Space>
            <Button onClick={loadCleanupSettings} loading={cleanupLoading}>
              Refresh
            </Button>
            <Button type="primary" onClick={handleSaveCleanup} loading={cleanupSaving}>
              Save
            </Button>
          </Space>
        }
      >
        <Spin spinning={cleanupLoading}>
          {cleanupData ? (
            <Form form={cleanupForm} layout="vertical" initialValues={cleanupData}>
              <Form.Item name="soft_delete_retention_days" label="Soft Delete Retention (days)">
                <InputNumber min={1} max={365} style={{ width: "100%" }} />
              </Form.Item>
              <Form.Item name="orphan_cleanup_enabled" label="Orphan Cleanup Enabled" valuePropName="checked">
                <Switch />
              </Form.Item>
              <Form.Item name="orphan_cleanup_interval_hours" label="Orphan Cleanup Interval (hours)">
                <InputNumber min={1} max={720} style={{ width: "100%" }} />
              </Form.Item>
              <Form.Item name="log_retention_days" label="Log Retention (days)">
                <InputNumber min={1} max={365} style={{ width: "100%" }} />
              </Form.Item>
            </Form>
          ) : (
            <Descriptions>
              <Descriptions.Item label="Status">Loading...</Descriptions.Item>
            </Descriptions>
          )}
        </Spin>
      </Card>

      {/* Registration Settings */}
      <Card
        title="Registration Settings"
        extra={
          <Space>
            <Button onClick={loadRegistrationSettings} loading={registrationLoading}>
              Refresh
            </Button>
            <Button type="primary" onClick={handleSaveRegistration} loading={registrationSaving}>
              Save
            </Button>
          </Space>
        }
      >
        <Spin spinning={registrationLoading}>
          {registrationData ? (
            <Form form={registrationForm} layout="vertical" initialValues={registrationData}>
              <Form.Item name="open_registration" label="Open Registration" valuePropName="checked">
                <Switch />
              </Form.Item>
              <Form.Item name="require_email_verification" label="Require Email Verification" valuePropName="checked">
                <Switch />
              </Form.Item>
              <Form.Item name="default_role" label="Default Role">
                <Input />
              </Form.Item>
              <Form.Item name="max_users" label="Maximum Users (0 = unlimited)">
                <InputNumber min={0} style={{ width: "100%" }} />
              </Form.Item>
            </Form>
          ) : (
            <Descriptions>
              <Descriptions.Item label="Status">Loading...</Descriptions.Item>
            </Descriptions>
          )}
        </Spin>
      </Card>
    </Space>
  )
}

export default RuntimeConfigPage
