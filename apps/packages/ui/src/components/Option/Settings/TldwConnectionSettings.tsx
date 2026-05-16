import { CheckIcon, XMarkIcon } from "@heroicons/react/24/outline"
import {
  Segmented,
  Space,
  Input,
  Form,
  Modal,
  Button,
  Tag
} from "antd"
import type { FormInstance } from "antd"
import React from "react"
import type { TFunction } from "i18next"
import { isFirefoxTarget } from "@/config/platform"
import { Alert } from "@/components/ui/primitives"
import {
  getCoreStatusLabel,
  getRagStatusLabel,
  type CoreStatus,
  type RagStatus
} from "./tldw-connection-status"

export type LoginMethod = "magic-link" | "password"

export type TldwConnectionSettingsProps = {
  t: TFunction
  form: FormInstance
  authMode: "single-user" | "multi-user"
  setAuthMode: (mode: "single-user" | "multi-user") => void
  isLoggedIn: boolean
  setIsLoggedIn: (loggedIn: boolean) => void
  loginMethod: LoginMethod
  setLoginMethod: (method: LoginMethod) => void
  magicEmail: string
  setMagicEmail: (email: string) => void
  magicToken: string
  setMagicToken: (token: string) => void
  magicSent: boolean
  setMagicSent: (sent: boolean) => void
  magicSending: boolean
  testingConnection: boolean
  connectionStatus: "success" | "error" | null
  connectionDetail: string
  coreStatus: CoreStatus
  ragStatus: RagStatus
  onTestConnection: () => void
  onLogin: () => void
  onSendMagicLink: () => void
  onVerifyMagicLink: () => void
  onLogout: () => void
  onGrantSiteAccess: () => void
  onOpenHealthDiagnostics: () => void
}

const coreStatusColor = (status: CoreStatus) => {
  switch (status) {
    case "connected":
      return "green"
    case "failed":
      return "red"
    default:
      return "default"
  }
}

const ragStatusColor = (status: RagStatus) => {
  switch (status) {
    case "healthy":
      return "green"
    case "unhealthy":
      return "red"
    default:
      return "default"
  }
}

export const TldwConnectionSettings = ({
  t,
  form,
  authMode,
  setAuthMode,
  isLoggedIn,
  setIsLoggedIn,
  loginMethod,
  setLoginMethod,
  magicEmail,
  setMagicEmail,
  magicToken,
  setMagicToken,
  magicSent,
  setMagicSent,
  magicSending,
  testingConnection,
  connectionStatus,
  connectionDetail,
  coreStatus,
  ragStatus,
  onTestConnection,
  onLogin,
  onSendMagicLink,
  onVerifyMagicLink,
  onLogout,
  onGrantSiteAccess,
  onOpenHealthDiagnostics
}: TldwConnectionSettingsProps) => {
  return (
    <>
      <Form.Item
        label={t('settings:tldw.fields.serverUrl.label', 'Server URL')}
        name="serverUrl"
        rules={[
          { required: true, message: t('settings:tldw.fields.serverUrl.required', 'Please enter the server URL') as string },
          { type: 'url', message: t('settings:tldw.fields.serverUrl.invalid', 'Please enter a valid URL') as string }
        ]}
        extra={t(
          'settings:tldw.fields.serverUrl.extra',
          'The URL of your tldw_server instance. Default address for local installs: http://127.0.0.1:8000'
        )}
      >
        <Input
          placeholder={t(
            'settings:tldw.fields.serverUrl.placeholder',
            'http://127.0.0.1:8000'
          ) as string}
        />
      </Form.Item>
      <Form.Item
        label={t('settings:tldw.authMode.label', 'Authentication Mode')}
        name="authMode"
        rules={[{ required: true }]}
      >
        <Segmented
          options={[
            { label: t('settings:tldw.authMode.single', 'Single User (API Key)'), value: 'single-user' },
            { label: t('settings:tldw.authMode.multi', 'Multi User (Login)'), value: 'multi-user' }
          ]}
          onChange={(value) => {
            if (authMode !== value) {
              Modal.confirm({
                title: t('settings:tldw.authModeChangeWarning.title', 'Change authentication mode?'),
                content: t('settings:tldw.authModeChangeWarning.content',
                  'Switching authentication modes will clear your current credentials. You will need to re-enter them after saving.'),
                okText: t('common:continue', 'Continue'),
                cancelText: t('common:cancel', 'Cancel'),
                centered: true,
                onOk: () => {
                  setAuthMode(value as 'single-user' | 'multi-user')
                  form.setFieldValue('apiKey', '')
                  form.setFieldValue('username', '')
                  form.setFieldValue('password', '')
                  form.setFieldValue('magicEmail', '')
                  form.setFieldValue('magicToken', '')
                  setMagicEmail('')
                  setMagicToken('')
                  setMagicSent(false)
                  setIsLoggedIn(false)
                },
                onCancel: () => {
                  // Reset the Segmented back to current value
                  form.setFieldValue('authMode', authMode)
                }
              })
            }
          }}
        />
      </Form.Item>
      {authMode === 'single-user' && (
        <Form.Item
          label={t('settings:tldw.fields.apiKey.label', 'API Key')}
          name="apiKey"
          rules={[{ required: true, message: t('settings:tldw.fields.apiKey.required', 'Please enter your API key') }]}
          extra={t('settings:tldw.fields.apiKey.extra', 'Your tldw_server API key for authentication')}
        >
          <Input.Password placeholder={t('settings:tldw.fields.apiKey.placeholder', 'Enter your API key')} />
        </Form.Item>
      )}

      {authMode === 'multi-user' && !isLoggedIn && (
        <>
          <DesignSystemAlert
            title={t('settings:tldw.loginRequired.title', 'Login Required')}
            variant="info"
            role="status"
            aria-live="polite"
            className="mb-4"
          >
            {t('settings:tldw.loginRequired.description', 'Please login with your tldw_server credentials')}
          </DesignSystemAlert>
          <Form.Item
            label={t('settings:tldw.loginMethod.label', 'Login Method')}
          >
            <Segmented
              options={[
                { label: t('settings:tldw.loginMethod.magic', 'Magic link'), value: 'magic-link' },
                { label: t('settings:tldw.loginMethod.password', 'Password'), value: 'password' }
              ]}
              value={loginMethod}
              onChange={(value) => {
                if (value === 'magic-link' || value === 'password') {
                  setLoginMethod(value)
                }
              }}
            />
          </Form.Item>

          {loginMethod === 'password' ? (
            <>
              <Form.Item
                label={t('settings:tldw.fields.username.label', 'Username')}
                name="username"
                rules={[{ required: true, message: t('settings:tldw.fields.username.required', 'Please enter your username') }]}
              >
                <Input placeholder={t('settings:tldw.fields.username.placeholder', 'Enter username')} />
              </Form.Item>

              <Form.Item
                label={t('settings:tldw.fields.password.label', 'Password')}
                name="password"
                rules={[{ required: true, message: t('settings:tldw.fields.password.required', 'Please enter your password') }]}
              >
                <Input.Password placeholder={t('settings:tldw.fields.password.placeholder', 'Enter password')} />
              </Form.Item>

              <Form.Item>
                <Button type="primary" onClick={onLogin}>
                  {t('settings:tldw.buttons.login', 'Login')}
                </Button>
              </Form.Item>
            </>
          ) : (
            <>
              <Form.Item
                label={t('settings:tldw.magicLink.email.label', 'Email')}
              >
                <Input
                  placeholder={t('settings:tldw.magicLink.email.placeholder', 'you@company.com')}
                  value={magicEmail}
                  onChange={(e) => setMagicEmail(e.target.value)}
                />
              </Form.Item>

              <Form.Item
                label={t('settings:tldw.magicLink.token.label', 'Magic link token')}
              >
                <Input
                  placeholder={t('settings:tldw.magicLink.token.placeholder', 'Paste the token from your email')}
                  value={magicToken}
                  onChange={(e) => setMagicToken(e.target.value)}
                />
              </Form.Item>

              <Form.Item>
                <Space>
                  <Button onClick={onSendMagicLink} loading={magicSending}>
                    {magicSent
                      ? t('settings:tldw.magicLink.resend', 'Resend magic link')
                      : t('settings:tldw.magicLink.send', 'Send magic link')}
                  </Button>
                  <Button type="primary" onClick={onVerifyMagicLink}>
                    {t('settings:tldw.magicLink.verify', 'Verify & Login')}
                  </Button>
                </Space>
              </Form.Item>
            </>
          )}
        </>
      )}

      {authMode === 'multi-user' && isLoggedIn && (
        <DesignSystemAlert
          title={t('settings:tldw.loggedIn.title', 'Logged In')}
          variant="success"
          role="status"
          aria-live="polite"
          action={
            {
              label: t('settings:tldw.buttons.logout', 'Logout'),
              onClick: onLogout,
              variant: "danger",
            }
          }
          className="mb-4"
        >
          {t('settings:tldw.loggedIn.description', 'You are currently logged in to tldw_server')}
        </DesignSystemAlert>
      )}

      <Space className="w-full justify-between">
        <Space>
          <Button type="primary" htmlType="submit">
            {t('common:save')}
          </Button>

          <Button
            onClick={onTestConnection}
            loading={testingConnection}
            icon={
              connectionStatus === 'success' ? (
                <CheckIcon className="w-4 h-4 text-success" />
              ) : connectionStatus === 'error' ? (
                <XMarkIcon className="w-4 h-4 text-danger" />
              ) : null
            }
          >
            {t('settings:tldw.buttons.testConnection', 'Test Connection')}
          </Button>

          {!isFirefoxTarget && (
            <Button onClick={onGrantSiteAccess}>
              {t('settings:tldw.buttons.grantSiteAccess', 'Grant Site Access')}
            </Button>
          )}
        </Space>

        <div className="flex flex-col items-start gap-1 ml-4">
          {testingConnection && (
            <span className="text-xs text-text-subtle">
              {t(
                "settings:tldw.connection.checking",
                "Checking connection and RAG health…"
              )}
            </span>
          )}
          {connectionStatus && !testingConnection && (
            <span
              className={`text-sm ${
                connectionStatus === "success"
                  ? "font-medium text-text"
                  : "text-danger"
              }`}>
              {connectionStatus === "success"
                ? t(
                    "settings:tldw.connection.success",
                    "Connection successful!"
                  )
                : t(
                    "settings:tldw.connection.failed",
                    "Connection failed. Please check your settings."
                  )}
            </span>
          )}
          {connectionDetail && connectionStatus !== "success" && (
            <span className="flex flex-wrap items-center gap-2 text-xs text-text-subtle">
              <span>{connectionDetail}</span>
              <button
                type="button"
                className="underline text-primary hover:text-primaryStrong"
                onClick={onOpenHealthDiagnostics}>
                {t(
                  "settings:healthSummary.diagnostics",
                  "Health & diagnostics"
                )}
              </button>
            </span>
          )}
          <div className="flex flex-wrap items-center gap-2 text-xs text-text-muted">
            <span className="font-medium">
              {t("settings:tldw.connection.checksLabel", "Checks")}
            </span>
            <Tag
              color={coreStatusColor(coreStatus)}>
              {getCoreStatusLabel(t, coreStatus)}
            </Tag>
            <Tag
              color={ragStatusColor(ragStatus)}>
              {getRagStatusLabel(t, ragStatus)}
            </Tag>
          </div>
        </div>
      </Space>
    </>
  )
}
