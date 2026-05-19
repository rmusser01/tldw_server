import React, { useState } from "react"
import { Card, Button, Input, Spin, Empty, Tag, message } from "antd"
import { Lock, Download } from "lucide-react"
import { useNavigate } from "react-router-dom"
import {
  usePublicPreview,
  useVerifySharePassword,
  useImportFromToken,
} from "@/hooks/useSharing"
import { ACCESS_LEVEL_LABELS, type AccessLevel } from "@/types/sharing"

interface PublicShareProps {
  token: string
}

export const PublicShare: React.FC<PublicShareProps> = ({ token }) => {
  const navigate = useNavigate()
  const { data, isLoading, error } = usePublicPreview(token)
  const verifyPassword = useVerifySharePassword()
  const importToken = useImportFromToken()
  const [password, setPassword] = useState("")
  const [passwordVerified, setPasswordVerified] = useState(false)

  const handleVerifyPassword = async () => {
    try {
      const result = await verifyPassword.mutateAsync({ token, password })
      if (result.verified) {
        setPasswordVerified(true)
        message.success("Password verified")
      }
    } catch {
      message.error("Invalid password")
    }
  }

  const handleImport = async () => {
    try {
      const result = await importToken.mutateAsync(token)
      message.success(result.message || "Resource imported")
    } catch (err: unknown) {
      message.error(err instanceof Error ? err.message : "Failed to import")
    }
  }

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <Spin size="large" />
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <Empty description="This share link is invalid, expired, or has been revoked." />
      </div>
    )
  }

  const needsPassword = data.is_password_protected && !passwordVerified
  const isPrototypeWorkspace = data.resource_type === "prototype_workspace"

  const handlePrototypeContinue = () => {
    const target = `/prototype-workspaces?share_token=${encodeURIComponent(token)}`
    if (data.is_password_protected && passwordVerified && password) {
      navigate(target, {
        state: {
          prototypeSharePassword: password
        }
      })
      return
    }
    navigate(target)
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4">
      <Card className="w-full max-w-md shadow-lg" bordered>
        <div className="space-y-4">
          <div className="text-center">
            <h2 className="text-xl font-semibold text-text">
              {data.resource_name || "Shared Resource"}
            </h2>
            {data.resource_description && (
              <p className="mt-1 text-sm text-text-muted">
                {data.resource_description}
              </p>
            )}
          </div>

          <div className="flex justify-center gap-2">
            <Tag color="blue">
              {data.resource_type === "workspace"
                ? "Workspace"
                : data.resource_type === "prototype_workspace"
                  ? "Prototype Workspace"
                  : "Chatbook"}
            </Tag>
            <Tag>
              {ACCESS_LEVEL_LABELS[data.access_level as AccessLevel] ||
                data.access_level}
            </Tag>
          </div>

          {needsPassword ? (
            <div className="space-y-3">
              <div className="flex items-center justify-center gap-2 text-sm text-text-muted">
                <Lock className="h-4 w-4" />
                <span>This resource is password protected</span>
              </div>
              <Input.Password
                placeholder="Enter password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                onPressEnter={handleVerifyPassword}
              />
              <Button
                type="primary"
                block
                loading={verifyPassword.isPending}
                onClick={handleVerifyPassword}
              >
                Verify Password
              </Button>
            </div>
          ) : (
            <div className="space-y-3">
              {isPrototypeWorkspace ? (
                <>
                  <Button
                    type="primary"
                    block
                    onClick={handlePrototypeContinue}
                  >
                    Open Prototype Collaboration
                  </Button>
                  <p className="text-center text-xs text-text-muted">
                    Continue into the prototype workspace flow to exchange the
                    share link and start a collaborator session.
                  </p>
                </>
              ) : (
                <>
                  <Button
                    type="primary"
                    block
                    icon={<Download className="h-4 w-4" />}
                    loading={importToken.isPending}
                    onClick={handleImport}
                  >
                    Import to My Account
                  </Button>
                  <p className="text-center text-xs text-text-muted">
                    You must be logged in to import this resource.
                  </p>
                </>
              )}
            </div>
          )}
        </div>
      </Card>
    </div>
  )
}

export default PublicShare
