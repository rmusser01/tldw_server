export const AUTH_CREDENTIALS_CHANGED_EVENT = "tldw:auth-credentials-changed"

export type AuthCredentialsChangedDetail = {
  authenticated: boolean
}

export function dispatchAuthCredentialsChanged(authenticated: boolean): void {
  if (typeof window === "undefined") return
  window.dispatchEvent(
    new CustomEvent<AuthCredentialsChangedDetail>(AUTH_CREDENTIALS_CHANGED_EVENT, {
      detail: { authenticated }
    })
  )
}
