export interface PrincipalIdentity {
  id?: number
  is_active?: boolean
}

type AuthenticatedGet = <T>(path: string) => Promise<T>

/** Read fresh profile identity; only profile absence permits the legacy auth endpoint. */
export async function fetchCurrentPrincipal(read: AuthenticatedGet): Promise<PrincipalIdentity | undefined> {
  try {
    const profile = await read<{ user?: PrincipalIdentity }>("/users/me/profile")
    return profile.user
  } catch (error) {
    const status = error && typeof error === "object" ? (error as { status?: number }).status : undefined
    if (status !== 404 && status !== 410) throw error
    return read<PrincipalIdentity>("/auth/me")
  }
}
