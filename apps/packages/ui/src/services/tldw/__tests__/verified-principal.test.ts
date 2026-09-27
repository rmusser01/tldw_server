import { describe, expect, it } from "vitest"
import { fetchCurrentPrincipal } from "../verified-principal"

describe("fresh authenticated principal lookup", () => {
  it("prefers the profile and reads fresh identity on every verification", async () => {
    let id = 1
    const paths: string[] = []
    const read = async <T>(path: string): Promise<T> => {
      paths.push(path)
      if (path !== "/users/me/profile") throw new Error("Unexpected fallback")
      return { user: { id, is_active: true } } as T
    }
    expect(await fetchCurrentPrincipal(read)).toEqual({ id: 1, is_active: true })
    id = 2
    expect(await fetchCurrentPrincipal(read)).toEqual({ id: 2, is_active: true })
    expect(paths).toEqual(["/users/me/profile", "/users/me/profile"])
  })

  it.each([404, 410])("uses the supplied authenticated transport after profile HTTP %s", async (status) => {
    const paths: string[] = []
    const read = async <T>(path: string): Promise<T> => {
      paths.push(path)
      if (path === "/users/me/profile") throw Object.assign(new Error("No profile"), { status })
      if (path === "/auth/me") return { id: 7, is_active: true } as T
      throw new Error("Unexpected endpoint")
    }
    expect(await fetchCurrentPrincipal(read)).toEqual({ id: 7, is_active: true })
    expect(paths).toEqual(["/users/me/profile", "/auth/me"])
  })

  it.each([401, 403, 429, 500, undefined])("propagates profile failures without fallback: %s", async (status) => {
    const failure = Object.assign(new Error("Verification unavailable"), { status })
    const paths: string[] = []
    const read = async <T>(path: string): Promise<T> => {
      paths.push(path)
      if (path === "/users/me/profile") throw failure
      return { id: 7, is_active: true } as T
    }
    await expect(fetchCurrentPrincipal(read)).rejects.toBe(failure)
    expect(paths).toEqual(["/users/me/profile"])
  })

  it("does not replace a missing profile identity with a fallback identity", async () => {
    const read = async <T>(path: string): Promise<T> => {
      if (path !== "/users/me/profile") throw new Error("Unexpected fallback")
      return {} as T
    }
    expect(await fetchCurrentPrincipal(read)).toBeUndefined()
  })

  it("propagates failure of the legacy endpoint", async () => {
    const failure = Object.assign(new Error("Not authenticated"), { status: 401 })
    const read = async <T>(path: string): Promise<T> => {
      if (path === "/users/me/profile") throw Object.assign(new Error("No profile"), { status: 404 })
      throw failure
    }
    await expect(fetchCurrentPrincipal(read)).rejects.toBe(failure)
  })
})
