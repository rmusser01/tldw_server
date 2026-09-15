import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { createQuickIngestSessionStore } from "@/store/quick-ingest-session"
import { createQuickIngestAuthority } from "@/services/tldw/quick-ingest-authority"

const mocks = vi.hoisted(() => ({ initialize: vi.fn() }))
vi.mock("@plasmohq/storage", async () => import("../../../../../tldw-frontend/extension/shims/plasmo-storage"))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {
  initialize: mocks.initialize,
  ensureConfigForRequest: async () => {
    const config = JSON.parse(localStorage.getItem("tldwConfig") || "null")
    if (!config) throw new Error("Sign in")
    return config
  }
} }))
vi.mock("@/services/tldw/TldwAuth", () => ({ tldwAuth: { getCurrentUser: async () => ({ id: 1 }) } }))
vi.mock("@/services/tldw/deployment-mode", () => ({ isHostedTldwDeployment: () => false }))
vi.mock("@/utils/browser-runtime", () => ({ isExtensionRuntime: () => false }))

const a = { serverUrl: "https://a.test", authMode: "single-user", apiKey: "synthetic-a" }
const b = { ...a, serverUrl: "https://b.test" }
const change = (config: unknown) => {
  const oldValue = localStorage.getItem("tldwConfig")
  const newValue = JSON.stringify(config)
  localStorage.setItem("tldwConfig", newValue)
  window.dispatchEvent(new StorageEvent("storage", { key: "tldwConfig", oldValue, newValue }))
}
const releases: Array<() => void> = []
const setup = async () => {
  const store = createQuickIngestSessionStore()
  const authority = createQuickIngestAuthority(store)
  releases.push(authority.retain())
  await vi.waitFor(() => expect(store.getState().authorityKey).toBeTruthy())
  return { store, authority }
}
beforeEach(() => { localStorage.clear(); sessionStorage.clear(); mocks.initialize.mockReset().mockResolvedValue(undefined); localStorage.setItem("tldwConfig", JSON.stringify(a)) })
afterEach(() => { releases.splice(0).forEach(release => release()) })

describe("Quick Ingest verified authority lifecycle", () => {
  it("masks mounted private state synchronously on logout and rejects a captured completion", async () => {
    const { store, authority } = await setup()
    store.getState().createDraftSession({ results: [{ id: "Bob-private", status: "ok", type: "pdf", mediaId: 7 }] })
    const operation = authority.capture()
    window.dispatchEvent(new Event("tldw:auth-credentials-changed"))
    expect(store.getState().session).toBeNull()
    expect(operation.isCurrent()).toBe(false)
    expect(operation.signal.aborted).toBe(true)
  })

  it("rejects a pending initial A to B to A resolution without exposing persisted state", async () => {
    let resolve!: () => void
    mocks.initialize.mockReturnValueOnce(new Promise<void>(done => { resolve = done }))
    const store = createQuickIngestSessionStore()
    const authority = createQuickIngestAuthority(store)
    releases.push(authority.retain())
    change(b); change(a)
    resolve()
    await vi.waitFor(() => expect(store.getState().authorityKey).toBeTruthy())
    expect(store.getState().session).toBeNull()
  })

  it("preserves same-owner close and reopen while invalidating abandoned UI callbacks", async () => {
    const { store, authority } = await setup()
    const session = store.getState().createDraftSession({ results: [{ id: "owned", status: "ok", type: "pdf" }] })
    const operation = authority.capture()
    releases.pop()!()
    expect(store.getState().session).toBeNull()
    expect(operation.isCurrent()).toBe(false)
    releases.push(authority.retain())
    await vi.waitFor(() => expect(store.getState().session?.id).toBe(session.id))
  })

  it("does not end another mounted owner's lifetime or depend on connectivity events", async () => {
    const { store, authority } = await setup()
    const secondRelease = authority.retain()
    const session = store.getState().createDraftSession()
    const operation = authority.capture()
    secondRelease()
    window.dispatchEvent(new Event("offline"))
    window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: false } }))
    expect(store.getState().session?.id).toBe(session.id)
    expect(operation.isCurrent()).toBe(true)
  })

  it("rejects the original operation after a cross-server A to B to A and a new draft", async () => {
    const { store, authority } = await setup()
    store.getState().createDraftSession()
    const operation = authority.capture()
    change(b); change(a)
    await vi.waitFor(() => expect(store.getState().authorityKey).toBeTruthy())
    store.getState().createDraftSession()
    expect(operation.isCurrent()).toBe(false)
    expect(() => operation.assertCurrent()).toThrow()
  })

  it('regression stale expiry hint preserves a valid rotated ingest session and operation', async () => {
    const { tldwClient } = await import('@/services/tldw/TldwApiClient')
    const { resolveDirectBrowserConfig } = await import('@/services/tldw/direct-browser-config')
    const { storeRefreshRotationIfCurrent } = await import('@/services/tldw/single-user-credential')
    const token = (revision: number) => 'test.' + btoa(JSON.stringify({sub:'1',revision})) + '.signature'
    const original = {serverUrl:'https://a.test',authMode:'multi-user' as const,accessToken:token(1),refreshToken:'source-refresh'}
    localStorage.setItem('tldwConfig', JSON.stringify(original))
    const storage={get:async <T,>(key:string)=>JSON.parse(localStorage.getItem(key)||'null') as T,
      set:async <T,>(key:string,value:T)=>{localStorage.setItem(key,JSON.stringify(value))},
      remove:async(key:string)=>{localStorage.removeItem(key)}}
    const ensure=vi.spyOn(tldwClient,'ensureConfigForRequest').mockImplementation(async()=>{
      const value=await resolveDirectBrowserConfig(storage)
      if(!value?.accessToken)throw new Error('Authentication required')
      return value
    })
    try {
      const {store,authority}=await setup()
      const session=store.getState().createDraftSession({results:[{id:'private-owned',status:'ok',type:'pdf',fileName:'Own.pdf',mediaId:7}]})
      const operation=authority.capture()
      expect(await storeRefreshRotationIfCurrent(storage,original,original.refreshToken,{accessToken:token(2),refreshToken:'rotated-refresh'})).toBe(true)
      window.dispatchEvent(new CustomEvent('tldw:config-updated',{detail:{authorityChanged:false}}))
      expect((await tldwClient.ensureConfigForRequest(true)).accessToken).toBe(token(2))
      window.dispatchEvent(new CustomEvent('tldw:config-updated',{detail:{refreshSessionInvalidated:true}}))
      await vi.waitFor(()=>expect(store.getState().authorityKey).toBeTruthy())
      expect({sessionId:store.getState().session?.id,operationCurrent:operation.isCurrent(),aborted:operation.signal.aborted}).toEqual({sessionId:session.id,operationCurrent:true,aborted:false})
    } finally {ensure.mockRestore()}
  })

  it('regression delayed native marker check preserves a newer valid rotation', async () => {
    const {tldwClient}=await import('@/services/tldw/TldwApiClient')
    const {resolveDirectBrowserConfig}=await import('@/services/tldw/direct-browser-config')
    const {storeRefreshRotationIfCurrent,refreshSessionInvalidationKey}=await import('@/services/tldw/single-user-credential')
    const token=(revision:number)=>'test.'+btoa(JSON.stringify({sub:'1',revision}))+'.signature'
    const original={serverUrl:'https://a.test',authMode:'multi-user' as const,accessToken:token(1),refreshToken:'source-refresh'}
    localStorage.setItem('tldwConfig',JSON.stringify(original))
    const storage={get:async<T,>(key:string)=>JSON.parse(localStorage.getItem(key)||'null') as T,set:async<T,>(key:string,value:T)=>{localStorage.setItem(key,JSON.stringify(value))},remove:async(key:string)=>{localStorage.removeItem(key)}}
    const read=async()=>{const value=await resolveDirectBrowserConfig(storage);if(!value?.accessToken)throw new Error('Authentication required');return value}
    const ensure=vi.spyOn(tldwClient,'ensureConfigForRequest').mockImplementation(read)
    try {
      const {store,authority}=await setup()
      const session=store.getState().createDraftSession({results:[{id:'owned',status:'ok',type:'pdf'}]})
      const operation=authority.capture()
      let fail!: (reason:unknown)=>void
      const pending=new Promise<Awaited<ReturnType<typeof read>>>((_resolve,reject)=>{fail=reject})
      ensure.mockImplementationOnce(()=>pending)
      const key=refreshSessionInvalidationKey(original)!
      await storage.set(key,true)
      window.dispatchEvent(new StorageEvent('storage',{key,newValue:'true'}))
      expect(await storeRefreshRotationIfCurrent(storage,original,original.refreshToken,{accessToken:token(2),refreshToken:'rotated-refresh'})).toBe(true)
      window.dispatchEvent(new CustomEvent('tldw:config-updated',{detail:{authorityChanged:false}}))
      expect((await read()).accessToken).toBe(token(2))
      fail(new Error('Earlier expired credential check'))
      await pending.catch(()=>undefined)
      await new Promise(resolve=>setTimeout(resolve,0))
      await vi.waitFor(()=>expect(store.getState().authorityKey).toBeTruthy())
      expect({sessionId:store.getState().session?.id,current:operation.isCurrent(),aborted:operation.signal.aborted}).toEqual({sessionId:session.id,current:true,aborted:false})
    } finally {ensure.mockRestore()}
  })

  it.each(["hint", "marker"])("ends the captured UI operation for a current terminal %s", async (event) => {
    const { tldwClient } = await import("@/services/tldw/TldwApiClient")
    const { resolveDirectBrowserConfig } = await import("@/services/tldw/direct-browser-config")
    const { refreshSessionInvalidationKey } = await import("@/services/tldw/single-user-credential")
    const original = { serverUrl: "https://a.test", authMode: "multi-user" as const, accessToken: "test." + btoa(JSON.stringify({ sub: "1" })) + ".signature", refreshToken: "terminal-refresh" }
    localStorage.setItem("tldwConfig", JSON.stringify(original))
    const storage = { get: async <T,>(key: string) => JSON.parse(localStorage.getItem(key) || "null") as T, set: async <T,>(key: string, value: T) => { localStorage.setItem(key, JSON.stringify(value)) }, remove: async (key: string) => { localStorage.removeItem(key) } }
    const ensure = vi.spyOn(tldwClient, "ensureConfigForRequest").mockImplementation(async () => {
      const current = await resolveDirectBrowserConfig(storage)
      if (!current?.accessToken) throw new Error("Authentication required")
      return current
    })
    try {
      const { store, authority } = await setup()
      store.getState().createDraftSession({ results: [{ id: "owned", status: "ok", type: "pdf" }] })
      const operation = authority.capture()
      const key = refreshSessionInvalidationKey(original)!
      await storage.set(key, true)
      if (event === "hint") window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { refreshSessionInvalidated: true } }))
      else window.dispatchEvent(new StorageEvent("storage", { key, newValue: "true" }))
      await vi.waitFor(() => expect(operation.signal.aborted).toBe(true))
      expect(store.getState().session).toBeNull()
      expect(operation.isCurrent()).toBe(false)
    } finally { ensure.mockRestore() }
  })

  it("ignores a marker rejection from an abandoned UI lifetime after same-owner resume", async () => {
    const { tldwClient } = await import("@/services/tldw/TldwApiClient")
    const { REFRESH_SESSION_INVALIDATION_PREFIX } = await import("@/services/tldw/single-user-credential")
    const { store, authority } = await setup()
    const session = store.getState().createDraftSession()
    const oldOperation = authority.capture()
    let fail!: (reason: unknown) => void
    const pending = new Promise<Awaited<ReturnType<typeof tldwClient.ensureConfigForRequest>>>((_resolve, reject) => { fail = reject })
    const ensure = vi.spyOn(tldwClient, "ensureConfigForRequest").mockImplementationOnce(() => pending)
    try {
      window.dispatchEvent(new StorageEvent("storage", { key: REFRESH_SESSION_INVALIDATION_PREFIX + "old-check", newValue: "true" }))
      releases.pop()!()
      releases.push(authority.retain())
      await vi.waitFor(() => expect(store.getState().session?.id).toBe(session.id))
      const resumedOperation = authority.capture()
      fail(new Error("Old lifetime credential read"))
      await pending.catch(() => undefined)
      await new Promise(resolve => setTimeout(resolve, 0))
      expect(oldOperation.isCurrent()).toBe(false)
      expect(resumedOperation.isCurrent()).toBe(true)
      expect(resumedOperation.signal.aborted).toBe(false)
      expect(store.getState().session?.id).toBe(session.id)
    } finally { fail(new Error("Cleanup")); ensure.mockRestore() }
  })
})
