type Entry<T> = { value: T; expiresAt: number }

/** Map-compatible ownership for the two shared API payload caches. */
export class BoundedTtlCache<T> extends Map<string, Entry<T>> {
  private sizes = new Map<string, number>()
  private timers = new Map<string, ReturnType<typeof setTimeout>>()
  private bytes = 0

  override get(key: string): Entry<T> | undefined {
    this.expire()
    return super.get(key)
  }

  override set(key: string, entry: Entry<T>): this {
    this.expire()
    this.delete(key)
    let bytes: number
    try {
      bytes = (JSON.stringify(entry.value)?.length ?? 0) * 2
    } catch {
      return this
    }
    // ponytail: conservative UTF-16 payload size, 8 MiB / 32 entries per cache;
    // FIFO eviction is enough for these short-lived responses (TASK-13450).
    if (bytes > 8 * 1024 * 1024 || entry.expiresAt <= Date.now()) return this
    while (this.size >= 32 || this.bytes + bytes > 8 * 1024 * 1024) {
      this.delete(this.keys().next().value!)
    }
    super.set(key, entry)
    this.sizes.set(key, bytes)
    this.bytes += bytes
    const timer = setTimeout(() => this.delete(key), entry.expiresAt - Date.now())
    timer.unref?.()
    this.timers.set(key, timer)
    return this
  }

  override delete(key: string): boolean {
    clearTimeout(this.timers.get(key))
    this.timers.delete(key)
    this.bytes -= this.sizes.get(key) ?? 0
    this.sizes.delete(key)
    return super.delete(key)
  }

  override clear(): void {
    for (const key of this.keys()) this.delete(key)
  }

  private expire(): void {
    const now = Date.now()
    for (const [key, entry] of this) {
      if (entry.expiresAt <= now) this.delete(key)
    }
  }
}
