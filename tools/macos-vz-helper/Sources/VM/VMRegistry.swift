import Foundation

final class VMRegistry {
    private var records: [String: VMRecord] = [:]
    private let lock = NSLock()

    func upsert(
        vmID: String,
        state: String,
        healthy: Bool,
        metadata: VMOwnershipMetadata? = nil,
        guestInfo: GuestAgentInfo? = nil,
        resourceSnapshot: VMResourceSnapshot? = nil,
        preserveGuestInfo: Bool = true,
        preserveResourceSnapshot: Bool = true
    ) {
        lock.lock()
        defer { lock.unlock() }
        let existing = records[vmID]
        let existingMetadata = existing?.metadata ?? .unknown
        let resolvedGuestInfo = guestInfo ?? (preserveGuestInfo ? existing?.guestInfo : nil)
        let resolvedResourceSnapshot = resourceSnapshot ?? (preserveResourceSnapshot ? existing?.resourceSnapshot : nil)
        records[vmID] = VMRecord(
            vmID: vmID,
            state: state,
            healthy: healthy,
            metadata: metadata ?? existingMetadata,
            guestInfo: resolvedGuestInfo,
            resourceSnapshot: resolvedResourceSnapshot
        )
    }

    func status(vmID: String) -> VMRecord? {
        lock.lock()
        defer { lock.unlock() }
        return records[vmID]
    }

    func list() -> [VMRecord] {
        lock.lock()
        defer { lock.unlock() }
        return records.values.sorted { $0.vmID < $1.vmID }
    }

    func remove(vmID: String) {
        lock.lock()
        defer { lock.unlock() }
        records.removeValue(forKey: vmID)
    }
}
