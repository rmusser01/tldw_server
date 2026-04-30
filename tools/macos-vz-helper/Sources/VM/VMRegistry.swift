import Foundation

final class VMRegistry {
    private var records: [String: VMRecord] = [:]
    private let lock = NSLock()

    func upsert(vmID: String, state: String, healthy: Bool, metadata: VMOwnershipMetadata? = nil) {
        lock.lock()
        defer { lock.unlock() }
        let existingMetadata = records[vmID]?.metadata ?? .unknown
        records[vmID] = VMRecord(
            vmID: vmID,
            state: state,
            healthy: healthy,
            metadata: metadata ?? existingMetadata
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
