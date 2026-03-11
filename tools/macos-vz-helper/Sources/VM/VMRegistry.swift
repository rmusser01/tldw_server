import Foundation

final class VMRegistry {
    private var records: [String: VMRecord] = [:]
    private let lock = NSLock()

    func upsert(vmID: String, state: String, healthy: Bool) {
        lock.lock()
        defer { lock.unlock() }
        records[vmID] = VMRecord(vmID: vmID, state: state, healthy: healthy)
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
}
