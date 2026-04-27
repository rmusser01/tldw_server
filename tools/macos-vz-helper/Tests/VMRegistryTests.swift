import Testing
@testable import MacOSVZHelperDaemon

@Test func vmRegistryTracksCreatedVMIDs() throws {
    let registry = VMRegistry()
    registry.upsert(vmID: "vm-123", state: "created", healthy: false)

    let status = registry.status(vmID: "vm-123")

    #expect(status?.vmID == "vm-123")
    #expect(status?.state == "created")
    #expect(status?.healthy == false)
}

@Test func listVMsReturnsKnownVMs() throws {
    let registry = VMRegistry()
    registry.upsert(vmID: "vm-1", state: "running", healthy: true)
    registry.upsert(vmID: "vm-2", state: "booting", healthy: false)

    let allVMs = registry.list()

    #expect(allVMs.count == 2)
    #expect(allVMs.contains(where: { $0.vmID == "vm-1" && $0.state == "running" }))
    #expect(allVMs.contains(where: { $0.vmID == "vm-2" && $0.state == "booting" }))
}
