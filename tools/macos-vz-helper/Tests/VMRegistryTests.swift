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

@Test func vmRegistryStoresOwnershipMetadata() throws {
    let registry = VMRegistry()
    let metadata = VMOwnershipMetadata(
        owner: "tldw",
        runtime: "vz_linux",
        runID: "run-1",
        sessionID: "session-1",
        sessionMode: true,
        templateID: "vz_linux:bundle",
        templatePath: "/tmp/bundle",
        runManifestPath: "/tmp/image-store/runs/run-1/manifest.json",
        planningSource: "image_store",
        workspacePath: "/tmp/workspace",
        createdAt: "2026-04-30T18:00:00Z"
    )

    registry.upsert(vmID: "vm-1", state: "booting", healthy: false, metadata: metadata)

    #expect(registry.status(vmID: "vm-1")?.metadata.owner == "tldw")
    #expect(registry.status(vmID: "vm-1")?.metadata.runID == "run-1")
    #expect(registry.status(vmID: "vm-1")?.metadata.templateID == "vz_linux:bundle")
}

@Test func vmRegistryPreservesMetadataAcrossStateUpdates() throws {
    let registry = VMRegistry()
    let metadata = VMOwnershipMetadata(
        owner: "tldw",
        runtime: "vz_linux",
        runID: "run-1",
        sessionID: "",
        sessionMode: false,
        templateID: "vz_linux:bundle",
        templatePath: "/tmp/bundle",
        runManifestPath: "/tmp/image-store/runs/run-1/manifest.json",
        planningSource: "image_store",
        workspacePath: "/tmp/workspace",
        createdAt: "2026-04-30T18:00:00Z"
    )

    registry.upsert(vmID: "vm-1", state: "booting", healthy: false, metadata: metadata)
    registry.upsert(vmID: "vm-1", state: "running", healthy: true)

    let record = registry.status(vmID: "vm-1")
    #expect(record?.state == "running")
    #expect(record?.metadata.runID == "run-1")
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
