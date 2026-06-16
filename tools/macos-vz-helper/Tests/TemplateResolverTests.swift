import Foundation
import Testing
@testable import MacOSVZHelperDaemon

private func templateFixturesRoot(file: StaticString = #filePath) -> URL {
    URL(fileURLWithPath: "\(file)")
        .deletingLastPathComponent()
        .appendingPathComponent("TemplateFixtures", isDirectory: true)
}

@Test func bundleResolverProducesStrongBootSpec() throws {
    let resolver = BundleTemplateResolver()
    let bundlePath = templateFixturesRoot()
        .appendingPathComponent("bundle", isDirectory: true)
        .path()

    let spec = try resolver.resolve(templatePath: bundlePath)

    switch spec {
    case let .bundle(bundle):
        #expect(bundle.bootMode == .bundle)
        #expect(bundle.validationStrength == .strong)
        #expect(bundle.kernelPath.hasSuffix("/kernel"))
        #expect(bundle.rootfsPath.hasSuffix("/rootfs.img"))
        #expect(bundle.workspaceMountTag == "workspace")
    case .rawDisk:
        Issue.record("Expected bundle boot spec")
    }
}

@Test func rawDiskResolverProducesCompatibilityBootSpec() throws {
    let resolver = RawDiskTemplateResolver()
    let diskPath = templateFixturesRoot()
        .appendingPathComponent("raw-disk", isDirectory: true)
        .appendingPathComponent("disk.img", isDirectory: false)
        .path()

    let spec = try resolver.resolve(templatePath: diskPath)

    switch spec {
    case .bundle:
        Issue.record("Expected raw-disk boot spec")
    case let .rawDisk(rawDisk):
        #expect(rawDisk.bootMode == .rawDisk)
        #expect(rawDisk.validationStrength == .compatibility)
        #expect(rawDisk.bootLoaderKind == .efi)
        #expect(rawDisk.diskImagePath.hasSuffix("/disk.img"))
    }
}

@Test func validatorRejectsBundleMissingKernel() throws {
    let validator = TemplateValidator()
    let bundleDirectory = try temporaryBundleDirectory(includeKernel: false)

    defer {
        try? FileManager.default.removeItem(at: bundleDirectory)
    }

    let response = validator.validate(runtime: "vz_linux", templatePath: bundleDirectory.path)

    #expect(response.ready == false)
    #expect(response.reasons.contains("vz_linux_bundle_kernel_missing"))
}

@Test func validatorAcceptsBundlePathContainingSpaces() throws {
    let validator = TemplateValidator()
    let bundleDirectory = try temporaryBundleDirectory(
        parentName: "bundle parent \(UUID().uuidString)",
        bundleName: "bundle child",
        includeKernel: true
    )

    defer {
        try? FileManager.default.removeItem(at: bundleDirectory.deletingLastPathComponent())
    }

    let response = validator.validate(runtime: "vz_linux", templatePath: bundleDirectory.path)

    #expect(response.ready == true)
    #expect(response.reasons.isEmpty)
    #expect(response.source.contains(" "))
    #expect(!response.source.contains("%20"))
}

private func temporaryBundleDirectory(includeKernel: Bool) throws -> URL {
    try temporaryBundleDirectory(
        parentName: nil,
        bundleName: UUID().uuidString,
        includeKernel: includeKernel
    )
}

private func temporaryBundleDirectory(
    parentName: String?,
    bundleName: String,
    includeKernel: Bool
) throws -> URL {
    let root = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        .appendingPathComponent(parentName ?? bundleName, isDirectory: true)
    let bundleRoot = parentName == nil
        ? root
        : root.appendingPathComponent(bundleName, isDirectory: true)
    try FileManager.default.createDirectory(at: bundleRoot, withIntermediateDirectories: true)

    if includeKernel {
        FileManager.default.createFile(atPath: bundleRoot.appendingPathComponent("kernel").path, contents: Data("kernel".utf8))
    }
    FileManager.default.createFile(atPath: bundleRoot.appendingPathComponent("rootfs.img").path, contents: Data("rootfs".utf8))

    let manifest = """
    {
      "bundle_version": "1",
      "boot_mode": "bundle",
      "kernel": "kernel",
      "rootfs": "rootfs.img",
      "guest_agent_path": "/usr/local/bin/tldw-agent-guest",
      "workspace_mount_tag": "workspace",
      "vsock_port": 1024
    }
    """
    try manifest.write(to: bundleRoot.appendingPathComponent("manifest.json"), atomically: true, encoding: .utf8)
    return bundleRoot
}
