import Foundation
import Testing
import Virtualization
@testable import MacOSVZHelperDaemon

@Test func configurationBuilderCreatesLinuxBootLoaderForBundleSpec() throws {
    let builder = VZLinuxConfigurationBuilder()
    let bundleDirectory = try temporaryBundleDirectory()
    let spec = try BundleTemplateResolver().resolve(templatePath: bundleDirectory.path())
    let workspace = try temporaryWorkspaceDirectory()

    defer {
        try? FileManager.default.removeItem(at: bundleDirectory)
        try? FileManager.default.removeItem(at: workspace)
    }

    let configuration = try builder.build(spec: spec, workspacePath: workspace.path())

    #expect(configuration.platform is VZGenericPlatformConfiguration)
    #expect(configuration.bootLoader is VZLinuxBootLoader)
    #expect(configuration.storageDevices.count == 1)
    #expect(configuration.directorySharingDevices.count == 1)
    #expect(configuration.socketDevices.count == 1)

    let fileSystemDevice = configuration.directorySharingDevices.first as? VZVirtioFileSystemDeviceConfiguration
    #expect(fileSystemDevice?.tag == "workspace")
}

@Test func configurationBuilderCreatesCompatibilityBootPathForRawDiskSpec() throws {
    let builder = VZLinuxConfigurationBuilder()
    let diskURL = try temporaryRawDiskImage()
    let spec = try RawDiskTemplateResolver().resolve(templatePath: diskURL.path())
    let workspace = try temporaryWorkspaceDirectory()

    defer {
        try? FileManager.default.removeItem(at: diskURL)
        try? FileManager.default.removeItem(at: workspace)
    }

    let configuration = try builder.build(spec: spec, workspacePath: workspace.path())

    #expect(configuration.platform is VZGenericPlatformConfiguration)
    #expect(configuration.bootLoader is VZEFIBootLoader)
    #expect(configuration.storageDevices.count == 1)
    #expect(configuration.directorySharingDevices.count == 1)
    #expect(configuration.socketDevices.count == 1)
}

@Test func configurationBuilderAddsSerialLogPortWhenConfigured() throws {
    let serialLogDirectory = try temporaryWorkspaceDirectory()
    let builder = VZLinuxConfigurationBuilder(serialLogDirectory: serialLogDirectory.path())
    let bundleDirectory = try temporaryBundleDirectory()
    let spec = try BundleTemplateResolver().resolve(templatePath: bundleDirectory.path())
    let workspace = try temporaryWorkspaceDirectory()
    let guestTransport = GuestTransportMetadata(
        vmID: "vm/serial log",
        connectionToken: "token",
        hostPort: 1024,
        workspaceRoot: "/workspace"
    )

    defer {
        try? FileManager.default.removeItem(at: serialLogDirectory)
        try? FileManager.default.removeItem(at: bundleDirectory)
        try? FileManager.default.removeItem(at: workspace)
    }

    let configuration = try builder.build(
        spec: spec,
        workspacePath: workspace.path(),
        guestTransport: guestTransport
    )

    #expect(configuration.serialPorts.count == 1)
    let logPath = serialLogDirectory.appendingPathComponent("vm_serial_log.serial.log").path()
    #expect(FileManager.default.fileExists(atPath: logPath))
}

private func temporaryWorkspaceDirectory() throws -> URL {
    let url = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func temporaryBundleDirectory() throws -> URL {
    let url = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    try Data("kernel".utf8).write(to: url.appendingPathComponent("kernel"))
    try Data("initrd".utf8).write(to: url.appendingPathComponent("initrd.img"))

    let rootfsURL = url.appendingPathComponent("rootfs.img")
    try createRawDiskImage(at: rootfsURL)

    let manifest = """
    {
      "bundle_version": "1",
      "boot_mode": "bundle",
      "kernel": "kernel",
      "initrd": "initrd.img",
      "rootfs": "rootfs.img",
      "guest_agent_path": "/usr/local/bin/tldw-agent-guest",
      "workspace_mount_tag": "workspace",
      "vsock_port": 1024
    }
    """
    try manifest.write(to: url.appendingPathComponent("manifest.json"), atomically: true, encoding: .utf8)
    return url
}

private func temporaryRawDiskImage() throws -> URL {
    let url = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        .appendingPathComponent(UUID().uuidString + ".img", isDirectory: false)
    try createRawDiskImage(at: url)
    return url
}

private func createRawDiskImage(at url: URL) throws {
    FileManager.default.createFile(atPath: url.path(), contents: Data())
    let handle = try FileHandle(forWritingTo: url)
    defer {
        try? handle.close()
    }
    try handle.truncate(atOffset: 8 * 1024 * 1024)
}
