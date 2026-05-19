import Foundation
import Testing
import Virtualization
@testable import MacOSVZHelperDaemon

@Test func bootDriverStartsMachineForCanonicalBundleSpec() throws {
    let workspace = try bootDriverWorkspaceDirectory()
    let bundleDirectory = try bootDriverBundleDirectory()
    defer {
        try? FileManager.default.removeItem(at: bundleDirectory)
        try? FileManager.default.removeItem(at: workspace)
    }

    let machineProvider = RecordingVirtualMachineProvider()
    let driver = VirtualizationLinuxBootDriver(
        templateValidator: TemplateValidator(),
        configurationBuilder: VZLinuxConfigurationBuilder(),
        machineProvider: machineProvider
    )

    try driver.boot(
        vmID: "vm-bundle",
        templatePath: bundleDirectory.path(),
        workspacePath: workspace.path(),
        startupTimeoutSeconds: 5
    )

    #expect(machineProvider.recordedConfigurations.count == 1)
    #expect(machineProvider.startCallCount == 1)
}

@Test func bootDriverStopsTrackedMachine() throws {
    let workspace = try bootDriverWorkspaceDirectory()
    let bundleDirectory = try bootDriverBundleDirectory()
    defer {
        try? FileManager.default.removeItem(at: bundleDirectory)
        try? FileManager.default.removeItem(at: workspace)
    }

    let machineProvider = RecordingVirtualMachineProvider()
    let driver = VirtualizationLinuxBootDriver(
        templateValidator: TemplateValidator(),
        configurationBuilder: VZLinuxConfigurationBuilder(),
        machineProvider: machineProvider
    )

    try driver.boot(
        vmID: "vm-stop",
        templatePath: bundleDirectory.path(),
        workspacePath: workspace.path(),
        startupTimeoutSeconds: 5
    )

    try driver.stop(vmID: "vm-stop")

    #expect(machineProvider.stopCallCount == 1)
}

@Test func bootDriverPassesStartupTimeoutToMachine() throws {
    let workspace = try bootDriverWorkspaceDirectory()
    let bundleDirectory = try bootDriverBundleDirectory()
    defer {
        try? FileManager.default.removeItem(at: bundleDirectory)
        try? FileManager.default.removeItem(at: workspace)
    }

    let machineProvider = RecordingVirtualMachineProvider()
    let driver = VirtualizationLinuxBootDriver(
        templateValidator: TemplateValidator(),
        configurationBuilder: VZLinuxConfigurationBuilder(),
        machineProvider: machineProvider
    )

    try driver.boot(
        vmID: "vm-startup-timeout",
        templatePath: bundleDirectory.path(),
        workspacePath: workspace.path(),
        startupTimeoutSeconds: 9
    )

    #expect(machineProvider.lastStartupTimeoutSeconds == 9)
}

private final class RecordingVirtualMachineProvider: VirtualMachineProviding {
    private(set) var recordedConfigurations: [VZVirtualMachineConfiguration] = []
    private(set) var startCallCount = 0
    private(set) var stopCallCount = 0
    private(set) var lastStartupTimeoutSeconds: TimeInterval?

    func makeVirtualMachine(configuration: VZVirtualMachineConfiguration) throws -> VirtualMachineControlling {
        recordedConfigurations.append(configuration)
        return RecordingVirtualMachine(
            onStart: { timeout in
                self.startCallCount += 1
                self.lastStartupTimeoutSeconds = timeout
            },
            onStop: { self.stopCallCount += 1 }
        )
    }
}

private final class RecordingVirtualMachine: VirtualMachineControlling {
    private let onStart: (TimeInterval) -> Void
    private let onStop: () -> Void

    init(onStart: @escaping (TimeInterval) -> Void, onStop: @escaping () -> Void) {
        self.onStart = onStart
        self.onStop = onStop
    }

    func start(timeoutSeconds: TimeInterval) throws {
        onStart(timeoutSeconds)
    }

    func stop() throws {
        onStop()
    }

    func installSocketListener(_ listener: VZVirtioSocketListener, port: UInt32) throws {}
}

private func bootDriverWorkspaceDirectory() throws -> URL {
    let url = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func bootDriverBundleDirectory() throws -> URL {
    let url = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    try Data("kernel".utf8).write(to: url.appendingPathComponent("kernel"))
    try Data("initrd".utf8).write(to: url.appendingPathComponent("initrd.img"))

    let rootfsURL = url.appendingPathComponent("rootfs.img")
    FileManager.default.createFile(atPath: rootfsURL.path(), contents: Data())
    let handle = try FileHandle(forWritingTo: rootfsURL)
    defer {
        try? handle.close()
    }
    try handle.truncate(atOffset: 8 * 1024 * 1024)

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
