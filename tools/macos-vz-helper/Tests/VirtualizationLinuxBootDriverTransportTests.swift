import Foundation
import Testing
import Virtualization
@testable import MacOSVZHelperDaemon

@Test func bootDriverCreatesVSockListenerStateBeforeStart() throws {
    let workspace = try bootDriverWorkspaceDirectory()
    let bundleDirectory = try bootDriverBundleDirectory()
    defer {
        try? FileManager.default.removeItem(at: bundleDirectory)
        try? FileManager.default.removeItem(at: workspace)
    }

    let sessionManager = VSockSessionManager()
    let machineProvider = TransportRecordingVirtualMachineProvider(sessionManager: sessionManager)
    let driver = VirtualizationLinuxBootDriver(
        templateValidator: TemplateValidator(),
        configurationBuilder: VZLinuxConfigurationBuilder(),
        machineProvider: machineProvider,
        sessionManager: sessionManager,
        connectionTokenFactory: { "token-transport" }
    )

    try driver.boot(
        vmID: "vm-transport",
        templatePath: bundleDirectory.path(),
        workspacePath: workspace.path(),
        startupTimeoutSeconds: 5
    )

    #expect(machineProvider.listenerInstalled == true)
    #expect(machineProvider.sessionPreparedBeforeStart == true)
    #expect(sessionManager.hasPreparedSession(vmID: "vm-transport") == true)
}

@Test func bootDriverInjectsVMIDAndConnectionTokenIntoGuestConfig() throws {
    let workspace = try bootDriverWorkspaceDirectory()
    let bundleDirectory = try bootDriverBundleDirectory()
    defer {
        try? FileManager.default.removeItem(at: bundleDirectory)
        try? FileManager.default.removeItem(at: workspace)
    }

    let machineProvider = TransportRecordingVirtualMachineProvider()
    let driver = VirtualizationLinuxBootDriver(
        templateValidator: TemplateValidator(),
        configurationBuilder: VZLinuxConfigurationBuilder(),
        machineProvider: machineProvider,
        sessionManager: VSockSessionManager(),
        connectionTokenFactory: { "token-fixed" }
    )

    try driver.boot(
        vmID: "vm-config",
        templatePath: bundleDirectory.path(),
        workspacePath: workspace.path(),
        startupTimeoutSeconds: 5
    )

    let configuration = try #require(machineProvider.recordedConfiguration)
    let bootLoader = try #require(configuration.bootLoader as? VZLinuxBootLoader)
    let commandLine = bootLoader.commandLine

    #expect(commandLine.contains("rootfstype=ext4"))
    #expect(commandLine.contains("rootwait"))
    #expect(commandLine.contains("TLDW_AGENT_GUEST_VM_ID=vm-config"))
    #expect(commandLine.contains("TLDW_AGENT_GUEST_CONNECTION_TOKEN=token-fixed"))
    #expect(commandLine.contains("TLDW_AGENT_GUEST_HOST_VSOCK_PORT=1024"))
    #expect(commandLine.contains("TLDW_AGENT_GUEST_WORKSPACE_ROOT=/workspace"))
}

private final class TransportRecordingVirtualMachineProvider: VirtualMachineProviding {
    private let sessionManager: VSockSessionManager?
    private(set) var recordedConfiguration: VZVirtualMachineConfiguration?
    private(set) var listenerInstalled = false
    private(set) var sessionPreparedBeforeStart = false

    init(sessionManager: VSockSessionManager? = nil) {
        self.sessionManager = sessionManager
    }

    func makeVirtualMachine(configuration: VZVirtualMachineConfiguration) throws -> VirtualMachineControlling {
        recordedConfiguration = configuration
        return TransportRecordingVirtualMachine(
            onInstall: {
                self.listenerInstalled = true
            },
            onStart: {
                self.sessionPreparedBeforeStart = self.sessionManager?.hasPreparedSession(vmID: "vm-transport") ?? false
            }
        )
    }
}

private final class TransportRecordingVirtualMachine: VirtualMachineControlling {
    private let onInstall: () -> Void
    private let onStart: () -> Void

    init(onInstall: @escaping () -> Void, onStart: @escaping () -> Void) {
        self.onInstall = onInstall
        self.onStart = onStart
    }

    func start(timeoutSeconds: TimeInterval) throws {
        onStart()
    }

    func stop() throws {}

    func installSocketListener(_ listener: VZVirtioSocketListener, port: UInt32) throws {
        onInstall()
    }
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
