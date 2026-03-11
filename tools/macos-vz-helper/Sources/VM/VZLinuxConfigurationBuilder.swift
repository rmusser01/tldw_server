import Foundation
import Virtualization

enum VZLinuxConfigurationBuilderError: Error {
    case workspaceDirectoryMissing
}

struct GuestTransportMetadata: Equatable {
    let vmID: String
    let connectionToken: String
    let hostPort: UInt32
    let workspaceRoot: String
}

protocol VZLinuxConfigurationBuilding {
    func build(
        spec: TemplateBootSpec,
        workspacePath: String,
        guestTransport: GuestTransportMetadata?
    ) throws -> VZVirtualMachineConfiguration
}

struct VZLinuxConfigurationBuilder: VZLinuxConfigurationBuilding {
    private let cpuCount: Int
    private let memorySize: UInt64

    init(cpuCount: Int = 2, memorySize: UInt64 = 1_073_741_824) {
        self.cpuCount = cpuCount
        self.memorySize = memorySize
    }

    func build(
        spec: TemplateBootSpec,
        workspacePath: String,
        guestTransport: GuestTransportMetadata? = nil
    ) throws -> VZVirtualMachineConfiguration {
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: workspacePath, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw VZLinuxConfigurationBuilderError.workspaceDirectoryMissing
        }

        let configuration = VZVirtualMachineConfiguration()
        configuration.bootLoader = try bootLoader(for: spec, guestTransport: guestTransport)
        configuration.cpuCount = cpuCount
        configuration.memorySize = memorySize
        configuration.storageDevices = [try storageDevice(for: spec)]
        configuration.directorySharingDevices = [directorySharingDevice(tag: spec.workspaceMountTag, workspacePath: workspacePath)]
        configuration.socketDevices = [VZVirtioSocketDeviceConfiguration()]
        configuration.entropyDevices = [VZVirtioEntropyDeviceConfiguration()]
        return configuration
    }

    private func bootLoader(for spec: TemplateBootSpec, guestTransport: GuestTransportMetadata?) throws -> VZBootLoader {
        switch spec {
        case let .bundle(bundle):
            let bootLoader = VZLinuxBootLoader(kernelURL: URL(fileURLWithPath: bundle.kernelPath))
            bootLoader.initialRamdiskURL = bundle.initrdPath.map { URL(fileURLWithPath: $0) }
            bootLoader.commandLine = linuxCommandLine(guestTransport: guestTransport)
            return bootLoader
        case let .rawDisk(rawDisk):
            switch rawDisk.bootLoaderKind {
            case .efi:
                return VZEFIBootLoader()
            case .linuxKernel:
                let bootLoader = VZLinuxBootLoader(kernelURL: URL(fileURLWithPath: rawDisk.diskImagePath))
                bootLoader.commandLine = linuxCommandLine(guestTransport: guestTransport, base: "console=hvc0")
                return bootLoader
            }
        }
    }

    private func linuxCommandLine(
        guestTransport: GuestTransportMetadata?,
        base: String = "console=hvc0 root=/dev/vda rw"
    ) -> String {
        guard let guestTransport else {
            return base
        }

        let guestEnv = [
            "TLDW_AGENT_GUEST_VM_ID=\(guestTransport.vmID)",
            "TLDW_AGENT_GUEST_CONNECTION_TOKEN=\(guestTransport.connectionToken)",
            "TLDW_AGENT_GUEST_HOST_VSOCK_PORT=\(guestTransport.hostPort)",
            "TLDW_AGENT_GUEST_WORKSPACE_ROOT=\(guestTransport.workspaceRoot)",
        ]
            .map { "systemd.setenv=\($0)" }
            .joined(separator: " ")

        return "\(base) \(guestEnv)"
    }

    private func storageDevice(for spec: TemplateBootSpec) throws -> VZVirtioBlockDeviceConfiguration {
        let diskPath: String
        switch spec {
        case let .bundle(bundle):
            diskPath = bundle.rootfsPath
        case let .rawDisk(rawDisk):
            diskPath = rawDisk.diskImagePath
        }

        let attachment = try VZDiskImageStorageDeviceAttachment(
            url: URL(fileURLWithPath: diskPath),
            readOnly: false
        )
        return VZVirtioBlockDeviceConfiguration(attachment: attachment)
    }

    private func directorySharingDevice(
        tag: String,
        workspacePath: String
    ) -> VZVirtioFileSystemDeviceConfiguration {
        let sharedDirectory = VZSharedDirectory(
            url: URL(fileURLWithPath: workspacePath, isDirectory: true),
            readOnly: false
        )
        let share = VZSingleDirectoryShare(directory: sharedDirectory)
        let device = VZVirtioFileSystemDeviceConfiguration(tag: tag)
        device.share = share
        return device
    }
}
