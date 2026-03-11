import Dispatch
import Foundation
import Virtualization

protocol VirtualMachineControlling {
    func start() throws
    func stop() throws
    func installSocketListener(_ listener: VZVirtioSocketListener, port: UInt32) throws
}

protocol VirtualMachineProviding {
    func makeVirtualMachine(configuration: VZVirtualMachineConfiguration) throws -> VirtualMachineControlling
}

enum VirtualizationLinuxBootDriverError: Error {
    case socketDeviceMissing
}

private final class VZVirtualMachineController: VirtualMachineControlling {
    private let machine: VZVirtualMachine

    init(machine: VZVirtualMachine) {
        self.machine = machine
    }

    func start() throws {
        let semaphore = DispatchSemaphore(value: 0)
        var startError: Error?
        machine.start { result in
            if case let .failure(error) = result {
                startError = error
            }
            semaphore.signal()
        }
        semaphore.wait()
        if let startError {
            throw startError
        }
    }

    func stop() throws {
        if machine.canRequestStop {
            try machine.requestStop()
        }
    }

    func installSocketListener(_ listener: VZVirtioSocketListener, port: UInt32) throws {
        guard let socketDevice = machine.socketDevices.first(where: { $0 is VZVirtioSocketDevice }) as? VZVirtioSocketDevice else {
            throw VirtualizationLinuxBootDriverError.socketDeviceMissing
        }
        socketDevice.setSocketListener(listener, forPort: port)
    }
}

struct VZVirtualMachineProvider: VirtualMachineProviding {
    func makeVirtualMachine(configuration: VZVirtualMachineConfiguration) throws -> VirtualMachineControlling {
        try configuration.validate()
        return VZVirtualMachineController(machine: VZVirtualMachine(configuration: configuration))
    }
}

final class VirtualizationLinuxBootDriver: VZBootDriving {
    private let templateValidator: TemplateValidator
    private let configurationBuilder: VZLinuxConfigurationBuilding
    private let machineProvider: VirtualMachineProviding
    private let sessionManager: VSockSessionManager
    private let connectionTokenFactory: () -> String
    private var machines: [String: VirtualMachineControlling] = [:]
    private let defaultWorkspaceRoot = "/workspace"

    init(
        templateValidator: TemplateValidator = TemplateValidator(),
        configurationBuilder: VZLinuxConfigurationBuilding = VZLinuxConfigurationBuilder(),
        machineProvider: VirtualMachineProviding = VZVirtualMachineProvider(),
        sessionManager: VSockSessionManager = VSockSessionManager(),
        connectionTokenFactory: @escaping () -> String = { UUID().uuidString }
    ) {
        self.templateValidator = templateValidator
        self.configurationBuilder = configurationBuilder
        self.machineProvider = machineProvider
        self.sessionManager = sessionManager
        self.connectionTokenFactory = connectionTokenFactory
    }

    func boot(vmID: String, templatePath: String, workspacePath: String) throws {
        let spec = try templateValidator.resolve(runtime: "vz_linux", templatePath: templatePath)
        let guestTransport = GuestTransportMetadata(
            vmID: vmID,
            connectionToken: connectionTokenFactory(),
            hostPort: spec.vsockPort,
            workspaceRoot: defaultWorkspaceRoot
        )
        let listener = sessionManager.prepareSession(
            vmID: vmID,
            connectionToken: guestTransport.connectionToken,
            port: spec.vsockPort,
            workspaceRoot: defaultWorkspaceRoot
        )
        let configuration = try configurationBuilder.build(
            spec: spec,
            workspacePath: workspacePath,
            guestTransport: guestTransport
        )
        let machine = try machineProvider.makeVirtualMachine(configuration: configuration)
        do {
            try machine.installSocketListener(listener.listener, port: spec.vsockPort)
            try machine.start()
            machines[vmID] = machine
        } catch {
            sessionManager.removeSession(vmID: vmID)
            throw error
        }
    }

    func stop(vmID: String) throws {
        guard let machine = machines.removeValue(forKey: vmID) else {
            sessionManager.removeSession(vmID: vmID)
            return
        }
        try machine.stop()
        sessionManager.removeSession(vmID: vmID)
    }
}
