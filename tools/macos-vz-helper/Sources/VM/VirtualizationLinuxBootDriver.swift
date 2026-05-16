import Dispatch
import Foundation
import Virtualization

protocol VirtualMachineControlling {
    func start(timeoutSeconds: TimeInterval) throws
    func stop() throws
    func installSocketListener(_ listener: VZVirtioSocketListener, port: UInt32) throws
}

protocol VirtualMachineProviding {
    func makeVirtualMachine(configuration: VZVirtualMachineConfiguration) throws -> VirtualMachineControlling
}

enum VirtualizationLinuxBootDriverError: Error {
    case socketDeviceMissing
    case startTimedOut(String)
}

private final class VZVirtualMachineController: VirtualMachineControlling {
    private let machine: VZVirtualMachine
    private let queue: DispatchQueue
    private let queueKey = DispatchSpecificKey<Bool>()

    init(machine: VZVirtualMachine, queue: DispatchQueue) {
        self.machine = machine
        self.queue = queue
        self.queue.setSpecific(key: queueKey, value: true)
    }

    func start(timeoutSeconds: TimeInterval) throws {
        let semaphore = DispatchSemaphore(value: 0)
        let resultLock = NSLock()
        var startResult: Result<Void, Error>?
        queue.async {
            self.machine.start { result in
                resultLock.lock()
                startResult = result
                resultLock.unlock()
                semaphore.signal()
            }
        }
        let timeout = max(timeoutSeconds, 0)
        let timedOut = semaphore.wait(timeout: .now() + timeout) == .timedOut
        if timedOut {
            throw VirtualizationLinuxBootDriverError.startTimedOut("vm_start_timed_out")
        }
        resultLock.lock()
        let result = startResult
        resultLock.unlock()
        if case let .failure(error) = result {
            throw error
        }
    }

    func stop() throws {
        try syncOnQueue {
            if machine.canRequestStop {
                try machine.requestStop()
            }
        }
    }

    func installSocketListener(_ listener: VZVirtioSocketListener, port: UInt32) throws {
        try syncOnQueue {
            guard let socketDevice = machine.socketDevices.first(where: { $0 is VZVirtioSocketDevice }) as? VZVirtioSocketDevice else {
                throw VirtualizationLinuxBootDriverError.socketDeviceMissing
            }
            socketDevice.setSocketListener(listener, forPort: port)
        }
    }

    private func syncOnQueue<T>(_ work: () throws -> T) throws -> T {
        if DispatchQueue.getSpecific(key: queueKey) == true {
            return try work()
        }
        return try queue.sync(execute: work)
    }
}

struct VZVirtualMachineProvider: VirtualMachineProviding {
    func makeVirtualMachine(configuration: VZVirtualMachineConfiguration) throws -> VirtualMachineControlling {
        try configuration.validate()
        let queue = DispatchQueue(label: "tldw.sandbox.vz.vm.\(UUID().uuidString)")
        let machine = VZVirtualMachine(configuration: configuration, queue: queue)
        return VZVirtualMachineController(machine: machine, queue: queue)
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

    @discardableResult
    func boot(vmID: String, templatePath: String, workspacePath: String, startupTimeoutSeconds: TimeInterval) throws -> VMResourceSnapshot {
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
        let resourceSnapshot = VMResourceSnapshot(
            cpuCount: configuration.cpuCount,
            memorySizeBytes: configuration.memorySize
        )
        do {
            try machine.installSocketListener(listener.listener, port: spec.vsockPort)
            try machine.start(timeoutSeconds: startupTimeoutSeconds)
            machines[vmID] = machine
            return resourceSnapshot
        } catch {
            try? machine.stop()
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
