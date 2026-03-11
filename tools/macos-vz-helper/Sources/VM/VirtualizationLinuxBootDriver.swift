import Dispatch
import Foundation
import Virtualization

protocol VirtualMachineControlling {
    func start() throws
    func stop() throws
}

protocol VirtualMachineProviding {
    func makeVirtualMachine(configuration: VZVirtualMachineConfiguration) throws -> VirtualMachineControlling
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
    private var machines: [String: VirtualMachineControlling] = [:]

    init(
        templateValidator: TemplateValidator = TemplateValidator(),
        configurationBuilder: VZLinuxConfigurationBuilding = VZLinuxConfigurationBuilder(),
        machineProvider: VirtualMachineProviding = VZVirtualMachineProvider()
    ) {
        self.templateValidator = templateValidator
        self.configurationBuilder = configurationBuilder
        self.machineProvider = machineProvider
    }

    func boot(vmID: String, templatePath: String, workspacePath: String) throws {
        let spec = try templateValidator.resolve(runtime: "vz_linux", templatePath: templatePath)
        let configuration = try configurationBuilder.build(spec: spec, workspacePath: workspacePath)
        let machine = try machineProvider.makeVirtualMachine(configuration: configuration)
        try machine.start()
        machines[vmID] = machine
    }

    func stop(vmID: String) throws {
        guard let machine = machines.removeValue(forKey: vmID) else {
            return
        }
        try machine.stop()
    }
}
