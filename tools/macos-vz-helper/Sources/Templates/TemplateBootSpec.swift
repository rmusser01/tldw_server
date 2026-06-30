import Foundation

enum TemplateValidationStrength: String, Codable, Equatable {
    case strong
    case compatibility
}

enum TemplateBootLoaderKind: String, Codable, Equatable {
    case linuxKernel = "linux_kernel"
    case efi
}

struct BundleTemplateBootSpec: Equatable {
    let bootMode: TemplateBootMode
    let kernelPath: String
    let initrdPath: String?
    let rootfsPath: String
    let workspaceMountTag: String
    let vsockPort: UInt32
    let guestAgentPath: String
    let validationStrength: TemplateValidationStrength

    init(
        kernelPath: String,
        initrdPath: String?,
        rootfsPath: String,
        workspaceMountTag: String,
        vsockPort: UInt32,
        guestAgentPath: String,
        validationStrength: TemplateValidationStrength = .strong
    ) {
        self.bootMode = .bundle
        self.kernelPath = kernelPath
        self.initrdPath = initrdPath
        self.rootfsPath = rootfsPath
        self.workspaceMountTag = workspaceMountTag
        self.vsockPort = vsockPort
        self.guestAgentPath = guestAgentPath
        self.validationStrength = validationStrength
    }
}

struct RawDiskTemplateBootSpec: Equatable {
    let bootMode: TemplateBootMode
    let diskImagePath: String
    let workspaceMountTag: String
    let vsockPort: UInt32
    let guestAgentPath: String
    let bootLoaderKind: TemplateBootLoaderKind
    let validationStrength: TemplateValidationStrength

    init(
        diskImagePath: String,
        workspaceMountTag: String,
        vsockPort: UInt32,
        guestAgentPath: String,
        bootLoaderKind: TemplateBootLoaderKind = .efi,
        validationStrength: TemplateValidationStrength = .compatibility
    ) {
        self.bootMode = .rawDisk
        self.diskImagePath = diskImagePath
        self.workspaceMountTag = workspaceMountTag
        self.vsockPort = vsockPort
        self.guestAgentPath = guestAgentPath
        self.bootLoaderKind = bootLoaderKind
        self.validationStrength = validationStrength
    }
}

enum TemplateBootSpec: Equatable {
    case bundle(BundleTemplateBootSpec)
    case rawDisk(RawDiskTemplateBootSpec)

    var bootMode: TemplateBootMode {
        switch self {
        case let .bundle(spec):
            return spec.bootMode
        case let .rawDisk(spec):
            return spec.bootMode
        }
    }

    var validationStrength: TemplateValidationStrength {
        switch self {
        case let .bundle(spec):
            return spec.validationStrength
        case let .rawDisk(spec):
            return spec.validationStrength
        }
    }

    var workspaceMountTag: String {
        switch self {
        case let .bundle(spec):
            return spec.workspaceMountTag
        case let .rawDisk(spec):
            return spec.workspaceMountTag
        }
    }

    var vsockPort: UInt32 {
        switch self {
        case let .bundle(spec):
            return spec.vsockPort
        case let .rawDisk(spec):
            return spec.vsockPort
        }
    }

    var guestAgentPath: String {
        switch self {
        case let .bundle(spec):
            return spec.guestAgentPath
        case let .rawDisk(spec):
            return spec.guestAgentPath
        }
    }
}
