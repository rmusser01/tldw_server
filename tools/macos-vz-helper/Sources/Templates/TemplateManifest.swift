import Foundation

enum TemplateBootMode: String, Codable, Equatable {
    case bundle
    case rawDisk = "raw_disk"
}

struct TemplateManifest: Codable, Equatable {
    let bundleVersion: String
    let bootMode: TemplateBootMode
    let kernel: String
    let initrd: String?
    let rootfs: String
    let guestAgentPath: String
    let workspaceMountTag: String
    let vsockPort: UInt32

    private enum CodingKeys: String, CodingKey {
        case bundleVersion = "bundle_version"
        case bootMode = "boot_mode"
        case kernel
        case initrd
        case rootfs
        case guestAgentPath = "guest_agent_path"
        case workspaceMountTag = "workspace_mount_tag"
        case vsockPort = "vsock_port"
    }
}
