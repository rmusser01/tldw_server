import Foundation

struct RawDiskTemplateResolver {
    private let workspaceMountTag = "workspace"
    private let vsockPort: UInt32 = 1024
    private let guestAgentPath = "/usr/local/bin/tldw-agent-guest"

    func resolve(templatePath: String) throws -> TemplateBootSpec {
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: templatePath, isDirectory: &isDirectory), !isDirectory.boolValue else {
            throw TemplateResolutionError.templateMissing
        }

        return .rawDisk(
            RawDiskTemplateBootSpec(
                diskImagePath: templatePath,
                workspaceMountTag: workspaceMountTag,
                vsockPort: vsockPort,
                guestAgentPath: guestAgentPath
            )
        )
    }
}
