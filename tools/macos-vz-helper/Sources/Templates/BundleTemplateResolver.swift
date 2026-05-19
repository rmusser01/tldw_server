import Foundation

enum TemplateResolutionError: Error, Equatable {
    case templateUnconfigured
    case templateMissing
    case bundleManifestMissing
    case bundleKernelMissing
    case bundleInitrdMissing
    case bundleRootfsMissing
    case unsupportedBundleBootMode(TemplateBootMode)

    func reason(for runtime: String) -> String {
        switch self {
        case .templateUnconfigured:
            return "template_unconfigured"
        case .templateMissing:
            return runtime == "vz_linux" ? "vz_linux_template_missing" : "template_missing"
        case .bundleManifestMissing:
            return "vz_linux_bundle_manifest_missing"
        case .bundleKernelMissing:
            return "vz_linux_bundle_kernel_missing"
        case .bundleInitrdMissing:
            return "vz_linux_bundle_initrd_missing"
        case .bundleRootfsMissing:
            return "vz_linux_bundle_rootfs_missing"
        case .unsupportedBundleBootMode:
            return "vz_linux_bundle_boot_mode_unsupported"
        }
    }
}

struct BundleTemplateResolver {
    func resolve(templatePath: String) throws -> TemplateBootSpec {
        let bundleURL = URL(fileURLWithPath: templatePath, isDirectory: true)
        let manifestURL = bundleURL.appendingPathComponent("manifest.json", isDirectory: false)

        guard FileManager.default.fileExists(atPath: manifestURL.path()) else {
            throw TemplateResolutionError.bundleManifestMissing
        }

        let manifestData = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(TemplateManifest.self, from: manifestData)

        guard manifest.bootMode == .bundle else {
            throw TemplateResolutionError.unsupportedBundleBootMode(manifest.bootMode)
        }

        let kernelURL = bundleURL.appendingPathComponent(manifest.kernel, isDirectory: false)
        guard FileManager.default.fileExists(atPath: kernelURL.path()) else {
            throw TemplateResolutionError.bundleKernelMissing
        }

        let rootfsURL = bundleURL.appendingPathComponent(manifest.rootfs, isDirectory: false)
        guard FileManager.default.fileExists(atPath: rootfsURL.path()) else {
            throw TemplateResolutionError.bundleRootfsMissing
        }

        let initrdPath: String?
        if let initrd = manifest.initrd {
            let initrdURL = bundleURL.appendingPathComponent(initrd, isDirectory: false)
            guard FileManager.default.fileExists(atPath: initrdURL.path()) else {
                throw TemplateResolutionError.bundleInitrdMissing
            }
            initrdPath = initrdURL.path()
        } else {
            initrdPath = nil
        }

        return .bundle(
            BundleTemplateBootSpec(
                kernelPath: kernelURL.path(),
                initrdPath: initrdPath,
                rootfsPath: rootfsURL.path(),
                workspaceMountTag: manifest.workspaceMountTag,
                vsockPort: manifest.vsockPort,
                guestAgentPath: manifest.guestAgentPath
            )
        )
    }
}
