import Foundation

enum TemplateResolutionError: Error, Equatable {
    case templateUnconfigured
    case templateMissing
    case bundleManifestMissing
    case bundleKernelMissing
    case bundleInitrdMissing
    case bundleRootfsMissing
    case bundleArtifactOutsideBundle(String)
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
        case .bundleArtifactOutsideBundle:
            return "vz_linux_bundle_artifact_outside_bundle"
        case .unsupportedBundleBootMode:
            return "vz_linux_bundle_boot_mode_unsupported"
        }
    }
}

struct BundleTemplateResolver {
    private func filesystemPath(for url: URL) -> String {
        url.path(percentEncoded: false)
    }

    private func artifactURL(in bundleURL: URL, artifactPath: String) throws -> URL {
        let rootURL = bundleURL.standardizedFileURL.resolvingSymlinksInPath()
        let candidateURL = rootURL
            .appendingPathComponent(artifactPath, isDirectory: false)
            .standardizedFileURL
            .resolvingSymlinksInPath()
        let rootPath = filesystemPath(for: rootURL)
        let candidatePath = filesystemPath(for: candidateURL)
        let rootPrefix = rootPath.hasSuffix("/") ? rootPath : "\(rootPath)/"

        guard candidatePath.hasPrefix(rootPrefix) else {
            throw TemplateResolutionError.bundleArtifactOutsideBundle(artifactPath)
        }
        return candidateURL
    }

    func resolve(templatePath: String) throws -> TemplateBootSpec {
        let bundleURL = URL(fileURLWithPath: templatePath, isDirectory: true)
        let manifestURL = bundleURL.appendingPathComponent("manifest.json", isDirectory: false)

        guard FileManager.default.fileExists(atPath: filesystemPath(for: manifestURL)) else {
            throw TemplateResolutionError.bundleManifestMissing
        }

        let manifestData = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(TemplateManifest.self, from: manifestData)

        guard manifest.bootMode == .bundle else {
            throw TemplateResolutionError.unsupportedBundleBootMode(manifest.bootMode)
        }

        let kernelURL = try artifactURL(in: bundleURL, artifactPath: manifest.kernel)
        guard FileManager.default.fileExists(atPath: filesystemPath(for: kernelURL)) else {
            throw TemplateResolutionError.bundleKernelMissing
        }

        let rootfsURL = try artifactURL(in: bundleURL, artifactPath: manifest.rootfs)
        guard FileManager.default.fileExists(atPath: filesystemPath(for: rootfsURL)) else {
            throw TemplateResolutionError.bundleRootfsMissing
        }

        let initrdPath: String?
        if let initrd = manifest.initrd {
            let initrdURL = try artifactURL(in: bundleURL, artifactPath: initrd)
            guard FileManager.default.fileExists(atPath: filesystemPath(for: initrdURL)) else {
                throw TemplateResolutionError.bundleInitrdMissing
            }
            initrdPath = filesystemPath(for: initrdURL)
        } else {
            initrdPath = nil
        }

        return .bundle(
            BundleTemplateBootSpec(
                kernelPath: filesystemPath(for: kernelURL),
                initrdPath: initrdPath,
                rootfsPath: filesystemPath(for: rootfsURL),
                workspaceMountTag: manifest.workspaceMountTag,
                vsockPort: manifest.vsockPort,
                guestAgentPath: manifest.guestAgentPath
            )
        )
    }
}
