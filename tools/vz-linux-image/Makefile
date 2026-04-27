ROOTFS ?=
BUNDLE_DIR ?=
KERNEL ?=
ROOTFS_IMAGE ?=
INITRD ?=

.PHONY: install-agent smoke-check build-bundle build-debian-bundle

install-agent:
	./scripts/install-agent.sh "$(ROOTFS)"

smoke-check:
	TLDW_VZ_LINUX_IMAGE_ROOTFS="$(ROOTFS)" bash ./scripts/smoke-check.sh

build-bundle:
	TLDW_VZ_LINUX_IMAGE_ROOTFS="$(ROOTFS)" \
	TLDW_VZ_LINUX_BUNDLE_KERNEL="$(KERNEL)" \
	TLDW_VZ_LINUX_BUNDLE_ROOTFS_IMAGE="$(ROOTFS_IMAGE)" \
	TLDW_VZ_LINUX_BUNDLE_INITRD="$(INITRD)" \
	bash ./scripts/build-bundle.sh "$(BUNDLE_DIR)"

build-debian-bundle:
	bash ./scripts/build-debian-bundle.sh --output-dir "$(BUNDLE_DIR)"
