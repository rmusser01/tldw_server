ROOTFS ?=

.PHONY: install-agent smoke-check

install-agent:
	./scripts/install-agent.sh "$(ROOTFS)"

smoke-check:
	TLDW_VZ_LINUX_IMAGE_ROOTFS="$(ROOTFS)" bash ./scripts/smoke-check.sh
