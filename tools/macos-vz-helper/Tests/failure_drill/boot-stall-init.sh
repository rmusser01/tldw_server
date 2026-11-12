#!/bin/sh
# Test-only PID 1 wrapper: no rootfs mount or agent startup precedes this proof.
set -eu
mkdir -p /proc
mount -t proc proc /proc
mkdir -p /sys
mount -t sysfs sysfs /sys
# Debian normally loads these in /init, which the stall deliberately never runs.
modprobe virtio_pci
modprobe virtio_console
IFS=: read -r major minor < /sys/class/tty/hvc0/dev
mknod /tldw-boot-console c "$major" "$minor"
exec > /tldw-boot-console 2>&1
read -r cmdline < /proc/cmdline
vm_id=
for argument in $cmdline; do
    case "$argument" in
        systemd.setenv=TLDW_AGENT_GUEST_VM_ID=*) vm_id=${argument#systemd.setenv=TLDW_AGENT_GUEST_VM_ID=} ;;
    esac
done
case "$vm_id" in
    ''|*[!a-zA-Z0-9._-]*) exit 1 ;;
esac
read -r nonce < /tldw-boot-nonce
read -r mode < /tldw-boot-mode
case "$nonce" in
    ''|*[!a-f0-9]*) exit 1 ;;
esac
case "$mode" in
    stall|continue) ;;
    *) exit 1 ;;
esac
printf 'TLDW_BOOT_PROOF {"nonce":"%s","vm_id":"%s","mode":"%s","stage":"initramfs"}\n' "$nonce" "$vm_id" "$mode"
if [ "$mode" = stall ]; then
    while :; do sleep 1; done
fi
umount /proc
umount /sys
exec /init.tldw-original
