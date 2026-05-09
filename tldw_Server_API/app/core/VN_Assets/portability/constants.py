"""Constants for VN asset pack portability archives."""

VNPACK_SCHEMA_VERSION = "tldw.vnpack.v1"
VNPACK_EXTENSION = ".tldw-vnpack"
MANIFEST_PATH = "manifest.json"
CHECKSUMS_PATH = "checksums/sha256.json"
ALLOWED_TOP_LEVEL_DIRS = {"assets", "metadata", "checksums", "signatures"}
ALLOWED_TOP_LEVEL_FILES = {"manifest.json", "README.md"}
REQUIRED_MEMBERS = {
    "manifest.json",
    "metadata/pack.json",
    "metadata/slots.json",
    "metadata/items.json",
    "checksums/sha256.json",
}
TRUST_MODE_TRUSTED_RESTORE = "trusted_restore"
TRUST_MODE_UNTRUSTED_IMPORT = "untrusted_import"
ASSET_BYTES_STATUS_PRESENT = "present"
ASSET_BYTES_STATUS_MISSING = "missing"
