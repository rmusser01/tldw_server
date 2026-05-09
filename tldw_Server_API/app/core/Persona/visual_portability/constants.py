"""Constants for persona visual pack portability archives."""

PERSONA_VISUAL_PACK_SCHEMA_VERSION = "tldw.persona_visual_pack.v1"
PERSONA_VISUAL_PACK_EXTENSION = ".tldw-persona-vpack"
MANIFEST_PATH = "manifest.json"
CHECKSUMS_PATH = "checksums/sha256.json"
ALLOWED_TOP_LEVEL_DIRS = {"assets", "metadata", "checksums", "signatures"}
ALLOWED_TOP_LEVEL_FILES = {"manifest.json", "README.md"}
REQUIRED_MEMBERS = {
    "manifest.json",
    "metadata/pack.json",
    "metadata/assets.json",
    "checksums/sha256.json",
}
TRUST_MODE_TRUSTED_RESTORE = "trusted_restore"
TRUST_MODE_UNTRUSTED_IMPORT = "untrusted_import"
ASSET_BYTES_STATUS_PRESENT = "present"
ASSET_BYTES_STATUS_MISSING = "missing"

