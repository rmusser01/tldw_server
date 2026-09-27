"""Closed users-column contracts shared by profile readers and write guards."""

PROFILE_VISIBLE_USER_FIELDS = frozenset(
    {
        "uuid",
        "username",
        "email",
        "role",
        "is_superuser",
        "is_active",
        "is_verified",
        "two_factor_enabled",
        "last_login",
        "storage_quota_mb",
        "storage_used_mb",
    }
)

# These fields are deliberately excluded from profile-version semantics. Unknown
# columns are never added implicitly; schema evolution must classify them here.
RAW_SAFE_USER_UPDATE_FIELDS = frozenset(
    {
        "password_hash",
        "totp_secret",
        "backup_codes",
        "email_verified",
        "updated_at",
        "password_changed_at",
        "failed_login_attempts",
        "is_locked",
        "locked_until",
        "metadata",
    }
)
