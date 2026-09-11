"""Cross-user isolation tests for legacy and canonical shared-workspace access."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.core.Sharing.shared_workspace_access_service import (
    SharedWorkspaceAccessService,
    SharedWorkspaceNotFound,
)
from tldw_Server_API.app.core.Sharing.shared_workspace_resolver import (
    SharedWorkspaceContext,
    SharedWorkspaceDBResolver,
)

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_canonical_recipient_access_rejects_nonmember_before_owner_database(
    repo,
    sharing_db,
) -> None:
    sharing_db.execute(
        "INSERT INTO users (id, username, email, password_hash) VALUES (3, 'eve', 'eve@test.com', 'hash')"
    )
    sharing_db.commit()
    share = await repo.create_share(
        workspace_id="ws-canonical-cross-user",
        owner_user_id=1,
        share_scope_type="team",
        share_scope_id=10,
        access_level="view_chat",
        created_by=1,
    )
    owner_loader_calls: list[int] = []

    class _UsersRepo:
        async def get_user_by_id(self, user_id: int):
            return {"id": user_id, "username": "owner"}

    async def _load_owner(owner_user_id: int):
        owner_loader_calls.append(owner_user_id)
        return MagicMock()

    service = SharedWorkspaceAccessService(repo, _UsersRepo(), _load_owner)

    with pytest.raises(SharedWorkspaceNotFound):
        await service.resolve(share_id=share["id"], recipient_user_id=3)

    assert owner_loader_calls == []


@pytest.fixture
def resolver(repo):
    return SharedWorkspaceDBResolver(repo)


@pytest.fixture
def mock_dbs():
    """Create mock DB instances for source and conversation."""
    source_chacha = MagicMock()
    source_media = MagicMock()
    conversation_chacha = MagicMock()
    return source_chacha, source_media, conversation_chacha


@pytest.mark.asyncio
async def test_resolve_returns_context(repo, resolver, mock_dbs):
    source_chacha, source_media, conversation_chacha = mock_dbs
    share = await repo.create_share(
        workspace_id="ws-resolve",
        owner_user_id=1,
        share_scope_type="team",
        share_scope_id=10,
        access_level="view_chat",
        created_by=1,
    )

    ctx = await resolver.resolve(
        share["id"],
        accessor_user_id=2,
        source_chacha_db=source_chacha,
        source_media_db=source_media,
        conversation_chacha_db=conversation_chacha,
    )

    assert isinstance(ctx, SharedWorkspaceContext)
    assert ctx.workspace_id == "ws-resolve"
    assert ctx.owner_user_id == 1
    assert ctx.accessor_user_id == 2
    assert ctx.access_level == "view_chat"
    assert ctx.embedding_namespace == "1"
    assert ctx.source_chacha_db is source_chacha
    assert ctx.source_media_db is source_media
    assert ctx.conversation_chacha_db is conversation_chacha


@pytest.mark.asyncio
async def test_resolve_revoked_share_raises(repo, resolver, mock_dbs):
    source_chacha, source_media, conversation_chacha = mock_dbs
    share = await repo.create_share(
        workspace_id="ws-revoked",
        owner_user_id=1,
        share_scope_type="team",
        share_scope_id=10,
        created_by=1,
    )
    await repo.revoke_share(share["id"])

    with pytest.raises(PermissionError, match="revoked"):
        await resolver.resolve(
            share["id"],
            accessor_user_id=2,
            source_chacha_db=source_chacha,
            source_media_db=source_media,
            conversation_chacha_db=conversation_chacha,
        )


@pytest.mark.asyncio
async def test_resolve_nonexistent_share_raises(resolver, mock_dbs):
    source_chacha, source_media, conversation_chacha = mock_dbs
    with pytest.raises(PermissionError, match="not found"):
        await resolver.resolve(
            9999,
            accessor_user_id=2,
            source_chacha_db=source_chacha,
            source_media_db=source_media,
            conversation_chacha_db=conversation_chacha,
        )


class TestWritePermissions:
    def test_view_chat_blocks_add_source(self, repo):
        ctx = SharedWorkspaceContext(
            share_id=1,
            workspace_id="ws-1",
            owner_user_id=1,
            accessor_user_id=2,
            access_level="view_chat",
            allow_clone=True,
            source_chacha_db=MagicMock(),
            source_media_db=MagicMock(),
            conversation_chacha_db=MagicMock(),
            embedding_namespace="1",
        )
        with pytest.raises(PermissionError, match="does not allow adding sources"):
            SharedWorkspaceDBResolver.check_write_permission(ctx, "add_source")

    def test_view_chat_blocks_edit(self):
        ctx = SharedWorkspaceContext(
            share_id=1,
            workspace_id="ws-1",
            owner_user_id=1,
            accessor_user_id=2,
            access_level="view_chat",
            allow_clone=True,
            source_chacha_db=MagicMock(),
            source_media_db=MagicMock(),
            conversation_chacha_db=MagicMock(),
            embedding_namespace="1",
        )
        with pytest.raises(PermissionError, match="does not allow edit_source"):
            SharedWorkspaceDBResolver.check_write_permission(ctx, "edit_source")

    def test_view_chat_add_allows_add_source(self):
        ctx = SharedWorkspaceContext(
            share_id=1,
            workspace_id="ws-1",
            owner_user_id=1,
            accessor_user_id=2,
            access_level="view_chat_add",
            allow_clone=True,
            source_chacha_db=MagicMock(),
            source_media_db=MagicMock(),
            conversation_chacha_db=MagicMock(),
            embedding_namespace="1",
        )
        # Should not raise
        SharedWorkspaceDBResolver.check_write_permission(ctx, "add_source")

    def test_view_chat_add_blocks_edit(self):
        ctx = SharedWorkspaceContext(
            share_id=1,
            workspace_id="ws-1",
            owner_user_id=1,
            accessor_user_id=2,
            access_level="view_chat_add",
            allow_clone=True,
            source_chacha_db=MagicMock(),
            source_media_db=MagicMock(),
            conversation_chacha_db=MagicMock(),
            embedding_namespace="1",
        )
        with pytest.raises(PermissionError, match="does not allow delete_source"):
            SharedWorkspaceDBResolver.check_write_permission(ctx, "delete_source")

    def test_full_edit_allows_everything(self):
        ctx = SharedWorkspaceContext(
            share_id=1,
            workspace_id="ws-1",
            owner_user_id=1,
            accessor_user_id=2,
            access_level="full_edit",
            allow_clone=True,
            source_chacha_db=MagicMock(),
            source_media_db=MagicMock(),
            conversation_chacha_db=MagicMock(),
            embedding_namespace="1",
        )
        SharedWorkspaceDBResolver.check_write_permission(ctx, "add_source")
        SharedWorkspaceDBResolver.check_write_permission(ctx, "edit_source")
        SharedWorkspaceDBResolver.check_write_permission(ctx, "delete_source")
        SharedWorkspaceDBResolver.check_write_permission(ctx, "edit_workspace")


class TestClonePermissions:
    def test_can_clone_when_allowed(self):
        ctx = SharedWorkspaceContext(
            share_id=1,
            workspace_id="ws-1",
            owner_user_id=1,
            accessor_user_id=2,
            access_level="view_chat",
            allow_clone=True,
            source_chacha_db=MagicMock(),
            source_media_db=MagicMock(),
            conversation_chacha_db=MagicMock(),
            embedding_namespace="1",
        )
        assert SharedWorkspaceDBResolver.can_clone(ctx) is True

    def test_cannot_clone_when_disallowed(self):
        ctx = SharedWorkspaceContext(
            share_id=1,
            workspace_id="ws-1",
            owner_user_id=1,
            accessor_user_id=2,
            access_level="full_edit",
            allow_clone=False,
            source_chacha_db=MagicMock(),
            source_media_db=MagicMock(),
            conversation_chacha_db=MagicMock(),
            embedding_namespace="1",
        )
        assert SharedWorkspaceDBResolver.can_clone(ctx) is False


class TestEmbeddingNamespace:
    @pytest.mark.asyncio
    async def test_namespace_is_owner_user_id(self, repo, resolver, mock_dbs):
        source_chacha, source_media, conversation_chacha = mock_dbs
        share = await repo.create_share(
            workspace_id="ws-ns",
            owner_user_id=1,
            share_scope_type="team",
            share_scope_id=10,
            created_by=1,
        )
        ctx = await resolver.resolve(
            share["id"],
            accessor_user_id=2,
            source_chacha_db=source_chacha,
            source_media_db=source_media,
            conversation_chacha_db=conversation_chacha,
        )
        # embedding_namespace should be the owner's user_id as string
        assert ctx.embedding_namespace == "1"
        # This would be passed as index_namespace to unified_pipeline


class TestAccessLevelChanges:
    @pytest.mark.asyncio
    async def test_updated_access_reflected(self, repo, resolver, mock_dbs):
        source_chacha, source_media, conversation_chacha = mock_dbs
        share = await repo.create_share(
            workspace_id="ws-update",
            owner_user_id=1,
            share_scope_type="team",
            share_scope_id=10,
            access_level="view_chat",
            created_by=1,
        )
        # Initially view_chat
        ctx = await resolver.resolve(
            share["id"],
            accessor_user_id=2,
            source_chacha_db=source_chacha,
            source_media_db=source_media,
            conversation_chacha_db=conversation_chacha,
        )
        assert ctx.access_level == "view_chat"

        # Update to full_edit
        await repo.update_share(share["id"], access_level="full_edit")
        ctx2 = await resolver.resolve(
            share["id"],
            accessor_user_id=2,
            source_chacha_db=source_chacha,
            source_media_db=source_media,
            conversation_chacha_db=conversation_chacha,
        )
        assert ctx2.access_level == "full_edit"
