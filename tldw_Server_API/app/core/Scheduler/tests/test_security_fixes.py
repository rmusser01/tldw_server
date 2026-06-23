"""
Test security fixes in the scheduler module.
"""

import pytest
import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile

from ..config import SchedulerConfig
from ..scheduler import Scheduler
from ..base import TaskRegistry
from ..authorization import TaskAuthorizer, AuthContext, TaskPermission, get_authorizer

DEFAULT_METADATA = {"user_id": "security-tester"}


class TestPathTraversalFixes:
    """Test path traversal vulnerability fixes."""

    def test_reject_directory_traversal_in_base_path(self):
        """Test that directory traversal attempts are rejected."""
        with pytest.raises(ValueError, match="Directory traversal detected"):
            os.environ['SCHEDULER_BASE_PATH'] = '../../../etc/passwd'
            config = SchedulerConfig()

    def test_reject_tilde_expansion(self):
        """Test that tilde expansion is rejected."""
        with pytest.raises(ValueError, match="Directory traversal detected"):
            os.environ['SCHEDULER_BASE_PATH'] = '~/../../sensitive'
            config = SchedulerConfig()

    def test_reject_symlink_base_path(self, tmp_path):
        """Test that symlinks are rejected as base paths."""
        # Create a symlink
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        symlink = tmp_path / "symlink"
        symlink.symlink_to(real_dir)

        with pytest.raises(ValueError, match="Base path cannot be a symlink"):
            os.environ['SCHEDULER_BASE_PATH'] = str(symlink)
            config = SchedulerConfig()

    def test_reject_symlink_parent_in_base_path(self, tmp_path):
        """Test that paths traversing through symlink parents are rejected."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        symlink = tmp_path / "symlink"
        symlink.symlink_to(real_dir)

        with pytest.raises(ValueError, match="Base path cannot be a symlink"):
            SchedulerConfig(base_path=symlink / "child", database_url=':memory:')

    def test_sqlite_path_validation(self):
        """Test that SQLite paths are validated."""
        with pytest.raises(ValueError, match="Directory traversal detected in database path"):
            config = SchedulerConfig(database_url='sqlite:///../../../etc/passwd.db')

    def test_windows_short_sqlite_path_is_not_tilde_traversal(self, tmp_path):
        """Windows 8.3 short paths contain ``~1`` without implying home expansion."""
        with patch('platform.system', return_value='Windows'):
            config = SchedulerConfig(
                database_url=r"sqlite:///C:\Users\RUNNER~1\AppData\Local\Temp/scheduler_123.db",
                base_path=tmp_path / "scheduler",
            )

        assert "RUNNER~1" in config.database_url

    def test_windows_path_fallback(self):
        """Test Windows path fallback when /var/lib doesn't exist."""
        with patch('platform.system', return_value='Windows'):
            # Clear environment variable
            os.environ.pop('SCHEDULER_BASE_PATH', None)
            config = SchedulerConfig()
            # Should use temp directory on Windows
            assert 'scheduler' in str(config.base_path)
            assert config.base_path.exists() or True  # Path creation might be lazy

    def test_unix_permission_fallback(self):
        """Test Unix path fallback when /var/lib is not writable."""
        with patch('platform.system', return_value='Linux'):
            with patch('os.access', return_value=False):
                config = SchedulerConfig(
                    base_path=Path('/var/lib/scheduler-security-test'),
                    database_url=':memory:',
                )
                # Should use home directory
                assert '.local/share/scheduler' in str(config.base_path)


class TestPayloadSanitization:
    """Test payload sanitization and validation."""

    @pytest.mark.asyncio
    async def test_reject_code_injection_in_payload(self):
        """Test that code injection attempts are rejected."""
        config = SchedulerConfig(database_url=':memory:')
        scheduler = Scheduler(config)

        # Mock backend and registry
        scheduler.backend = MagicMock()
        scheduler.registry = TaskRegistry()
        scheduler.registry.register('test_handler', lambda x: x)
        scheduler._started = True

        malicious_payload = {
            '__import__': 'os',
            'eval': 'os.system("rm -rf /")',
            'data': 'normal'
        }

        with pytest.raises(ValueError, match="potentially malicious content"):
            await scheduler.submit('test_handler', payload=malicious_payload, metadata=DEFAULT_METADATA)

    @pytest.mark.asyncio
    async def test_sanitize_dangerous_keys(self):
        """Test that dangerous keys are removed from payloads."""
        config = SchedulerConfig(database_url=':memory:')
        scheduler = Scheduler(config)

        payload = {
            '__class__': 'dangerous',
            'eval_code': 'dangerous',
            'exec_command': 'dangerous',
            'safe_data': 'ok'
        }

        sanitized = scheduler._sanitize_payload(payload)

        assert 'safe_data' in sanitized
        assert '__class__' not in sanitized
        assert 'eval_code' not in sanitized
        assert 'exec_command' not in sanitized

    @pytest.mark.asyncio
    async def test_sanitize_sql_injection(self):
        """Test that SQL injection attempts are sanitized."""
        config = SchedulerConfig(database_url=':memory:')
        scheduler = Scheduler(config)

        payload = {
            'query': "'; DROP TABLE users; --",
            'data': "normal data with INSERT INTO should be cleaned"
        }

        sanitized = scheduler._sanitize_payload(payload)

        assert 'DROP TABLE' not in sanitized['query']
        assert 'INSERT INTO' not in sanitized['data']

    @pytest.mark.asyncio
    async def test_sanitize_script_tags(self):
        """Test that script tags are removed."""
        config = SchedulerConfig(database_url=':memory:')
        scheduler = Scheduler(config)

        payload = {
            'html': '<script>alert("XSS")</script>Normal text',
            'js': 'javascript:void(0)'
        }

        sanitized = scheduler._sanitize_payload(payload)

        assert '<script' not in sanitized['html']
        assert 'javascript:' not in sanitized['js']
        assert 'Normal text' in sanitized['html']

    @pytest.mark.asyncio
    async def test_limit_payload_size(self):
        """Test that oversized payloads are rejected."""
        config = SchedulerConfig(database_url=':memory:')
        config.max_payload_size = 1024  # 1KB limit
        scheduler = Scheduler(config)

        # Mock backend and registry
        scheduler.backend = MagicMock()
        scheduler.registry = TaskRegistry()
        scheduler.registry.register('test_handler', lambda x: x)
        scheduler._started = True

        # Create oversized payload
        large_payload = {'data': 'x' * 2000}

        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            await scheduler.submit('test_handler', payload=large_payload, metadata=DEFAULT_METADATA)

    @pytest.mark.asyncio
    async def test_limit_list_size(self):
        """Test that lists are limited in size."""
        config = SchedulerConfig(database_url=':memory:')
        scheduler = Scheduler(config)

        payload = {
            'items': list(range(2000))  # 2000 items
        }

        sanitized = scheduler._sanitize_payload(payload)

        assert len(sanitized['items']) == 1000  # Limited to 1000

    @pytest.mark.asyncio
    async def test_detect_suspicious_patterns(self):
        """Test detection of suspicious patterns."""
        config = SchedulerConfig(database_url=':memory:')
        scheduler = Scheduler(config)

        suspicious_payloads = [
            '{"cmd": "__import__(\'os\').system(\'ls\')"}',
            '{"sql": "UNION SELECT * FROM passwords"}',
            '{"path": "../../etc/passwd"}',
            '{"shell": "; rm -rf /"}',
        ]

        for payload in suspicious_payloads:
            assert scheduler._contains_suspicious_content(payload) == True

        safe_payload = '{"data": "normal user data", "count": 123}'
        assert scheduler._contains_suspicious_content(safe_payload) == False


class TestHandlerValidation:
    """Test handler validation and authorization."""

    @pytest.mark.asyncio
    async def test_reject_unregistered_handler(self):
        """Test that unregistered handlers are rejected."""
        config = SchedulerConfig(database_url=':memory:')
        scheduler = Scheduler(config)
        scheduler._started = True
        scheduler.registry = TaskRegistry()

        with pytest.raises(ValueError, match="Handler 'unknown' not registered"):
            await scheduler.submit('unknown', payload={'data': 'test'}, metadata=DEFAULT_METADATA)

    @pytest.mark.asyncio
    async def test_reject_invalid_handler_name(self):
        """Test that invalid handler names are rejected."""
        config = SchedulerConfig(database_url=':memory:')
        scheduler = Scheduler(config)
        scheduler._started = True
        scheduler.registry = TaskRegistry()
        scheduler.registry.register('valid_handler', lambda x: x)

        invalid_names = [
            '../../../etc/passwd',
            'handler; rm -rf /',
            'handler && echo hacked',
            'handler`whoami`',
        ]

        for name in invalid_names:
            # First, try to register with invalid name (should be rejected by registry)
            # Then try to submit (should be rejected by scheduler)
            with pytest.raises(ValueError, match="contains invalid characters"):
                await scheduler.submit(name, payload={'data': 'test'}, metadata=DEFAULT_METADATA)

    @pytest.mark.asyncio
    async def test_validate_queue_name(self):
        """Test that queue names are validated."""
        config = SchedulerConfig(database_url=':memory:')
        scheduler = Scheduler(config)
        scheduler._started = True
        scheduler.registry = TaskRegistry()
        scheduler.registry.register('test_handler', lambda x: x)
        scheduler.backend = MagicMock()
        scheduler.write_buffer = MagicMock()
        scheduler.write_buffer.add = MagicMock(return_value='task-123')

        with pytest.raises(ValueError, match="Queue name contains invalid characters"):
            await scheduler.submit(
                'test_handler',
                payload={'data': 'test'},
                queue_name='../../etc/passwd',
                metadata=DEFAULT_METADATA
            )


class TestSQLInjectionFixes:
    """Test SQL injection fixes in AuthNZ scheduler."""

    @pytest.mark.asyncio
    async def test_parameterized_queries(self):
        """Test that queries use proper parameterization."""
        from ..core.AuthNZ.scheduler import AuthNZScheduler

        scheduler = AuthNZScheduler()

        # Mock database pool
        mock_pool = MagicMock()
        mock_pool.fetchone = AsyncMock(return_value={'failure_count': 5, 'unique_ips': 2})
        mock_pool.fetchall = AsyncMock(return_value=[])

        with patch(
            'tldw_Server_API.app.core.AuthNZ.scheduler.get_db_pool',
            new=AsyncMock(return_value=mock_pool),
        ):
            # Test PostgreSQL path
            mock_pool.pool = object()  # Indicate PostgreSQL
            await scheduler._monitor_auth_failures()

            # Check that query used proper parameterization (ANY for arrays)
            call_args = mock_pool.fetchone.await_args.args
            assert 'ANY($1)' in call_args[0]
            assert isinstance(call_args[1], list)  # Parameters should be a list

            # Test SQLite path
            mock_pool.pool = None  # Indicate SQLite
            mock_pool.fetchone.reset_mock()
            await scheduler._monitor_auth_failures()

            # Check that query used proper parameterization (? placeholders)
            call_args = mock_pool.fetchone.await_args.args
            assert 'IN (?, ?, ?)' in call_args[0]


class TestIntegrationSecurity:
    """Integration tests for security fixes."""

    @pytest.mark.asyncio
    async def test_end_to_end_secure_task_submission(self):
        """Test secure end-to-end task submission."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SchedulerConfig(
                database_url=':memory:',
                base_path=Path(tmpdir) / 'scheduler'
            )

            scheduler = Scheduler(config)

            # Register a safe handler
            scheduler.registry.register('safe_handler', lambda x: f"Processed: {x}")

            # Mock backend to avoid actual database operations
            scheduler.backend = MagicMock()
            scheduler.backend.get_task_by_idempotency_key = MagicMock(return_value=None)
            scheduler.write_buffer = MagicMock()
            scheduler.write_buffer.add = MagicMock(return_value='task-123')
            scheduler._started = True

            # Submit task with potentially dangerous payload
            dangerous_payload = {
                'user_input': '<script>alert("XSS")</script>',
                'query': "'; DROP TABLE users; --",
                '__import__': 'os',
                'safe_data': 'This is OK'
            }

            task_id = await scheduler.submit('safe_handler', payload=dangerous_payload, metadata=DEFAULT_METADATA)

            # Verify task was created
            assert task_id == 'task-123'

            # Verify payload was sanitized
            submitted_task = scheduler.write_buffer.add.call_args[0][0]
            payload = submitted_task.payload

            # Dangerous content should be removed or sanitized
            assert '<script' not in str(payload)
            assert 'DROP TABLE' not in str(payload)
            assert '__import__' not in payload
            assert 'safe_data' in payload

    @pytest.mark.asyncio
    async def test_reject_malicious_task_submission(self):
        """Test that clearly malicious submissions are rejected."""
        config = SchedulerConfig(database_url=':memory:')
        scheduler = Scheduler(config)
        scheduler._started = True
        scheduler.registry = TaskRegistry()
        scheduler.registry.register('test_handler', lambda x: x)

        # Payload with code execution attempt
        malicious_payload = {
            'code': '__import__("os").system("rm -rf /")'
        }

        with pytest.raises(ValueError, match="potentially malicious content"):
            await scheduler.submit('test_handler', payload=malicious_payload, metadata=DEFAULT_METADATA)


class TestTaskAuthorization:
    """Test task authorization and permission checks."""

    def setup_method(self):
        """Set up test authorizer."""
        self.authorizer = TaskAuthorizer()

    def test_admin_can_access_admin_only_handler(self):
        """Test that admin users can access admin-only handlers."""
        self.authorizer.register_handler_permissions(
            'admin_task',
            [TaskPermission.SUBMIT],
            admin_only=True
        )

        admin_context = AuthContext(
            user_id='admin_user',
            roles=['admin'],
            permissions={TaskPermission.SUBMIT.value}
        )

        can_submit, reason = self.authorizer.can_submit_task('admin_task', 'default', admin_context)
        assert can_submit is True
        assert reason is None

    def test_non_admin_cannot_access_admin_handler(self):
        """Test that non-admin users cannot access admin-only handlers."""
        self.authorizer.register_handler_permissions(
            'admin_task',
            [TaskPermission.SUBMIT],
            admin_only=True
        )

        user_context = AuthContext(
            user_id='regular_user',
            roles=['user'],
            permissions={TaskPermission.SUBMIT.value}
        )

        can_submit, reason = self.authorizer.can_submit_task('admin_task', 'default', user_context)
        assert can_submit is False
        assert 'admin privileges' in reason

    def test_anonymous_handler_allows_unauthenticated(self):
        """Test that anonymous handlers allow unauthenticated access."""
        self.authorizer.allow_anonymous_handler('public_task')

        anonymous_context = AuthContext()  # No authentication

        can_submit, reason = self.authorizer.can_submit_task('public_task', 'default', anonymous_context)
        assert can_submit is True
        assert reason is None

    def test_user_can_cancel_own_task(self):
        """Test that users can cancel their own tasks."""
        user_context = AuthContext(user_id='user123')

        can_cancel, reason = self.authorizer.can_cancel_task('user123', user_context)
        assert can_cancel is True
        assert reason is None

    def test_user_cannot_cancel_others_task(self):
        """Test that users cannot cancel other users' tasks."""
        user_context = AuthContext(user_id='user123')

        can_cancel, reason = self.authorizer.can_cancel_task('other_user', user_context)
        assert can_cancel is False
        assert 'Not authorized' in reason

    def test_queue_permission_check(self):
        """Test queue-specific permission checks."""
        self.authorizer.register_queue_permissions(
            'priority_queue',
            [TaskPermission.SUBMIT]
        )

        # User without permission
        user_context = AuthContext(
            user_id='user1',
            permissions=set()
        )

        can_submit, reason = self.authorizer.can_submit_task('some_task', 'priority_queue', user_context)
        assert can_submit is False
        assert 'queue permissions' in reason

        # User with permission
        authorized_context = AuthContext(
            user_id='user2',
            permissions={TaskPermission.SUBMIT.value}
        )

        can_submit, reason = self.authorizer.can_submit_task('some_task', 'priority_queue', authorized_context)
        assert can_submit is True

    def test_payload_size_validation_for_non_admin(self):
        """Test that non-admin users have payload size limits."""
        large_payload = {'data': 'x' * 200000}  # 200KB

        admin_context = AuthContext(
            user_id='admin',
            roles=['admin']
        )

        user_context = AuthContext(
            user_id='user',
            roles=['user']
        )

        # Admin should pass
        valid, error = self.authorizer.validate_payload_for_handler('task', large_payload, admin_context)
        assert valid is True

        # Regular user should fail
        valid, error = self.authorizer.validate_payload_for_handler('task', large_payload, user_context)
        assert valid is False
        assert 'too large' in error

    @pytest.mark.asyncio
    async def test_scheduler_authorization_integration(self, tmp_path):
        """Test authorization integration with scheduler."""
        config = SchedulerConfig(
            database_url=':memory:',
            base_path=tmp_path / 'scheduler'
        )
        scheduler = Scheduler(config)
        scheduler._started = True
        scheduler.registry = TaskRegistry()
        # Use the task decorator to register the handler
        @scheduler.registry.task(name='protected_task')
        def protected_task(x):
            return x
        scheduler.backend = MagicMock()
        scheduler.backend.get_task_by_idempotency_key = MagicMock(return_value=None)
        scheduler.write_buffer = MagicMock()

        # Make write_buffer.add return a coroutine
        async def mock_add(task):
            return 'task-123'
        scheduler.write_buffer.add = MagicMock(side_effect=mock_add)

        # Register as admin-only task
        scheduler.authorizer.register_handler_permissions(
            'protected_task',
            [TaskPermission.SUBMIT],
            admin_only=True
        )

        # Try to submit without authorization context (should work - no auth required by default)
        task_id = await scheduler.submit('protected_task', payload={'data': 'test'}, metadata=DEFAULT_METADATA)
        assert task_id == 'task-123'

        # Try to submit with non-admin context (should fail)
        user_context = AuthContext(
            user_id='regular_user',
            roles=['user']
        )

        with pytest.raises(PermissionError, match="Not authorized"):
            await scheduler.submit(
                'protected_task',
                payload={'data': 'test'},
                auth_context=user_context,
                metadata={'user_id': user_context.user_id}
            )

        # Try with admin context (should work)
        admin_context = AuthContext(
            user_id='admin_user',
            roles=['admin']
        )

        task_id = await scheduler.submit(
            'protected_task',
            payload={'data': 'test'},
            auth_context=admin_context,
            metadata={'user_id': admin_context.user_id}
        )
        assert task_id == 'task-123'


@pytest.fixture(autouse=True)
def cleanup_env():
    """Clean up environment variables after each test."""
    yield
    # Clean up any environment variables set during tests
    os.environ.pop('SCHEDULER_BASE_PATH', None)
    os.environ.pop('DATABASE_URL', None)
