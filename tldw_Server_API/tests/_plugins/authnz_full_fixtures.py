"""AuthNZ fixture bridge safe to register as a pytest plugin.

Do not register ``tldw_Server_API.tests.AuthNZ.conftest`` directly as a
pytest plugin. Pytest also auto-discovers that file as a conftest when the
AuthNZ suite is collected, and registering the same module twice breaks
collection.
"""

from tldw_Server_API.tests.AuthNZ.conftest import *  # noqa: F401,F403
