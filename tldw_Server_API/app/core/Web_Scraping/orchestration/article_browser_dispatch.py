"""Compatibility exports for the shared same-connection CDP transfer helpers."""

from ..runtime.browser_dispatch import ChildSession as ChildSession
from ..runtime.browser_dispatch import ChildSessions as _ChildSessions
from ..runtime.browser_dispatch import TransferSessions as _TransferSessions
from .article_models import ArticleFailure


class ChildSessions(_ChildSessions):
    """Retain the article owner\'s existing failure contract."""

    def __init__(self, root, lease, fail):
        super().__init__(root, lease, fail, ArticleFailure)


class TransferSessions(_TransferSessions):
    """Retain article imports while sharing the existing CDP connection helper."""

    def __init__(self, root, lease, ledger, fail):
        super().__init__(root, lease, ledger, fail, ArticleFailure)
