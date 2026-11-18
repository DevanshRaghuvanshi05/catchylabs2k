# tests/unit/conftest.py
# This subtree contains pure unit tests that do not touch any database.
# We declare stub fixtures here so that autouse fixtures from the parent
# tests/conftest.py (which require a live test DB) cannot apply to this
# subtree.
#
# pytest walks conftest.py files from the rootdir inward.  Fixtures defined
# in a closer conftest.py override those in a farther one for the same name.

import pytest


@pytest.fixture(autouse=True)
def _patch_session_local():
    """Override the DB-requiring autouse fixture from tests/conftest.py."""
    # No-op: unit tests never touch a database.
    yield
