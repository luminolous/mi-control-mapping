"""Session-wide test setup.

MNE resolves its configuration directory from the user's home, which on Windows
means `USERPROFILE`. Some `make` builds do not pass that variable through to
recipes, so `pytest` passes and `make test` fails on the same code.

Pointing MNE at a scratch directory removes the dependency entirely. The test
suite then behaves the same under `make`, under a bare `pytest`, and in CI, and
it never reads or writes the developer's real MNE configuration.

This must run before `mne` is imported, which is why it lives at module scope in
conftest rather than in a fixture.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_FAKE_HOME_VAR = "_MNE_FAKE_HOME_DIR"

if _FAKE_HOME_VAR not in os.environ:
    scratch = Path(tempfile.gettempdir()) / "micm-test-home"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ[_FAKE_HOME_VAR] = str(scratch)
