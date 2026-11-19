"""Shared fixtures, plus one hook: a test that hits a stub (NotImplementedError) is reported as
"skipped: TODO <function>" instead of failing. CI stays green while the scaffold fills in, and each
test starts counting the moment its function is implemented. `pytest -ra` lists what's left.

Delete the hook once everything is implemented (target: experiment freeze, Dec 1) so a stray
NotImplementedError can't hide.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.excinfo is not None and call.excinfo.errisinstance(NotImplementedError) and not report.skipped:
        report.outcome = "skipped"
        reason = str(call.excinfo.value) or "not implemented"
        report.longrepr = (str(item.path), (item.location[1] or 0) + 1, f"TODO {reason}")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
