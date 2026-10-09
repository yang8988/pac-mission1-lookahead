"""Pytest fixtures for stages 5-1/5-2 (taehyeon). Helpers: th_helpers.py."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from th_helpers import REPO, make_context  # noqa: E402
from pac_candidates import CandidateBackend, CandidateConfig  # noqa: E402


@pytest.fixture
def backend():
    return CandidateBackend(make_context(), CandidateConfig())


@pytest.fixture
def repo_root():
    return REPO
