"""Shared test fixtures."""

import pytest

import manifesto.workflow as workflow


@pytest.fixture(autouse=True)
def reset_workflow_caches():
    """Keep memoized cluster discovery from leaking between tests.

    ``manifesto.cli.main`` resets it per command, so tests that call into
    ``workflow`` directly would otherwise inherit whatever a neighbor cached.
    """

    workflow.reset_caches()
    yield
    workflow.reset_caches()
