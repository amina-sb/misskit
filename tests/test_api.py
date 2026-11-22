from __future__ import annotations

import misskit


def test_public_names_resolve() -> None:
    for name in misskit.__all__:
        assert hasattr(misskit, name), name


def test_version_matches_metadata() -> None:
    from importlib.metadata import version

    assert misskit.__version__ == version("misskit")
