"""Smoke tests — the package + its submodules import without side effects."""

import importlib


def test_package_imports():
    import evalscope_ext

    assert hasattr(evalscope_ext, "__version__")
    assert evalscope_ext.UPSTREAM_PINNED_SHA.startswith("e9d42d8")


def test_submodules_import():
    for sub in [
        "evalscope_ext.calibration",
        "evalscope_ext.pruners",
        "evalscope_ext.adapters",
        "evalscope_ext.tools",
    ]:
        importlib.import_module(sub)


def test_pinned_sha_is_full_length():
    import evalscope_ext

    # Full 40-character SHA so a reader can `git checkout` it directly.
    assert len(evalscope_ext.UPSTREAM_PINNED_SHA) == 40
