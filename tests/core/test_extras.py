"""
``requires_extra``: a missing third-party module becomes an ImportError naming the extra to
install; a missing ``axiomatics_ai`` module is a bug and re-raised as it is.
"""
from __future__ import annotations

import importlib
import sys

import pytest

from axiomatics_ai._extras import requires_extra


def test_missing_third_party_module_names_the_extra_and_the_module():
    with pytest.raises(ImportError) as exc_info, requires_extra("strands", "axiomatics_ai.strands"):
        import axiomatics_no_such_dependency  # noqa: F401
    error = exc_info.value
    assert not isinstance(error, ModuleNotFoundError)
    assert "axiomatics_ai.strands" in str(error)
    assert "'strands' extra" in str(error)
    assert "missing module: axiomatics_no_such_dependency" in str(error)
    assert isinstance(error.__cause__, ModuleNotFoundError)
    assert error.__cause__.name == "axiomatics_no_such_dependency"


def test_missing_submodule_names_the_full_module():
    with pytest.raises(ImportError, match=r"missing module: json\.no_such_submodule"), requires_extra("x", "feature"):
        import json.no_such_submodule  # noqa: F401


def test_missing_axiomatics_ai_module_is_re_raised_unchanged():
    with pytest.raises(ModuleNotFoundError) as exc_info, requires_extra("strands", "axiomatics_ai.strands"):
        import axiomatics_ai.no_such_module  # noqa: F401
    assert exc_info.value.name == "axiomatics_ai.no_such_module"
    assert "extra" not in str(exc_info.value)


def test_no_error_passes_through():
    with requires_extra("strands", "axiomatics_ai.strands"):
        import json
    assert json.dumps([]) == "[]"


def test_other_errors_are_not_touched():
    with pytest.raises(ValueError, match="boom"), requires_extra("strands", "axiomatics_ai.strands"):
        raise ValueError("boom")


def test_importing_a_subpackage_without_its_dependency_says_which_extra(monkeypatch):
    saved = {name: mod for name, mod in sys.modules.items()
             if name == "axiomatics_ai.strands" or name.startswith("axiomatics_ai.strands.")}
    package = sys.modules["axiomatics_ai"]
    saved_attr = getattr(package, "strands", None)
    try:
        for name in saved:
            del sys.modules[name]
        # "Not installed": strands and every strands.* submodule already imported.
        for name in [n for n in sys.modules if n == "strands" or n.startswith("strands.")]:
            monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.setitem(sys.modules, "strands", None)
        with pytest.raises(ImportError, match=r"'strands' extra.*missing module: strands"):
            importlib.import_module("axiomatics_ai.strands")
    finally:
        for name in [n for n in sys.modules if n == "axiomatics_ai.strands" or n.startswith("axiomatics_ai.strands.")]:
            del sys.modules[name]
        sys.modules.update(saved)
        if saved_attr is not None:
            package.strands = saved_attr
