"""Backward-compatibility alias package for pydantic_ai_durable."""

import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from typing import Any

import pydantic_ai_durable as _canonical
from pydantic_ai_durable import *  # noqa: F403


class _PyaAliasFinder(MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: Any = None,
    ) -> ModuleSpec | None:
        if fullname == "pya_durable" or fullname.startswith("pya_durable."):
            canonical_name = "pydantic_ai_durable" + fullname[len("pya_durable"):]
            __import__(canonical_name)
            canonical_mod = sys.modules[canonical_name]
            sys.modules[fullname] = canonical_mod
            return getattr(canonical_mod, "__spec__", None)
        return None


if not any(isinstance(finder, _PyaAliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _PyaAliasFinder())

sys.modules["pya_durable"] = _canonical

__all__ = _canonical.__all__
