"""Name -> factory registries for runners, checks and renderers.

Suite YAML refers to plug-ins by name. Built-in plug-ins register themselves
when their package is imported; `load_builtins()` imports those packages.
A suite's own plug-ins are listed under `plugins:` in its suite.yaml and loaded
by `load_plugins()`: each entry is an importable module name (`my_evals.checks`)
or a `.py` file relative to the suite folder (`plugins/checks.py`).
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_REGISTRIES: dict[str, dict[str, Callable[..., Any]]] = {
    "runner": {},
    "check": {},
    "pairwise": {},
    "renderer": {},
}

BUILTIN_PACKAGES = (
    "llm_evals.runners",
    "llm_evals.checks",
    "llm_evals.cases.renderers",
)


_PLUGIN_NAMES: set[tuple[str, str]] = set()  # (kind, name) registered by a suite's plug-in
_loading_plugin = False


def _register(kind: str, name: str) -> Callable[[F], F]:
    def decorator(factory: F) -> F:
        registry = _REGISTRIES[kind]
        if name in registry and registry[name] is not factory:
            # A suite's plug-in may replace another suite's plug-in of the same name
            # (the suite being loaded is the one that runs), never a built-in.
            if not (_loading_plugin and (kind, name) in _PLUGIN_NAMES):
                raise ValueError(f"{kind} {name!r} is already registered")
        registry[name] = factory
        if _loading_plugin:
            _PLUGIN_NAMES.add((kind, name))
        return factory

    return decorator


def register_runner(name: str) -> Callable[[F], F]:
    return _register("runner", name)


def register_check(name: str) -> Callable[[F], F]:
    return _register("check", name)


def register_pairwise(name: str) -> Callable[[F], F]:
    return _register("pairwise", name)


def register_renderer(name: str) -> Callable[[F], F]:
    return _register("renderer", name)


def get(kind: str, name: str) -> Callable[..., Any]:
    load_builtins()
    try:
        return _REGISTRIES[kind][name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRIES[kind])) or "none"
        raise KeyError(f"no {kind} named {name!r} (known: {known})") from None


def names(kind: str) -> list[str]:
    load_builtins()
    return sorted(_REGISTRIES[kind])


def load_builtins() -> None:
    for package in BUILTIN_PACKAGES:
        importlib.import_module(package)


_LOADED_PLUGINS: dict[str, Any] = {}  # resolved spec -> module, so a plug-in registers once


def load_plugins(specs: list[str], base_dir: Path) -> None:
    """Import a suite's plug-in modules so their decorators register them.

    A spec ending in `.py` (or containing a slash) is a file, resolved against
    `base_dir`; anything else is a module name, imported with `base_dir` on the
    path so a module that sits next to suite.yaml works without installing it.
    """
    global _loading_plugin
    load_builtins()
    base_dir = Path(base_dir).resolve()
    _loading_plugin = True
    try:
        _load_plugin_specs(specs, base_dir)
    finally:
        _loading_plugin = False


def _load_plugin_specs(specs: list[str], base_dir: Path) -> None:
    for spec in specs:
        is_file = spec.endswith(".py") or "/" in spec
        key = str((base_dir / spec).resolve()) if is_file else spec
        if key in _LOADED_PLUGINS:
            continue
        if is_file:
            path = Path(key)
            if not path.is_file():
                raise FileNotFoundError(f"plug-in file not found: {spec} (looked in {base_dir})")
            name = f"llm_evals_plugin_{path.stem}_{hashlib.sha256(key.encode()).hexdigest()[:8]}"
            module_spec = importlib.util.spec_from_file_location(name, path)
            if module_spec is None or module_spec.loader is None:
                raise ImportError(f"cannot load plug-in file {path}")
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[name] = module
            module_spec.loader.exec_module(module)
        else:
            added = str(base_dir) not in sys.path
            if added:
                sys.path.insert(0, str(base_dir))
            try:
                module = importlib.import_module(spec)
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    f"plug-in module {spec!r} not found: install it, or put it next to suite.yaml "
                    f"({exc})") from exc
            finally:
                if added:
                    sys.path.remove(str(base_dir))
        _LOADED_PLUGINS[key] = module


def load_callable(spec: str, base_dir: Path) -> Callable[..., Any]:
    """`module:function` or `path/to/file.py:function`, resolved like a plug-in."""
    module_spec, sep, attr = spec.rpartition(":")
    if not sep or not module_spec or not attr:
        raise ValueError(f"expected 'module:function' or 'file.py:function', got {spec!r}")
    load_plugins([module_spec], base_dir)
    is_file = module_spec.endswith(".py") or "/" in module_spec
    key = str((Path(base_dir).resolve() / module_spec).resolve()) if is_file else module_spec
    fn = getattr(_LOADED_PLUGINS[key], attr, None)
    if not callable(fn):
        raise AttributeError(f"{module_spec} has no function {attr!r}")
    return fn
