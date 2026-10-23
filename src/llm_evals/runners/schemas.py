"""Output schemas the prompt runner validates parsed JSON against.

Two kinds, picked by the suite's `runner.output_schema`:
- a registered name (`spec_ticket`): a pydantic model in code, registered with
  `@register_schema` (built in, or from a suite plug-in);
- a path ending in .json, .yaml or .yml, relative to the suite folder: a JSON
  Schema document. No Python needed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

_SCHEMAS: dict[str, type[BaseModel]] = {}


def register_schema(name: str):
    def decorator(model: type[BaseModel]) -> type[BaseModel]:
        if name in _SCHEMAS and _SCHEMAS[name] is not model:
            raise ValueError(f"schema {name!r} is already registered")
        _SCHEMAS[name] = model
        return model

    return decorator


def get_schema(name: str) -> type[BaseModel]:
    try:
        return _SCHEMAS[name]
    except KeyError:
        known = ", ".join(sorted(_SCHEMAS)) or "none"
        raise KeyError(f"no output schema named {name!r} (known: {known})") from None


def schema_names() -> list[str]:
    return sorted(_SCHEMAS)


SCHEMA_FILE_SUFFIXES = (".json", ".yaml", ".yml")


def is_schema_file(spec: str) -> bool:
    return spec.lower().endswith(SCHEMA_FILE_SUFFIXES)


@lru_cache(maxsize=64)
def _file_validator(path: str, mtime_ns: int) -> Any:  # mtime busts the cache
    import jsonschema

    p = Path(path)
    text = p.read_text(encoding="utf-8")
    document = json.loads(text) if p.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(document, dict):
        raise ValueError(f"output schema {path} must be a JSON Schema object")
    cls = jsonschema.validators.validator_for(document)
    try:
        cls.check_schema(document)
    except jsonschema.SchemaError as exc:
        raise ValueError(f"output schema {path} is not a valid JSON Schema: {exc.message}") from exc
    return cls(document)


def validator_for(spec: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """A function that validates parsed output against `spec` and returns it,
    raising ValueError with a readable message. Raises KeyError or
    FileNotFoundError for an unknown name or a missing file (config problems)."""
    if is_schema_file(spec):
        p = Path(spec)
        if not p.is_file():
            raise FileNotFoundError(f"output schema file not found: {spec}")

        def check_file(data: dict[str, Any]) -> dict[str, Any]:
            validator = _file_validator(str(p.resolve()), p.stat().st_mtime_ns)
            errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))
            if errors:
                where = lambda e: "/".join(str(x) for x in e.absolute_path) or "(root)"  # noqa: E731
                detail = "; ".join(f"{where(e)}: {e.message}" for e in errors[:3])
                more = f" (+{len(errors) - 3} more)" if len(errors) > 3 else ""
                raise ValueError(f"output does not match schema {p.name}: {detail}{more}")
            return data

        _file_validator(str(p.resolve()), p.stat().st_mtime_ns)  # fail fast on a broken schema
        return check_file

    model = get_schema(spec)

    def check_model(data: dict[str, Any]) -> dict[str, Any]:
        try:
            return model.model_validate(data).model_dump()
        except ValidationError as exc:
            raise ValueError(f"output does not match schema {spec!r}: {exc}") from exc

    return check_model


@register_schema("spec_ticket")
class SpecTicket(BaseModel):
    """The spec writer's output (docs/contracts.md). Extra keys are rejected.

    The 80-char title limit is left to a check, so a long title is a scored
    failure rather than a SYSTEM error.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    title: str
    description: str
    acceptance_criteria: str
    assumptions: list[str]
    open_questions: list[str]
