"""VER-1: load a version manifest and give it a content-derived id.

The id is the first 12 hex chars of sha256 over canonical JSON of the model,
params, skills, code SHA and the sha256 of each prompt template's bytes. The
manifest's name, its file name and where the templates live don't affect it;
any byte change to a template does.

`load_manifest` stores template paths resolved against the suite dir
(absolute), so runners can read them without knowing the suite dir.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from llm_evals.types import VersionManifest

ID_LENGTH = 12


class ManifestError(ValueError):
    """The manifest file is malformed or refers to a missing template."""


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        raise ManifestError(f"prompt template not found: {path}") from None


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def manifest_hash_input(m: VersionManifest) -> dict[str, Any]:
    """The exact object hashed into the id (useful for debugging id changes)."""
    return {
        "model": m.model,
        "params": m.params,
        "skills": m.skills,
        "code_sha": m.code_sha,
        "templates": {
            role: _sha256_file(Path(path)) for role, path in sorted(m.prompt_templates.items())
        },
    }


def compute_manifest_id(m: VersionManifest) -> str:
    """Template paths must be resolvable from the current directory (absolute after load)."""
    digest = hashlib.sha256(canonical_json(manifest_hash_input(m)).encode("utf-8"))
    return digest.hexdigest()[:ID_LENGTH]


def load_manifest(path: Path, suite_dir: Path) -> VersionManifest:
    """Parse a manifest YAML, resolve template paths against `suite_dir`, fill `id`."""
    path = Path(path)
    suite_dir = Path(suite_dir)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ManifestError(f"manifest not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ManifestError(f"manifest {path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ManifestError(f"manifest {path} must be a YAML mapping")
    if "id" in raw:
        raise ManifestError(f"manifest {path}: 'id' is computed from the content; remove it")
    templates = raw.get("prompt_templates")
    if not isinstance(templates, dict) or not templates:
        raise ManifestError(f"manifest {path}: 'prompt_templates' must map role -> path")

    resolved: dict[str, str] = {}
    for role, rel in templates.items():
        template_path = (suite_dir / str(rel)).resolve()
        if not template_path.is_file():
            raise ManifestError(
                f"manifest {path}: template for role {role!r} not found at {template_path} "
                f"(paths are relative to the suite dir {suite_dir})"
            )
        resolved[str(role)] = str(template_path)

    try:
        manifest = VersionManifest(**{**raw, "prompt_templates": resolved})
    except Exception as exc:  # pydantic ValidationError
        raise ManifestError(f"manifest {path} is invalid: {exc}") from exc
    return manifest.model_copy(update={"id": compute_manifest_id(manifest)})


def manifest_with_overrides(m: VersionManifest, **changes: Any) -> VersionManifest:
    """A copy of `m` with fields changed and the id recomputed."""
    if "id" in changes:
        raise ValueError("id is computed; don't override it")
    updated = VersionManifest(**{**m.model_dump(), **changes})
    return updated.model_copy(update={"id": compute_manifest_id(updated)})
