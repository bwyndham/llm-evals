from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from llm_evals.runners.manifest import ManifestError, load_manifest, manifest_with_overrides


def test_id_is_12_hex_and_templates_resolved(version, suite_dir: Path) -> None:
    assert len(version.id) == 12
    int(version.id, 16)
    assert Path(version.prompt_templates["user"]) == (suite_dir / "prompts" / "user.j2").resolve()


def test_id_stable_across_manifest_name_and_location(version, suite_dir: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "other" / "renamed-manifest.yml"
    elsewhere.parent.mkdir()
    text = (suite_dir / "versions" / "v1.yaml").read_text().replace("name: v1", "name: other")
    elsewhere.write_text(text)
    assert load_manifest(elsewhere, suite_dir).id == version.id


def test_id_stable_when_templates_move(version, suite_dir: Path, tmp_path: Path) -> None:
    moved = tmp_path / "moved"
    shutil.copytree(suite_dir, moved)
    assert load_manifest(moved / "versions" / "v1.yaml", moved).id == version.id


def test_template_byte_change_changes_id(version, suite_dir: Path) -> None:
    (suite_dir / "prompts" / "user.j2").write_text("Ask: {{ ask }} \n")  # one extra space
    assert load_manifest(suite_dir / "versions" / "v1.yaml", suite_dir).id != version.id


def test_param_change_changes_id(version) -> None:
    changed = manifest_with_overrides(version, params={"max_tokens": 4000, "effort": "high"})
    assert changed.id != version.id
    same = manifest_with_overrides(version, name="renamed")
    assert same.id == version.id


def test_missing_template_errors(suite_dir: Path) -> None:
    (suite_dir / "prompts" / "system.j2").unlink()
    with pytest.raises(ManifestError, match="system"):
        load_manifest(suite_dir / "versions" / "v1.yaml", suite_dir)


def test_id_in_yaml_rejected(suite_dir: Path) -> None:
    p = suite_dir / "versions" / "v1.yaml"
    p.write_text(p.read_text() + "id: abc\n")
    with pytest.raises(ManifestError, match="computed"):
        load_manifest(p, suite_dir)
