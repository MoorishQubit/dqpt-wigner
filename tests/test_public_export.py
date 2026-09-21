"""Check the publication boundary and integrity checks using small repositories."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


integrity = _load("verify_reproduction_data")
export = _load("export_public_repository")


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    for name, content in {
        "manuscript/main.tex": "private document",
        "results/raw.csv": "x,y\n1,2\n",
        "results/editorial/main.txt": "private extracted text",
        "configs/smoke.json": "{}\n",
        "README.md": "Public project\n",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    # Apply exclusions after adding private files, as in an existing checkout.
    (root / ".gitignore").write_text("/manuscript/\n/results/editorial/\n")
    (root / "results/new.csv").write_text("x,y\n3,4\n")
    return root


def test_export_excludes_ignored_tracked_documents_and_keeps_untracked_data(checkout, tmp_path):
    output = tmp_path / "public"
    manifest = checkout / integrity.MANIFEST
    assert integrity.write_manifest(checkout, manifest) == 3
    export.export_repository(checkout, output)
    assert not (output / "manuscript").exists()
    assert not (output / "results/editorial").exists()
    assert not (output / ".git").exists()
    assert (output / "results/new.csv").read_text() == "x,y\n3,4\n"
    assert (checkout / "manuscript/main.tex").read_text() == "private document"
    assert integrity.verify_manifest(output, output / integrity.MANIFEST) == (3, [])
    first = manifest.read_bytes()
    integrity.write_manifest(checkout, manifest)
    assert manifest.read_bytes() == first


def test_manifest_detects_missing_and_changed_data_without_git(checkout, tmp_path):
    integrity.write_manifest(checkout, checkout / integrity.MANIFEST)
    output = tmp_path / "public"
    export.export_repository(checkout, output)
    (output / "results/raw.csv").write_text("changed")
    (output / "configs/smoke.json").unlink()
    count, failures = integrity.verify_manifest(output, output / integrity.MANIFEST)
    assert count == 3
    assert set(failures) == {"CHANGED results/raw.csv", "MISSING configs/smoke.json"}


@pytest.mark.parametrize("name", ["../private", "/tmp/private", "results/../../private"])
def test_manifest_rejects_escaping_paths(tmp_path, name):
    manifest = tmp_path / "manifest"
    manifest.write_text("0" * 64 + "  " + name + "\n")
    with pytest.raises(ValueError, match="Invalid scientific data path"):
        integrity.verify_manifest(tmp_path, manifest)


def test_export_refuses_nonempty_or_in_tree_destination(checkout, tmp_path):
    with pytest.raises(ValueError, match="outside"):
        export.export_repository(checkout, checkout / "export")
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("preserve")
    with pytest.raises(ValueError, match="empty"):
        export.export_repository(checkout, output)
    assert (output / "keep.txt").read_text() == "preserve"


def test_export_refuses_symlink_to_private_document(checkout, tmp_path):
    (checkout / "results/leak.txt").symlink_to(checkout / "manuscript/main.tex")
    with pytest.raises(ValueError, match="symlinks"):
        export.export_repository(checkout, tmp_path / "public")
    assert not (tmp_path / "public").exists()


def test_export_refuses_tracked_file_through_symlink_directory(checkout, tmp_path):
    # A tracked directory can be replaced by a symlink after it enters the index.
    (checkout / "configs/smoke.json").unlink()
    (checkout / "configs").rmdir()
    (checkout / "manuscript/smoke.json").write_text("private replacement")
    (checkout / "configs").symlink_to(checkout / "manuscript", target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks|symbolic link"):
        export.export_repository(checkout, tmp_path / "public")


def test_export_requires_git_checkout(tmp_path):
    root = tmp_path / "plain"
    root.mkdir()
    with pytest.raises(ValueError, match="Git checkout"):
        export.export_repository(root, tmp_path / "public")
