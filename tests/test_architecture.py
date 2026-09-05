"""Static checks on module boundaries.

Imports are read with `ast`, never by importing the modules, so these tests
stay fast and keep working while a subpackage is still half-written.

The boundaries exist for a concrete reason. `micm.mapping` and `micm.env` read
cached posterior files and nothing else, which is what makes designing a new
mapping cheap: it never re-runs a decoder. Boundary erosion happens one
convenient import at a time, so it is checked mechanically.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "micm"

# Physics engines are banned by design: the environment is a few hundred lines
# of NumPy and has to run roughly 10,000 episodes.
PHYSICS_ENGINES = ("pybullet", "mujoco", "dm_control", "gymnasium", "gym")


def _module_name(path: Path) -> str:
    """Dotted name of a source file, e.g. src/micm/env/task.py -> micm.env.task.

    A file outside `src/` has no package context, so it falls back to its stem.
    That only affects relative-import resolution, which cannot occur outside the
    package anyway.
    """
    try:
        relative = path.relative_to(SRC_ROOT)
    except ValueError:
        return path.stem
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _absolute_target(module: str, node: ast.ImportFrom) -> str:
    """Resolve a possibly-relative `from ... import ...` to a dotted module name."""
    if node.level == 0:
        return node.module or ""

    # A module `micm.a.b` sits in package `micm.a`; level 1 means that package,
    # level 2 its parent, and so on.
    package_parts = module.split(".")[:-1]
    ascend = node.level - 1
    if ascend:
        package_parts = package_parts[:-ascend] if ascend < len(package_parts) else []
    base = ".".join(package_parts)
    return f"{base}.{node.module}" if node.module else base


def _with_prefixes(name: str) -> set[str]:
    """Expand `a.b.c` to {a, a.b, a.b.c} so a package-level check catches submodules."""
    parts = name.split(".")
    return {".".join(parts[: i + 1]) for i in range(len(parts))}


def imports_of(target: str | Path) -> set[str]:
    """Every module imported anywhere under `target`, expanded to include prefixes.

    Args:
        target: path relative to the repository root; a file or a directory.

    Returns:
        Dotted module names. Because prefixes are included, `"micm.decoders" in
        imports_of(...)` is true for `from micm.decoders.base import Decoder`.

    Raises:
        FileNotFoundError: if `target` does not exist. A missing path must not
            quietly return an empty set, which would make the check pass by
            accident.
    """
    path = REPO_ROOT / target
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    files = sorted(path.rglob("*.py")) if path.is_dir() else [path]

    found: set[str] = set()
    for file in files:
        module = _module_name(file)
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found |= _with_prefixes(alias.name)
            elif isinstance(node, ast.ImportFrom):
                resolved = _absolute_target(module, node)
                if resolved:
                    found |= _with_prefixes(resolved)
    return found


# --- the parser itself, so an empty tree cannot make every check below pass ---


def test_imports_of_finds_plain_and_from_imports(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text(
        "import numpy as np\nfrom micm.decoders.base import Decoder\n", encoding="utf-8"
    )
    found = imports_of(source)
    assert "numpy" in found
    assert {"micm", "micm.decoders", "micm.decoders.base"} <= found


def test_imports_of_resolves_relative_imports() -> None:
    """`from . import x` inside micm.utils must resolve to micm.utils.x."""
    node = ast.ImportFrom(module="hashing", names=[], level=1)
    assert _absolute_target("micm.utils.config", node) == "micm.utils.hashing"

    parent = ast.ImportFrom(module="env", names=[], level=2)
    assert _absolute_target("micm.eval.runner", parent) == "micm.env"


def test_imports_of_raises_on_missing_path() -> None:
    with pytest.raises(FileNotFoundError):
        imports_of("src/micm/does_not_exist")


def test_source_tree_is_where_the_checks_expect_it() -> None:
    """Guards the checks below against silently passing on a moved tree."""
    for subpackage in ("data", "decoders", "replay", "mapping", "env", "eval", "utils", "viz"):
        assert (PACKAGE_ROOT / subpackage / "__init__.py").is_file(), subpackage


# --- the boundaries themselves ---


@pytest.mark.parametrize("subpackage", ["mapping", "env"])
@pytest.mark.parametrize("forbidden", ["micm.decoders", "micm.data"])
def test_mapping_and_env_do_not_import_the_decoder_side(subpackage: str, forbidden: str) -> None:
    assert forbidden not in imports_of(f"src/micm/{subpackage}")


def test_eval_does_not_import_render() -> None:
    """Rendering costs about 100x the simulation, so the runner must not reach it.

    Checked over the whole `eval` subpackage rather than `runner.py` alone: no
    module on the evaluation path has a reason to render, and the wider check
    starts holding before `runner.py` exists.
    """
    assert "micm.env.render" not in imports_of("src/micm/eval")


def test_utils_is_a_leaf() -> None:
    """`micm.utils` imports nothing from other micm subpackages, so it cannot cycle."""
    external = {
        name
        for name in imports_of("src/micm/utils")
        if name.startswith("micm.") and not name.startswith("micm.utils")
    }
    assert external == set()


def test_no_physics_engine_anywhere() -> None:
    found = imports_of("src/micm")
    assert found.isdisjoint(PHYSICS_ENGINES)


def test_viz_is_not_imported_by_the_pipeline() -> None:
    """Figures read run directories; nothing upstream of them depends on plotting."""
    for subpackage in ("data", "decoders", "replay", "mapping", "env", "eval"):
        assert "micm.viz" not in imports_of(f"src/micm/{subpackage}")


# --- version control ---

# Paths that belong in the repository. A gitignore pattern swallowing any of
# these loses work silently: the files exist locally, every test passes, and
# they are simply absent from the next clone.
MUST_BE_TRACKED = (
    "src/micm",
    "tests",
    "scripts",
    "configs",
    "pyproject.toml",
    "Makefile",
    ".python-version",
)

# Deliberately excluded, checked so a future edit cannot start committing a
# multi-gigabyte dataset or the local working documents.
MUST_BE_IGNORED = (
    "data/raw/anything",
    "artifacts/posteriors/anything.npz",
    ".venv/pyvenv.cfg",
    ".agents/00-overview.md",
    "CLAUDE.md",
    "docs/decisions.md",
)


def _ignored(paths: list[str]) -> set[str]:
    """Which of `paths` git would ignore, asked in one call.

    NUL-separated in both directions. In its normal mode git quotes and escapes
    paths, and a Windows checkout adds carriage returns, both of which would make
    the comparisons below fail for reasons unrelated to gitignore.
    """
    result = subprocess.run(
        ["git", "check-ignore", "--stdin", "-z"],
        input="\0".join(paths).encode("utf-8"),
        capture_output=True,
        cwd=REPO_ROOT,
        check=False,
    )
    return {
        entry.decode("utf-8").replace("\\", "/") for entry in result.stdout.split(b"\0") if entry
    }


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
@pytest.mark.skipif(not (REPO_ROOT / ".git").exists(), reason="not a git working tree")
def test_no_source_file_is_gitignored() -> None:
    """The stock Python gitignore excludes `env/`, `ENV/`, `lib/`, `build/` and more.

    None of those are anchored, so each matches a directory of that name at any
    depth. `env/` and `ENV/` were excluding `src/micm/env/` and `configs/env/`,
    and an unanchored `data/` was excluding `src/micm/data/` and `configs/data/`,
    which is the entire data pipeline. Nothing about that raises: the code runs,
    the tests pass, and the files are missing from the next clone.
    """
    candidates: list[str] = []
    for entry in MUST_BE_TRACKED:
        path = REPO_ROOT / entry
        if path.is_dir():
            candidates.extend(
                str(file.relative_to(REPO_ROOT)).replace("\\", "/")
                for file in path.rglob("*")
                if file.is_file() and "__pycache__" not in file.parts
            )
        elif path.is_file():
            candidates.append(entry)

    assert candidates, "found no source files to check"
    assert _ignored(candidates) == set()


# Directories that do not exist yet but would be ordinary places to put source.
# Every name here appears in the stock Python gitignore as a build or tool
# artefact, so an unanchored pattern would swallow it the day it is created.
COULD_BE_ADDED = (
    "src/micm/lib/thing.py",
    "src/micm/var/thing.py",
    "src/micm/parts/thing.py",
    "configs/target/thing.yaml",
    "configs/instance/thing.yaml",
    "configs/share/thing.yaml",
    "tests/cover/test_thing.py",
    "tests/fixtures/golden_subj01.npz",
    "scripts/downloads/thing.py",
    "notebooks/01_thing.ipynb",
    "paper/main.tex",
)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
@pytest.mark.skipif(not (REPO_ROOT / ".git").exists(), reason="not a git working tree")
def test_a_source_directory_added_later_would_not_be_swallowed() -> None:
    """`test_no_source_file_is_gitignored` only sees files that exist today.

    That leaves the same failure available to anyone who adds a directory the
    stock gitignore happens to name. Creating `src/micm/lib/` would have removed
    it from version control with no error and no failing test, which is how
    `src/micm/data/` was lost once already. These paths are hypothetical on
    purpose: the point is to fail before the directory exists.
    """
    assert _ignored(list(COULD_BE_ADDED)) == set()


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
@pytest.mark.skipif(not (REPO_ROOT / ".git").exists(), reason="not a git working tree")
def test_large_and_local_paths_stay_ignored() -> None:
    """The other direction: anchoring the patterns must not have un-ignored the data."""
    assert _ignored(list(MUST_BE_IGNORED)) == {p.replace("\\", "/") for p in MUST_BE_IGNORED}
