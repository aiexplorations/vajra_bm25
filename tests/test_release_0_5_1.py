"""
Pre-release checks for vajra-bm25 0.5.1 (deprecation release).

0.5.1 adds a DeprecationWarning on import that points users to vajra-search.
These tests check the version metadata, the warning, and that the public API
still works. They fail until the version bump and the warning text are final.

Run: pytest tests/test_release_0_5_1.py -v
"""

import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

import vajra_bm25

TARGET_VERSION = "0.5.1rc1"
ROOT = Path(__file__).resolve().parent.parent


def _import_in_fresh_interpreter(*flags: str) -> subprocess.CompletedProcess[str]:
    """Import vajra_bm25 in a new process so the import-time warning fires."""
    return subprocess.run(
        [sys.executable, *flags, "-c", "import vajra_bm25"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )


# ============================================================================
# Version metadata
# ============================================================================


class TestVersion:
    def test_pyproject_version(self):
        data = tomllib.loads((ROOT / "pyproject.toml").read_text())
        assert data["project"]["version"] == TARGET_VERSION

    def test_package_version_matches_pyproject(self):
        data = tomllib.loads((ROOT / "pyproject.toml").read_text())
        assert vajra_bm25.__version__ == data["project"]["version"]

    def test_warning_text_names_current_version(self):
        """The warning text must not quote an older version than __version__."""
        proc = _import_in_fresh_interpreter("-W", "always::DeprecationWarning")
        assert f"v{vajra_bm25.__version__}" in proc.stderr


# ============================================================================
# Deprecation warning
# ============================================================================


class TestDeprecationWarning:
    def test_import_emits_deprecation_warning(self):
        proc = _import_in_fresh_interpreter("-W", "always::DeprecationWarning")
        assert proc.returncode == 0, proc.stderr
        assert "DeprecationWarning" in proc.stderr

    def test_warning_points_to_vajra_search(self):
        proc = _import_in_fresh_interpreter("-W", "always::DeprecationWarning")
        assert "vajra-search" in proc.stderr

    def test_warning_is_attributed_to_caller(self):
        """stacklevel must blame the importing code, not importlib internals."""
        proc = _import_in_fresh_interpreter("-W", "always::DeprecationWarning")
        assert "importlib" not in proc.stderr

    def test_warning_is_not_fatal_under_error_filter_for_other_categories(self):
        proc = _import_in_fresh_interpreter("-W", "error::UserWarning")
        assert proc.returncode == 0, proc.stderr

    def test_warning_can_be_silenced(self):
        proc = _import_in_fresh_interpreter("-W", "ignore::DeprecationWarning")
        assert proc.returncode == 0
        assert "DeprecationWarning" not in proc.stderr


# ============================================================================
# Public API still works
# ============================================================================


class TestPublicApi:
    def test_all_exports_resolve(self):
        missing = [n for n in vajra_bm25.__all__ if not hasattr(vajra_bm25, n)]
        assert missing == []

    def test_bm25_search_smoke(self):
        corpus = vajra_bm25.create_sample_corpus()
        engine = vajra_bm25.VajraSearchOptimized(corpus)
        results = engine.search("category functors", top_k=3)
        assert 0 < len(results) <= 3
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_hnsw_search_smoke(self):
        from vajra_bm25.vector.hnsw import NativeHNSWIndex

        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((100, 64)).astype(np.float32)
        ids = [f"doc_{i}" for i in range(100)]
        index = NativeHNSWIndex(dimension=64, metric="cosine", M=8, ef_construction=50)
        index.add(ids, vectors)
        results = index.search(vectors[0], k=5)
        assert len(results) == 5
        assert results[0].id == "doc_0"


# ============================================================================
# Build artifacts
# ============================================================================


class TestBuild:
    @pytest.mark.skipif(
        importlib.util.find_spec("build") is None
        or importlib.util.find_spec("twine") is None,
        reason="pip install build twine",
    )
    def test_sdist_and_wheel_build_and_pass_twine_check(self, tmp_path):
        build = subprocess.run(
            [sys.executable, "-m", "build", "--outdir", str(tmp_path)],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )
        assert build.returncode == 0, build.stderr
        artifacts = sorted(p.name for p in tmp_path.iterdir())
        assert any(a.endswith(".whl") and TARGET_VERSION in a for a in artifacts)
        assert any(a.endswith(".tar.gz") and TARGET_VERSION in a for a in artifacts)

        twine = subprocess.run(
            [sys.executable, "-m", "twine", "check", *map(str, tmp_path.iterdir())],
            capture_output=True,
            text=True,
            check=False,
        )
        assert twine.returncode == 0, twine.stdout + twine.stderr
