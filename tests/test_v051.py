"""
Tests for the 0.5.1 fixes reported by the Verity backend team (benchmark of 17 Nov 2026).

1. add_document / remove_document must rebuild every scorer
2. index build tokenises in-process by default (no multiprocessing.Pool)
3. analyzer hook and tag_aware_analyzer
4. library logging propagates to the application
5. __version__ comes from package metadata
"""

import logging
import multiprocessing
from importlib import metadata

import pytest

import vajra_bm25
from vajra_bm25 import Document, DocumentCorpus, VajraSearchOptimized
from vajra_bm25 import optimized
from vajra_bm25.optimized import (
    VectorizedIndex,
    VectorizedIndexSparse,
    NUMBA_AVAILABLE,
    SCIPY_AVAILABLE,
)
from vajra_bm25.text_processing import preprocess_text, tag_aware_analyzer

pytestmark = pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy required")


def make_corpus(n=6):
    docs = [
        Document(id=f"item{i}", title=f"Item {i}", content=f"item{i} alpha beta common filler text {i}")
        for i in range(n)
    ]
    return DocumentCorpus(docs)


def gamma_doc():
    return Document(id="new", title="New", content="gamma gamma delta")


# Scorer configurations covering every search path of the sparse index
SPARSE_CONFIGS = [
    pytest.param(dict(use_eager=True, use_numba=False), id="eager"),
    pytest.param(dict(use_eager=False, use_numba=True), id="numba",
                 marks=pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba required")),
    pytest.param(dict(use_eager=False, use_numba=False, use_maxscore=True), id="maxscore"),
    pytest.param(dict(use_eager=False, use_numba=False, use_maxscore=False), id="traditional"),
]


def ids(results):
    return [r.document.id for r in results]


# --------------------------------------------------------------------------- #
# Fix 1: stale scorers
# --------------------------------------------------------------------------- #
class TestAddRemoveRebuildsScorers:
    @pytest.mark.parametrize("config", SPARSE_CONFIGS)
    def test_added_document_found_sparse(self, config):
        engine = VajraSearchOptimized(make_corpus(), use_sparse=True, cache_size=0, **config)
        assert ids(engine.search("gamma")) == []

        engine.add_document(gamma_doc())

        assert ids(engine.search("gamma")) == ["new"]
        # old documents still found
        assert "item3" in ids(engine.search("item3"))

    def test_added_document_found_dense(self):
        engine = VajraSearchOptimized(make_corpus(), use_sparse=False, cache_size=0)
        engine.add_document(gamma_doc())
        assert ids(engine.search("gamma")) == ["new"]

    @pytest.mark.parametrize("config", SPARSE_CONFIGS)
    def test_removed_document_gone_sparse(self, config):
        engine = VajraSearchOptimized(make_corpus(), use_sparse=True, cache_size=0, **config)
        assert "item2" in ids(engine.search("item2"))

        engine.remove_document("item2")

        assert "item2" not in ids(engine.search("item2"))
        assert "item3" in ids(engine.search("item3"))

    def test_every_scorer_references_the_rebuilt_index(self):
        engine = VajraSearchOptimized(make_corpus(), use_sparse=True, use_maxscore=True,
                                      use_numba=NUMBA_AVAILABLE, use_eager=True)
        engine.add_document(gamma_doc())
        for scorer in (engine.scorer, engine.maxscore_scorer, engine.numba_scorer, engine.eager_scorer):
            if scorer is not None:
                assert scorer.index is engine.index
        assert engine.index.num_docs == 7
        assert engine.index.score_matrix.shape[1] == 7

    def test_scorer_parameters_survive_rebuild(self):
        engine = VajraSearchOptimized(make_corpus(), use_sparse=True, k1=1.2, b=0.5)
        engine.add_document(gamma_doc())
        assert (engine.scorer.k1, engine.scorer.b) == (1.2, 0.5)
        assert (engine.index._eager_k1, engine.index._eager_b) == (1.2, 0.5)

    def test_query_cache_cleared_on_add(self):
        engine = VajraSearchOptimized(make_corpus(), use_sparse=True, cache_size=10)
        assert ids(engine.search("gamma")) == []
        engine.add_document(gamma_doc())
        assert ids(engine.search("gamma")) == ["new"]

    def test_add_documents_rebuilds_once(self, monkeypatch):
        engine = VajraSearchOptimized(make_corpus(), use_sparse=True, cache_size=0)
        calls = []
        original = VajraSearchOptimized._build_index
        monkeypatch.setattr(VajraSearchOptimized, "_build_index",
                            lambda self: (calls.append(1), original(self))[1])

        engine.add_documents([
            Document(id="n1", title="", content="gamma one"),
            Document(id="n2", title="", content="gamma two"),
            Document(id="n3", title="", content="gamma three"),
        ])

        assert len(calls) == 1
        assert set(ids(engine.search("gamma"))) == {"n1", "n2", "n3"}

    def test_batch_add_documents_still_works(self):
        engine = VajraSearchOptimized(make_corpus(), use_sparse=True, cache_size=0)
        engine.batch_add_documents([gamma_doc()])
        assert ids(engine.search("gamma")) == ["new"]

    def test_duplicate_ids_rejected(self):
        engine = VajraSearchOptimized(make_corpus(), cache_size=0)
        with pytest.raises(ValueError):
            engine.add_document(Document(id="item1", title="", content="x"))
        with pytest.raises(ValueError):
            engine.add_documents([Document(id="a", title="", content="x"),
                                  Document(id="a", title="", content="y")])
        assert len(engine.corpus) == 6

    def test_remove_unknown_document_raises(self):
        engine = VajraSearchOptimized(make_corpus(), cache_size=0)
        with pytest.raises(ValueError):
            engine.remove_document("nope")

    def test_callers_corpus_not_mutated(self):
        corpus = make_corpus()
        engine = VajraSearchOptimized(corpus, cache_size=0)
        engine.add_document(gamma_doc())
        assert len(corpus) == 6

    def test_vector_engine_gets_added_documents(self):
        class FakeVector:
            def __init__(self):
                self.added = []

            def add_document(self, doc):
                self.added.append(doc.id)

        engine = VajraSearchOptimized(make_corpus(), cache_size=0)
        engine._vector_engine = FakeVector()
        engine._vector_initialized = True

        engine.add_document(gamma_doc())

        assert engine._vector_engine.added == ["new"]

    def test_vector_engine_reset_on_remove(self):
        engine = VajraSearchOptimized(make_corpus(), cache_size=0)
        engine._vector_engine = object()
        engine._hybrid_engine = object()
        engine._vector_initialized = True

        engine.remove_document("item1")

        assert engine._vector_engine is None and engine._hybrid_engine is None
        assert engine._vector_initialized is False


# --------------------------------------------------------------------------- #
# Fix 2: no Pool by default
# --------------------------------------------------------------------------- #
class TestBuildWithoutPool:
    def test_default_build_uses_no_pool(self, monkeypatch):
        def forbidden(*args, **kwargs):
            raise AssertionError("multiprocessing context requested")

        monkeypatch.setattr(optimized.multiprocessing, "get_context", forbidden)
        monkeypatch.setattr(optimized.multiprocessing, "Pool", forbidden)

        index = VectorizedIndexSparse()
        index.build(make_corpus(629))

        assert index.num_docs == 629
        assert multiprocessing.active_children() == []

    def test_engine_default_n_jobs_is_one(self):
        engine = VajraSearchOptimized(make_corpus(), use_sparse=True)
        assert engine.n_jobs == 1

    def test_pool_not_used_below_threshold(self, monkeypatch):
        def forbidden(*args, **kwargs):
            raise AssertionError("pool used for a small corpus")

        monkeypatch.setattr(optimized.multiprocessing, "get_context", forbidden)
        VectorizedIndexSparse().build(make_corpus(50), n_jobs=4)

    def test_pool_path_matches_in_process_index(self, monkeypatch):
        monkeypatch.setattr(optimized, "POOL_MIN_DOCS", 10)
        corpus = make_corpus(40)

        serial = VectorizedIndexSparse()
        serial.build(corpus, n_jobs=1)
        pooled = VectorizedIndexSparse()
        pooled.build(corpus, n_jobs=2)

        assert pooled.term_to_id == serial.term_to_id
        assert pooled.doc_to_id == serial.doc_to_id
        assert (pooled.term_doc_matrix != serial.term_doc_matrix).nnz == 0
        assert (pooled.doc_lengths == serial.doc_lengths).all()

    def test_n_jobs_minus_one_means_all_cores(self):
        assert optimized._resolve_n_jobs(-1) == multiprocessing.cpu_count()
        assert optimized._resolve_n_jobs(None) == 1
        assert optimized._resolve_n_jobs(3) == 3


# --------------------------------------------------------------------------- #
# Fix 3: analyzer hook
# --------------------------------------------------------------------------- #
PRESSURE_DOCS = [
    Document(id="p1", title="Pressure notes",
             content="Design pressure pressure pressure pressure of the system is reviewed. "
                     "Pressure drop pressure rating pressure test."),
    Document(id="p2", title="Pressure basis",
             content="Pressure vessel pressure limits and pressure relief pressure margins."),
    Document(id="v110", title="Vessel V-110",
             content="The design pressure of V-110 is 10 barg."),
    Document(id="v120", title="Vessel V-120",
             content="The design pressure of V-120 is 3 barg."),
    Document(id="pump", title="Pumps",
             content="The P-400A/B pumps deliver produced water at 15 barg pressure."),
]


TAG_CONFUSION_DOCS = [
    Document(id="target", title="Separator", content="Vessel V-110 is a test separator rated 10 barg."),
    Document(id="decoy1", title="Vessel list",
             content="Vessel V-120 runs at 110 degC. Vessel V-130 and vessel V-140 and vessel V-150 "
                     "are similar vessels, each vessel with vessel nozzles."),
    Document(id="decoy2", title="Vessel notes",
             content="Vessel V-160 vessel V-170 vessel V-180 vessel V-190 listed with 110 mm nozzles "
                     "and vessel supports."),
    Document(id="other", title="Pumps", content="Pump P-200 delivers water at 15 barg."),
    Document(id="other2", title="Compressors", content="Compressor K-300 has 2 stages."),
]


class TestAnalyzer:
    def test_default_analyzer_is_preprocess_text(self):
        engine = VajraSearchOptimized(make_corpus(), cache_size=0)
        assert engine.analyzer is preprocess_text

    def test_default_results_unchanged(self):
        corpus = DocumentCorpus(PRESSURE_DOCS)
        default = VajraSearchOptimized(corpus, cache_size=0)
        explicit = VajraSearchOptimized(corpus, cache_size=0, analyzer=preprocess_text)
        for q in ["design pressure of V-110", "P-400A/B pumps", "pressure"]:
            assert [(r.document.id, r.score) for r in default.search(q)] == \
                   [(r.document.id, r.score) for r in explicit.search(q)]

    @pytest.mark.parametrize("use_sparse", [False, True])
    def test_tag_aware_analyzer_ranks_the_tag(self, use_sparse):
        corpus = DocumentCorpus(TAG_CONFUSION_DOCS)
        plain = VajraSearchOptimized(corpus, cache_size=0, use_sparse=use_sparse)
        tagged = VajraSearchOptimized(corpus, cache_size=0, use_sparse=use_sparse,
                                      analyzer=tag_aware_analyzer)

        # The decoys contain "v" and "110" separately, so plain tokens cannot tell them
        # from the passage that names V-110; the joined tag v110 is a rare term.
        assert ids(plain.search("What is V-110?"))[0] != "target"
        assert ids(tagged.search("What is V-110?"))[0] == "target"

    def test_tag_aware_analyzer_keeps_ordinary_queries(self):
        corpus = DocumentCorpus(PRESSURE_DOCS)
        tagged = VajraSearchOptimized(corpus, cache_size=0, analyzer=tag_aware_analyzer)
        assert ids(tagged.search("design pressure of V-110"))[0] == "v110"

    def test_tag_aware_analyzer_finds_slash_tags(self):
        engine = VajraSearchOptimized(DocumentCorpus(PRESSURE_DOCS), cache_size=0,
                                      analyzer=tag_aware_analyzer)
        assert ids(engine.search("P-400B"))[0] == "pump"
        assert ids(engine.search("p400a"))[0] == "pump"

    def test_custom_analyzer_used_for_documents_and_queries(self):
        seen = []

        def shout(text):
            seen.append(text)
            return [w.upper() for w in preprocess_text(text)]

        engine = VajraSearchOptimized(make_corpus(), cache_size=0, analyzer=shout)
        before = len(seen)
        engine.search("alpha")

        assert len(seen) > before  # the query went through the analyzer
        assert "ALPHA" in engine.index.term_to_id
        assert "alpha" not in engine.index.term_to_id
        assert len(ids(engine.search("alpha"))) == 6

    def test_analyzer_survives_add_document(self):
        engine = VajraSearchOptimized(DocumentCorpus(PRESSURE_DOCS), cache_size=0,
                                      analyzer=tag_aware_analyzer)
        engine.add_document(Document(id="new", title="Vessel V-130", content="The vessel V-130 is new."))
        assert ids(engine.search("V-130"))[0] == "new"

    def test_analyzer_round_trips_through_save_and_load(self, tmp_path):
        pytest.importorskip("joblib")
        corpus = DocumentCorpus(PRESSURE_DOCS)
        engine = VajraSearchOptimized(corpus, cache_size=0, analyzer=tag_aware_analyzer)
        path = tmp_path / "index.joblib"
        engine.save_index(path)

        loaded = VajraSearchOptimized.load_index(path, corpus)

        assert loaded.analyzer is tag_aware_analyzer
        assert ids(loaded.search("V-110"))[0] == "v110"

    def test_unpicklable_analyzer_fails_clearly_on_save(self, tmp_path):
        pytest.importorskip("joblib")
        engine = VajraSearchOptimized(make_corpus(), cache_size=0,
                                      analyzer=lambda text: preprocess_text(text))
        path = tmp_path / "index.joblib"
        with pytest.raises(ValueError, match="analyzer"):
            engine.save_index(path)
        assert not path.exists()

    def test_old_index_file_loads_with_default_analyzer(self, tmp_path):
        pytest.importorskip("joblib")
        import joblib
        corpus = make_corpus()
        engine = VajraSearchOptimized(corpus, cache_size=0)
        path = tmp_path / "old.joblib"
        engine.save_index(path)
        data = joblib.load(path)
        data.pop("analyzer", None)
        data.pop("n_jobs", None)
        joblib.dump(data, path)

        loaded = VajraSearchOptimized.load_index(path, corpus)

        assert loaded.analyzer is preprocess_text
        assert "item1" in ids(loaded.search("item1"))


class TestTagAwareAnalyzer:
    def test_keeps_default_tokens_and_adds_joined_tag(self):
        assert tag_aware_analyzer("V-110") == ["v", "110", "v110"]

    def test_slash_suffix_expands_both_tags(self):
        terms = tag_aware_analyzer("P-400A/B pumps")
        assert "p400a" in terms and "p400b" in terms
        assert terms[:2] == ["p", "400a"]  # default tokens come first, in order

    def test_multi_hyphen_codes(self):
        assert "dnvstf101" in tag_aware_analyzer("DNV-ST-F101")
        assert "lhh1" in tag_aware_analyzer("high-high level (LHH-1)")

    def test_unhyphenated_tag_matches_hyphenated(self):
        assert "v110" in tag_aware_analyzer("V110")
        assert "v110" in tag_aware_analyzer("V-110")

    def test_no_false_tags(self):
        assert tag_aware_analyzer("state-of-the-art design on 2026-11-19") == \
               preprocess_text("state-of-the-art design on 2026-11-19")

    def test_plain_text_equals_default(self):
        text = "The quick brown fox jumps over the lazy dog"
        assert tag_aware_analyzer(text) == preprocess_text(text)

    def test_repeated_tags_keep_counts(self):
        assert tag_aware_analyzer("V-110 and V-110").count("v110") == 2

    def test_empty(self):
        assert tag_aware_analyzer("") == []


# --------------------------------------------------------------------------- #
# Fix 4: logging
# --------------------------------------------------------------------------- #
class TestLogging:
    def test_library_installs_only_a_null_handler(self):
        logger = logging.getLogger("vajra_bm25")
        assert logger.propagate is True
        assert logger.handlers
        assert all(isinstance(h, logging.NullHandler) for h in logger.handlers)

    def test_messages_reach_a_root_handler_once(self):
        records = []

        class Collect(logging.Handler):
            def emit(self, record):
                records.append(record)

        root = logging.getLogger()
        handler = Collect(level=logging.DEBUG)
        old_level = root.level
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        try:
            VajraSearchOptimized(make_corpus(), cache_size=0)
        finally:
            root.removeHandler(handler)
            root.setLevel(old_level)

        messages = [r.getMessage() for r in records if r.name.startswith("vajra_bm25")]
        assert any("Built dense index" in m for m in messages)
        assert len(messages) == len(set(messages))  # no message delivered twice

    def test_configure_logging_is_still_available_as_opt_in(self):
        from vajra_bm25.logging_config import configure_logging
        logger = configure_logging(level=logging.WARNING, logger_name="vajra_bm25_optin_test")
        try:
            assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
            assert logger.propagate is False
        finally:
            logger.handlers.clear()


# --------------------------------------------------------------------------- #
# Fix 5: version
# --------------------------------------------------------------------------- #
class TestVersion:
    def test_version_matches_package_metadata(self):
        assert vajra_bm25.__version__ == metadata.version("vajra-bm25")

    def test_version_is_0_5_1(self):
        assert vajra_bm25.__version__ == "0.5.1"
