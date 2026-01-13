"""
Tests for Document and DocumentCorpus classes.

Tests cover:
- Document creation and serialization
- DocumentCorpus operations
- JSONL persistence (save/load)
- Edge cases and error handling
"""

import pytest
import tempfile
import json
from pathlib import Path

from vajra_bm25.documents import Document, DocumentCorpus, create_sample_corpus


class TestDocument:
    """Tests for Document dataclass."""

    def test_document_creation(self):
        """Test basic document creation."""
        doc = Document(
            id="test_id",
            title="Test Title",
            content="Test content here"
        )

        assert doc.id == "test_id"
        assert doc.title == "Test Title"
        assert doc.content == "Test content here"
        assert doc.metadata is None

    def test_document_with_metadata(self):
        """Test document creation with metadata."""
        metadata = {"topic": "testing", "difficulty": "easy"}
        doc = Document(
            id="test_id",
            title="Test Title",
            content="Test content",
            metadata=metadata
        )

        assert doc.metadata == metadata
        assert doc.metadata["topic"] == "testing"

    def test_document_is_immutable(self):
        """Test that Document is frozen (immutable)."""
        doc = Document("1", "Title", "Content")

        with pytest.raises(Exception):  # FrozenInstanceError
            doc.id = "new_id"

    def test_document_hash(self):
        """Test that Document is hashable based on id."""
        doc1 = Document("1", "Title A", "Content A")
        doc2 = Document("1", "Title B", "Content B")
        doc3 = Document("2", "Title A", "Content A")

        # Same id should have same hash
        assert hash(doc1) == hash(doc2)

        # Different id should (likely) have different hash
        assert hash(doc1) != hash(doc3)

    def test_document_in_set(self):
        """Test that documents can be used in sets."""
        doc1 = Document("1", "Title", "Content")
        doc2 = Document("1", "Title", "Content")
        doc3 = Document("2", "Title", "Content")

        doc_set = {doc1, doc2, doc3}

        # doc1 and doc2 have same hash, so set should have 2 items
        assert len(doc_set) == 2

    def test_document_to_dict(self):
        """Test document serialization to dict."""
        doc = Document(
            id="test_id",
            title="Test Title",
            content="Test content",
            metadata={"key": "value"}
        )

        d = doc.to_dict()

        assert d["id"] == "test_id"
        assert d["title"] == "Test Title"
        assert d["content"] == "Test content"
        assert d["metadata"] == {"key": "value"}

    def test_document_from_dict(self):
        """Test document deserialization from dict."""
        data = {
            "id": "test_id",
            "title": "Test Title",
            "content": "Test content",
            "metadata": {"key": "value"}
        }

        doc = Document.from_dict(data)

        assert doc.id == "test_id"
        assert doc.title == "Test Title"
        assert doc.content == "Test content"
        assert doc.metadata == {"key": "value"}

    def test_document_from_dict_minimal(self):
        """Test document from dict without metadata."""
        data = {
            "id": "test_id",
            "title": "Test Title",
            "content": "Test content"
        }

        doc = Document.from_dict(data)

        assert doc.id == "test_id"
        assert doc.metadata is None

    def test_document_roundtrip(self):
        """Test document serialization roundtrip."""
        original = Document(
            id="test_id",
            title="Test Title",
            content="Test content",
            metadata={"nested": {"key": "value"}}
        )

        d = original.to_dict()
        restored = Document.from_dict(d)

        assert restored.id == original.id
        assert restored.title == original.title
        assert restored.content == original.content
        assert restored.metadata == original.metadata


class TestDocumentCorpus:
    """Tests for DocumentCorpus class."""

    def test_corpus_creation_empty(self):
        """Test empty corpus creation."""
        corpus = DocumentCorpus()

        assert len(corpus) == 0

    def test_corpus_creation_with_documents(self):
        """Test corpus creation with documents."""
        docs = [
            Document("1", "Title 1", "Content 1"),
            Document("2", "Title 2", "Content 2"),
        ]
        corpus = DocumentCorpus(docs)

        assert len(corpus) == 2

    def test_corpus_get_by_id(self):
        """Test retrieving document by ID."""
        docs = [
            Document("1", "Title 1", "Content 1"),
            Document("2", "Title 2", "Content 2"),
        ]
        corpus = DocumentCorpus(docs)

        doc = corpus.get("1")
        assert doc is not None
        assert doc.title == "Title 1"

    def test_corpus_get_missing(self):
        """Test getting non-existent document."""
        corpus = DocumentCorpus([Document("1", "Title", "Content")])

        doc = corpus.get("nonexistent")
        assert doc is None

    def test_corpus_add_document(self):
        """Test adding document to corpus."""
        corpus = DocumentCorpus()
        doc = Document("1", "Title", "Content")

        corpus.add(doc)

        assert len(corpus) == 1
        assert corpus.get("1") == doc

    def test_corpus_iteration(self):
        """Test iterating over corpus."""
        docs = [
            Document("1", "Title 1", "Content 1"),
            Document("2", "Title 2", "Content 2"),
            Document("3", "Title 3", "Content 3"),
        ]
        corpus = DocumentCorpus(docs)

        iterated = list(corpus)

        assert len(iterated) == 3
        assert iterated[0].id == "1"
        assert iterated[2].id == "3"


class TestJSONLPersistence:
    """Tests for JSONL save/load functionality."""

    def test_save_jsonl(self):
        """Test saving corpus to JSONL."""
        docs = [
            Document("1", "Title 1", "Content 1"),
            Document("2", "Title 2", "Content 2"),
        ]
        corpus = DocumentCorpus(docs)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            filepath = Path(f.name)

        try:
            corpus.save_jsonl(filepath)

            assert filepath.exists()

            # Read raw lines
            with open(filepath) as f:
                lines = f.readlines()

            assert len(lines) == 2

            # Verify JSON structure
            data1 = json.loads(lines[0])
            assert data1["id"] == "1"
            assert data1["title"] == "Title 1"
        finally:
            filepath.unlink()

    def test_load_jsonl(self):
        """Test loading corpus from JSONL."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"id": "1", "title": "Title 1", "content": "Content 1"}\n')
            f.write('{"id": "2", "title": "Title 2", "content": "Content 2"}\n')
            filepath = Path(f.name)

        try:
            corpus = DocumentCorpus.load_jsonl(filepath)

            assert len(corpus) == 2
            assert corpus.get("1").title == "Title 1"
            assert corpus.get("2").content == "Content 2"
        finally:
            filepath.unlink()

    def test_save_load_roundtrip(self):
        """Test complete save/load roundtrip."""
        original_docs = [
            Document("1", "Title 1", "Content 1", {"key": "value1"}),
            Document("2", "Title 2", "Content 2", {"key": "value2"}),
            Document("3", "Title 3", "Content 3", None),
        ]
        original_corpus = DocumentCorpus(original_docs)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            filepath = Path(f.name)

        try:
            # Save
            original_corpus.save_jsonl(filepath)

            # Load
            loaded_corpus = DocumentCorpus.load_jsonl(filepath)

            # Verify
            assert len(loaded_corpus) == 3

            for original in original_docs:
                loaded = loaded_corpus.get(original.id)
                assert loaded is not None
                assert loaded.id == original.id
                assert loaded.title == original.title
                assert loaded.content == original.content
                assert loaded.metadata == original.metadata
        finally:
            filepath.unlink()

    def test_load_jsonl_with_empty_lines(self):
        """Test loading JSONL with empty lines."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"id": "1", "title": "Title 1", "content": "Content 1"}\n')
            f.write('\n')  # Empty line
            f.write('{"id": "2", "title": "Title 2", "content": "Content 2"}\n')
            f.write('   \n')  # Whitespace line
            filepath = Path(f.name)

        try:
            corpus = DocumentCorpus.load_jsonl(filepath)

            # Should skip empty lines
            assert len(corpus) == 2
        finally:
            filepath.unlink()

    def test_save_jsonl_with_special_characters(self):
        """Test saving documents with special characters."""
        docs = [
            Document("1", "Title with \"quotes\"", "Content with\nnewlines"),
            Document("2", "Unicode: 日本語", "Emoji: 🎉"),
        ]
        corpus = DocumentCorpus(docs)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            filepath = Path(f.name)

        try:
            corpus.save_jsonl(filepath)
            loaded = DocumentCorpus.load_jsonl(filepath)

            assert loaded.get("1").title == "Title with \"quotes\""
            assert loaded.get("1").content == "Content with\nnewlines"
            assert loaded.get("2").title == "Unicode: 日本語"
            assert loaded.get("2").content == "Emoji: 🎉"
        finally:
            filepath.unlink()


class TestCreateSampleCorpus:
    """Tests for sample corpus creation helper."""

    def test_create_sample_corpus(self):
        """Test sample corpus creation."""
        corpus = create_sample_corpus()

        assert len(corpus) == 10  # Should have 10 sample documents
        assert corpus.get("doc1") is not None
        assert corpus.get("doc10") is not None

    def test_sample_corpus_has_metadata(self):
        """Test that sample corpus documents have metadata."""
        corpus = create_sample_corpus()

        doc = corpus.get("doc1")
        assert doc.metadata is not None
        assert "topic" in doc.metadata

    def test_sample_corpus_searchable(self):
        """Test that sample corpus works with search engine."""
        from vajra_bm25 import VajraSearch

        corpus = create_sample_corpus()
        engine = VajraSearch(corpus)

        results = engine.search("category functors", top_k=3)

        assert len(results) > 0


class TestPDFLoading:
    """Tests for PDF loading functionality."""

    @pytest.fixture
    def sample_pdf(self, tmp_path):
        """Create a sample PDF file for testing."""
        try:
            from pypdf import PdfWriter
        except ImportError:
            pytest.skip("pypdf not installed")

        pdf_path = tmp_path / "test_document.pdf"

        writer = PdfWriter()

        # Create a simple PDF with text
        # pypdf's PdfWriter needs pages from an existing PDF or created differently
        # We'll use a minimal approach - create empty pages and add annotations
        # For a proper test, we need reportlab or similar

        # Alternative: Create a minimal valid PDF manually
        # This is a workaround since pypdf PdfWriter can't easily create text
        minimal_pdf = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Test PDF Content) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000359 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
434
%%EOF"""

        with open(pdf_path, "wb") as f:
            f.write(minimal_pdf)

        return pdf_path

    @pytest.fixture
    def pdf_directory(self, tmp_path):
        """Create a directory with multiple PDF files."""
        try:
            from pypdf import PdfWriter
        except ImportError:
            pytest.skip("pypdf not installed")

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        minimal_pdf_template = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 50 >>
stream
BT
/F1 12 Tf
100 700 Td
(Document {num} content) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000365 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
440
%%EOF"""

        for i in range(3):
            pdf_path = pdf_dir / f"doc{i+1}.pdf"
            with open(pdf_path, "wb") as f:
                f.write(minimal_pdf_template.replace(b"{num}", str(i+1).encode()))

        return pdf_dir

    def test_load_pdf_single_file(self, sample_pdf):
        """Test loading a single PDF file."""
        corpus = DocumentCorpus.load_pdf(sample_pdf)

        assert len(corpus) == 1
        doc = corpus.documents[0]
        assert doc.id == "test_document"
        assert doc.metadata["format"] == "pdf"
        assert doc.metadata["pages"] == 1

    def test_load_pdf_with_custom_id(self, sample_pdf):
        """Test loading PDF with custom document ID."""
        corpus = DocumentCorpus.load_pdf(sample_pdf, doc_id="custom_id")

        assert len(corpus) == 1
        assert corpus.documents[0].id == "custom_id"

    def test_load_pdf_not_found(self, tmp_path):
        """Test loading non-existent PDF raises error."""
        with pytest.raises(FileNotFoundError):
            DocumentCorpus.load_pdf(tmp_path / "nonexistent.pdf")

    def test_load_pdf_directory(self, pdf_directory):
        """Test loading a directory of PDFs."""
        corpus = DocumentCorpus.load_pdf_directory(pdf_directory)

        assert len(corpus) == 3

        # Check all documents loaded
        doc_ids = {doc.id for doc in corpus.documents}
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids
        assert "doc3" in doc_ids

    def test_load_pdf_directory_empty(self, tmp_path):
        """Test loading empty directory returns empty corpus."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        corpus = DocumentCorpus.load_pdf_directory(empty_dir)

        assert len(corpus) == 0

    def test_load_pdf_directory_not_a_dir(self, sample_pdf):
        """Test loading file as directory raises error."""
        with pytest.raises(NotADirectoryError):
            DocumentCorpus.load_pdf_directory(sample_pdf)

    def test_load_auto_detect_pdf(self, sample_pdf):
        """Test auto-detection of PDF format."""
        corpus = DocumentCorpus.load(sample_pdf)

        assert len(corpus) == 1
        assert corpus.documents[0].metadata["format"] == "pdf"

    def test_load_auto_detect_directory(self, pdf_directory):
        """Test auto-detection of PDF directory."""
        corpus = DocumentCorpus.load(pdf_directory)

        assert len(corpus) == 3

    def test_load_auto_detect_jsonl(self, tmp_path):
        """Test auto-detection of JSONL format."""
        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write('{"id": "1", "title": "Title", "content": "Content"}\n')

        corpus = DocumentCorpus.load(jsonl_path)

        assert len(corpus) == 1
        assert corpus.documents[0].id == "1"

    def test_load_explicit_format(self, tmp_path):
        """Test loading with explicit format override."""
        jsonl_path = tmp_path / "data.txt"  # Non-standard extension
        with open(jsonl_path, "w") as f:
            f.write('{"id": "1", "title": "Title", "content": "Content"}\n')

        corpus = DocumentCorpus.load(jsonl_path, format="jsonl")

        assert len(corpus) == 1

    def test_load_unknown_format(self, tmp_path):
        """Test loading with unknown format raises error."""
        unknown_file = tmp_path / "data.xyz"
        unknown_file.touch()

        with pytest.raises(ValueError, match="Cannot auto-detect"):
            DocumentCorpus.load(unknown_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
