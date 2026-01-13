"""
Document model and persistence

Documents are objects in our category.
JSONL provides serialization/deserialization morphisms.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

from vajra_bm25.logging_config import get_logger

# Initialize logger for this module
logger = get_logger("documents")


@dataclass(frozen=True)
class Document:
    """
    A document object.

    Frozen (immutable) because documents are objects in a category.
    Identity is preserved through transformations.
    """
    id: str
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

    def __hash__(self):
        return hash(self.id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize document to dict"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Deserialize document from dict"""
        return cls(**data)


class DocumentCorpus:
    """
    A corpus of documents.

    Categorical interpretation:
    - Corpus is a collection of objects
    - Load/save are morphisms between file system and memory
    """

    def __init__(self, documents: List[Document] = None):
        self.documents = documents or []
        self._index: Dict[str, Document] = {doc.id: doc for doc in self.documents}

    def add(self, doc: Document):
        """Add a document to the corpus"""
        self.documents.append(doc)
        self._index[doc.id] = doc

    def get(self, doc_id: str) -> Optional[Document]:
        """Retrieve document by ID"""
        return self._index.get(doc_id)

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self):
        return iter(self.documents)

    def save_jsonl(self, filepath: Path):
        """
        Persist corpus to JSONL.

        Morphism: Corpus -> File
        """
        with open(filepath, 'w') as f:
            for doc in self.documents:
                f.write(json.dumps(doc.to_dict()) + '\n')

    @classmethod
    def load_jsonl(cls, filepath: Path) -> 'DocumentCorpus':
        """
        Load corpus from JSONL.

        Morphism: File -> Corpus
        """
        documents = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    documents.append(Document.from_dict(data))

        return cls(documents)

    @classmethod
    def load_pdf(cls, filepath: Path, doc_id: Optional[str] = None) -> 'DocumentCorpus':
        """
        Load a single PDF file as a document.

        Morphism: PDF File -> Corpus (single document)

        Args:
            filepath: Path to PDF file
            doc_id: Optional document ID (defaults to filename stem)

        Returns:
            DocumentCorpus containing the PDF as a single document

        Raises:
            ImportError: If pypdf is not installed
            FileNotFoundError: If PDF file doesn't exist
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "PDF support requires pypdf. Install with: pip install vajra-bm25[pdf]"
            )

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"PDF file not found: {filepath}")

        reader = PdfReader(filepath)

        # Extract text from all pages
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages_text.append(text)

        content = "\n\n".join(pages_text)

        # Try to extract title from PDF metadata
        title = filepath.name
        if reader.metadata and reader.metadata.title:
            title = reader.metadata.title

        doc = Document(
            id=doc_id or filepath.stem,
            title=title,
            content=content,
            metadata={
                "source": str(filepath.absolute()),
                "format": "pdf",
                "pages": len(reader.pages),
                "author": reader.metadata.author if reader.metadata else None,
            }
        )

        logger.info(f"Loaded PDF: {filepath.name} ({len(reader.pages)} pages, {len(content)} chars)")
        return cls([doc])

    @classmethod
    def load_pdf_directory(
        cls,
        dirpath: Path,
        recursive: bool = False
    ) -> 'DocumentCorpus':
        """
        Load all PDF files from a directory.

        Morphism: Directory -> Corpus

        Args:
            dirpath: Path to directory containing PDFs
            recursive: If True, search subdirectories recursively

        Returns:
            DocumentCorpus containing all PDFs as documents
        """
        dirpath = Path(dirpath)
        if not dirpath.is_dir():
            raise NotADirectoryError(f"Not a directory: {dirpath}")

        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = sorted(dirpath.glob(pattern))

        if not pdf_files:
            logger.warning(f"No PDF files found in {dirpath}")
            return cls([])

        documents = []
        for pdf_file in pdf_files:
            try:
                corpus = cls.load_pdf(pdf_file)
                documents.extend(corpus.documents)
            except Exception as e:
                logger.error(f"Failed to load {pdf_file}: {e}")

        logger.info(f"Loaded {len(documents)} PDFs from {dirpath}")
        return cls(documents)

    @classmethod
    def load(cls, path: Path, format: Optional[str] = None) -> 'DocumentCorpus':
        """
        Load corpus from file or directory, auto-detecting format.

        Morphism: Path -> Corpus

        Args:
            path: Path to file or directory
            format: Optional format hint ('jsonl', 'pdf', 'pdf_dir')
                   If None, auto-detects based on path

        Returns:
            DocumentCorpus
        """
        path = Path(path)

        # Auto-detect format
        if format is None:
            if path.is_dir():
                format = "pdf_dir"
            elif path.suffix.lower() == ".pdf":
                format = "pdf"
            elif path.suffix.lower() in (".jsonl", ".json"):
                format = "jsonl"
            else:
                raise ValueError(f"Cannot auto-detect format for: {path}")

        # Load based on format
        if format == "jsonl":
            return cls.load_jsonl(path)
        elif format == "pdf":
            return cls.load_pdf(path)
        elif format == "pdf_dir":
            return cls.load_pdf_directory(path)
        else:
            raise ValueError(f"Unknown format: {format}")


def create_sample_corpus() -> DocumentCorpus:
    """
    Create a sample corpus for demonstration.

    Topics: Category theory, functional programming, search algorithms
    """
    documents = [
        Document(
            id="doc1",
            title="Introduction to Category Theory",
            content="""Category theory is a general theory of mathematical structures and their relations.
            It provides a unifying framework for mathematics. Categories consist of objects and morphisms.
            Morphisms are structure-preserving maps between objects. Composition of morphisms is associative.
            Every object has an identity morphism. Functors map between categories preserving structure.""",
            metadata={"topic": "category_theory", "difficulty": "introductory"}
        ),
        Document(
            id="doc2",
            title="Functors and Natural Transformations",
            content="""Functors are structure-preserving maps between categories. A functor F maps objects to objects
            and morphisms to morphisms. Natural transformations provide morphisms between functors.
            They satisfy a naturality condition. Examples include the list functor and maybe functor.
            Functors preserve composition and identities.""",
            metadata={"topic": "category_theory", "difficulty": "intermediate"}
        ),
        Document(
            id="doc3",
            title="Coalgebras and Dynamics",
            content="""Coalgebras capture generative or productive processes. A coalgebra consists of a carrier
            and a structure map from the carrier to a functor applied to the carrier. Unlike algebras which
            fold or consume, coalgebras unfold or produce. They are perfect for modeling dynamics, streams,
            and transition systems. The structure map determines how states evolve.""",
            metadata={"topic": "category_theory", "difficulty": "advanced"}
        ),
        Document(
            id="doc4",
            title="Functional Programming Basics",
            content="""Functional programming emphasizes pure functions and immutable data structures.
            Functions are first-class citizens. Higher-order functions accept functions as arguments.
            Map, filter, and reduce are fundamental operations. Recursion replaces iteration.
            Functional programming draws heavily from lambda calculus and category theory.""",
            metadata={"topic": "programming", "difficulty": "introductory"}
        ),
        Document(
            id="doc5",
            title="Monads in Programming",
            content="""Monads are a design pattern from category theory used in functional programming.
            A monad wraps values with context. The Maybe monad handles nullable values.
            The List monad represents nondeterminism. The State monad encapsulates stateful computation.
            Monads have unit and bind operations satisfying monad laws.""",
            metadata={"topic": "programming", "difficulty": "intermediate"}
        ),
        Document(
            id="doc6",
            title="Breadth-First Search Algorithm",
            content="""BFS explores a graph level by level. It uses a queue data structure.
            BFS finds shortest paths in unweighted graphs. Time complexity is O(V + E).
            Applications include web crawling, social networks, and GPS navigation.
            BFS is complete and optimal for unweighted graphs.""",
            metadata={"topic": "algorithms", "difficulty": "introductory"}
        ),
        Document(
            id="doc7",
            title="Depth-First Search Algorithm",
            content="""DFS explores as far as possible along each branch before backtracking.
            It uses a stack data structure or recursion. DFS is used for topological sorting,
            cycle detection, and maze solving. Time complexity is O(V + E).
            DFS is not guaranteed to find shortest paths.""",
            metadata={"topic": "algorithms", "difficulty": "introductory"}
        ),
        Document(
            id="doc8",
            title="Advanced Search Techniques",
            content="""Modern search algorithms combine multiple techniques. A* search uses heuristics
            for efficient pathfinding. Iterative deepening combines BFS and DFS benefits.
            Bidirectional search starts from both ends. Best-first search uses priority queues.
            These algorithms are used in AI, robotics, and game development.""",
            metadata={"topic": "algorithms", "difficulty": "advanced"}
        ),
        Document(
            id="doc9",
            title="Lambda Calculus Foundations",
            content="""Lambda calculus is a formal system for expressing computation through function
            abstraction and application. It consists of variables, abstraction, and application.
            Church encoding represents data as functions. Lambda calculus is Turing complete.
            It forms the theoretical basis for functional programming languages.""",
            metadata={"topic": "theory", "difficulty": "advanced"}
        ),
        Document(
            id="doc10",
            title="Category Theory Applications",
            content="""Category theory has applications in computer science, physics, and linguistics.
            In programming, it provides abstractions like functors and monads. In database theory,
            it models schemas and queries. In quantum mechanics, it describes quantum processes.
            Type theory and category theory are deeply connected. Categorical logic provides
            foundations for constructive mathematics.""",
            metadata={"topic": "category_theory", "difficulty": "advanced"}
        ),
    ]

    return DocumentCorpus(documents)


if __name__ == "__main__":
    # Create sample corpus
    corpus = create_sample_corpus()

    # Save to JSONL
    output_path = Path("sample_corpus.jsonl")
    corpus.save_jsonl(output_path)
    logger.info(f"Saved {len(corpus)} documents to {output_path}")

    # Load back
    loaded = DocumentCorpus.load_jsonl(output_path)
    logger.info(f"Loaded {len(loaded)} documents from {output_path}")

    # Display sample
    logger.info("Sample documents:")
    for doc in list(loaded)[:3]:
        logger.info(f"[{doc.id}] {doc.title}")
        logger.info(f"  {doc.content[:100]}...")
