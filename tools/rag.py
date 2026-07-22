"""Local RAG index: embeds documents from ./data into a persistent Chroma store."""

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
PERSIST_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "research_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEXABLE_SUFFIXES = {".txt", ".md", ".pdf"}

_embeddings = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def _get_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_get_embeddings(),
        persist_directory=str(PERSIST_DIR),
    )


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def build_index() -> int:
    """(Re)build the Chroma index from every .txt/.md/.pdf file under ./data.

    Returns the number of chunks indexed.
    """
    paths = [
        p for p in DATA_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in INDEXABLE_SUFFIXES
    ]
    if not paths:
        logger.warning("[RAG] No documents found in %s", DATA_DIR)
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts: list[str] = []
    metadatas: list[dict] = []

    for path in paths:
        content = _read_file(path)
        if not content.strip():
            continue
        chunks = splitter.split_text(content)
        texts.extend(chunks)
        metadatas.extend({"source": path.name} for _ in chunks)

    store = _get_store()
    store.reset_collection()
    if texts:
        store.add_texts(texts, metadatas=metadatas)

    logger.info("[RAG] Indexed %d chunk(s) from %d file(s)", len(texts), len(paths))
    return len(texts)


def get_relevant_docs(query: str, k: int = 3) -> list[dict]:
    """Return up to k relevant chunks from the local knowledge base for query.

    Returns an empty list if the index hasn't been built yet or has no match.
    """
    if not PERSIST_DIR.exists():
        return []

    store = _get_store()
    if not store.get(limit=1)["ids"]:
        return []

    results = store.similarity_search(query, k=k)
    return [{"source": doc.metadata.get("source", "unknown"), "content": doc.page_content} for doc in results]
