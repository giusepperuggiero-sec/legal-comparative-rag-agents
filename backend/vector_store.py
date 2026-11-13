"""FAISS Flat L2 vector store lifecycle manager with dynamic config and path verification."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.configuration import RagConfig


class VectorStoreManager:
    """Manage chunking, embedding generation, FAISS indexing, and verified local loading."""

    def __init__(self, store_path: Path | str | None = None) -> None:
        active_cfg = RagConfig.load()
        self.base_dir = RagConfig.BASE_DIR.resolve()
        target_path = Path(store_path) if store_path is not None else RagConfig.VECTOR_STORE_PATH
        self.store_path = target_path.resolve()

        # Prevent path escape outside project root unless an explicit test temp path is supplied
        if store_path is None and self.base_dir not in self.store_path.parents and self.store_path != self.base_dir:
            raise ValueError(f"Unsafe vector store path outside workspace: {self.store_path}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=str(active_cfg["EMBEDDING_MODEL_NAME"]),
            model_kwargs={"device": "cpu"},
        )

    def create_database(self, documents: List[Document]) -> None:
        """Split documents using current configuration and persist a FAISS Flat L2 index."""
        if not documents:
            print("[VECTOR STORE ERROR] No documents provided for indexing.")
            return

        active_cfg = RagConfig.load()
        chunk_size = int(active_cfg["CHUNK_SIZE"])
        chunk_overlap = int(active_cfg["CHUNK_OVERLAP"])

        print(f"[VECTOR STORE] Indexing {len(documents)} documents (chunk={chunk_size}, overlap={chunk_overlap})...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        final_docs = splitter.split_documents(documents)
        print(f"[VECTOR STORE] Generated {len(final_docs)} chunks.")

        if self.store_path.exists():
            try:
                shutil.rmtree(self.store_path)
            except PermissionError:
                print("[VECTOR STORE WARNING] Existing directory locked; attempting in-place overwrite.")
            except OSError as exc:
                print(f"[VECTOR STORE WARNING] Cleanup issue: {exc}")

        self.store_path.mkdir(parents=True, exist_ok=True)
        try:
            vector_store = FAISS.from_documents(final_docs, self.embeddings)
            vector_store.save_local(str(self.store_path))
            print(f"[VECTOR STORE] Index persisted at {self.store_path}")
        except PermissionError as exc:
            raise PermissionError(
                "Vector store files are locked by an active session. Close the Chatbot page and retry."
            ) from exc

    def load_local(self) -> FAISS:
        """Load the locally built trusted FAISS index after verifying artifacts exist."""
        index_file = self.store_path / "index.faiss"
        metadata_file = self.store_path / "index.pkl"
        if not index_file.is_file() or not metadata_file.is_file():
            raise FileNotFoundError(
                f"FAISS index not found in '{self.store_path}'. Run the Vector DB Builder page first."
            )

        # Trusted local index generated strictly from the project's corpus via VectorStoreManager
        return FAISS.load_local(
            str(self.store_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
