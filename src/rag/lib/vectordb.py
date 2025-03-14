import json
import logging
import os
from pathlib import Path
import pickle
import time
from dataclasses import dataclass
from itertools import batched
from typing import Literal, Optional, TypedDict
from uuid import uuid4

import numpy as np
import pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from rag.lib.context import embedding_model_context, reranker_context
from rag.lib.schema import Chunk, Document
from rag.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Scores(TypedDict):
    dense_score: float
    sparse_score: float
    hybrid_score: float
    rerank_score: float


@dataclass
class RetrievedChunk:
    chunk: Chunk
    scores: Scores

    def __repl__(self) -> str:
        return (
            f"Source {self.chunk.doc_id}\n"
            f"Scores: (Dense: {self.scores['dense_score']:.3f}, Sparse: {self.scores['sparse_score']:.3f})"
            f"Hybrid {self.scores['hybrid_score']:.3f} - Rerank {self.scores['rerank_score']:.3f}"
            f"Text:\n {self.chunk.text[:300]} ... "
        )


class VectorDB:
    def __init__(
        self,
        name: str,
        splitter=RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=True),
        embedding_model_path: str = settings.EMBEDDING_MODEL_PATH,
        reranker_model_path: str = settings.RERANKER_MODEL_PATH,
        device: Literal["cuda", "cpu"] = settings.DEVICE,
    ) -> None:
        self.name = name
        self.splitter = splitter
        self.db_path = (
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / name
            / "vectordb.pkl"
        )
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self.dense_embeddings = []
        self.sparse_embeddings = []
        self.query_cache = {}
        # params
        self.embedding_model_path = embedding_model_path
        self.reranker_model_path = reranker_model_path
        self.device = device

    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    def insert(self, paths: list[str]) -> None:
        """Process and index documents"""
        assert paths, "You should pass a list of file paths"
        assert all(os.path.exists(path) for path in paths), (
            f"Each paths should exists {paths}"
        )
        assert all(path.endswith(".pdf") for path in paths), (
            f"Every file should be pdf {paths}"
        )

        docs: list[Document] = []
        chunks: list[Chunk] = []

        for path in paths:
            logger.info(f"Processing document: {path}")

            doc_id = str(uuid4())
            docs.append(
                Document(id=doc_id, source=path, metadata={"created_at": time.time()})
            )
            with pymupdf.open(path) as doc:
                for index, page in enumerate(doc):
                    page_text = page.get_text()
                    page_chunks_texts: list[str] = self.splitter.split_text(page_text)
                    page_chunks = (
                        Chunk(
                            id=str(uuid4()),
                            doc_id=doc_id,
                            page=index,
                            text=text.strip(),
                        )
                        for text in page_chunks_texts
                        if text.strip()
                    )
                    chunks.extend(page_chunks)
                    logger.info(f"Generated {len(chunks)} chunks from {path}")

        self._insert(docs, chunks)

    def _insert(
        self, docs: list[Document], chunks: list[Chunk], batch_size: int = 32
    ) -> None:
        """Add documents with automatic batching"""
        assert all(isinstance(doc, Document) for doc in docs), (
            "Every doc should be of type `Document`"
        )

        if os.path.exists(self.db_path):
            self.load_db()
            return

        self.documents.extend(docs)
        self.chunks.extend(chunks)
        texts = (chunk.text for chunk in chunks)

        logger.debug(f"Indexing {len(docs)} documents.")
        with (
            tqdm(total=len(chunks), desc="Embedding", mininterval=0.5) as pbar,
            embedding_model_context(
                path=self.embedding_model_path, device=self.device
            ) as model,
        ):
            for batch in batched(texts, batch_size):
                result = model.encode(
                    batch,
                    batch_size=batch_size,
                    return_dense=True,
                    return_sparse=True,
                )
                self.dense_embeddings.extend(result["dense_vecs"])
                self.sparse_embeddings.extend(result["lexical_weights"])
                pbar.update(len(batch))
        logger.info(f"Loaded {len(chunks)} chunks.")
        self.save_db()

    # see: https://github.com/FlagOpen/FlagEmbedding/blob/master/Tutorials/1_Embedding/1.2.4_BGE-M3.ipynb
    # fix: different sparse_scores each time
    def search(
        self,
        query: str,
        threshold: Optional[float] = None,
        top_k: int = 20,
        top_r: int = 10,
        alpha: float = 0.6,
        rerank: bool = True,
    ) -> list[RetrievedChunk]:
        """Hybrid Search with Reranking"""
        if not self.dense_embeddings or not self.sparse_embeddings:
            logger.error("No data loaded.")
            raise ValueError("No data loaded in the database.")

        # Get or cache query embedding
        if query not in self.query_cache:
            with embedding_model_context(
                path=self.embedding_model_path, device=self.device
            ) as model:
                result = model.encode([query], return_dense=True, return_sparse=True)
                self.query_cache[query] = (
                    result["dense_vecs"][0],
                    result["lexical_weights"][0],
                )

        q_dense, q_sparse = self.query_cache[query]

        # Calculate scores
        dense_scores = q_dense @ np.array(self.dense_embeddings).T
        sparse_scores = np.array(
            [
                self._compute_lexical_matching_score(q_sparse, doc_sparse)
                for doc_sparse in self.sparse_embeddings
            ]
        )
        # Normalize the Sparse Scores [0..1]
        min_score, max_score = sparse_scores.min(), sparse_scores.max()
        sparse_scores = (
            ((sparse_scores - min_score) / (max_score - min_score))
            if max_score != min_score
            else np.zeros_like(sparse_scores)
        )

        # Combine scores (w/ alpha weighting)
        hybrid_scores = alpha * dense_scores + (1 - alpha) * sparse_scores

        # Get initial candidates
        top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]
        if threshold:
            top_indices = [
                idx for idx in top_indices if hybrid_scores[idx] >= threshold
            ]

        # Rerank top results
        rerank_scores = None
        if rerank and len(top_indices) > 0:
            top_texts = (self.chunks[i].text for i in top_indices)
            pairs = [(query, text) for text in top_texts]

            with reranker_context(
                path=self.reranker_model_path, device=self.device
            ) as reranker:
                # todo: impl batch
                rerank_scores = reranker.compute_score(pairs, normalize=True)

            top_indices = np.argsort(rerank_scores)[-top_r:][::-1]
            if threshold:
                top_indices = [
                    idx for idx in top_indices if rerank_scores[idx] >= threshold
                ]

        return [
            RetrievedChunk(
                chunk=self.chunks[idx],
                scores={
                    "dense_score": float(dense_scores[idx]),
                    "sparse_score": float(sparse_scores[idx]),
                    "hybrid_score": float(hybrid_scores[idx]),
                    "rerank_score": float(rerank_scores[idx]) if rerank_scores else 0.0,
                },
            )
            for idx in top_indices
        ]

    # This is a copy of the function from the FlagEmbedding library
    # The purpose of copying the fn is to avoid using the model again
    def _compute_lexical_matching_score(
        self, lexical_weights_1: dict, lexical_weights_2: dict
    ):
        scores = 0
        for token, weight in lexical_weights_1.items():
            if token in lexical_weights_2:
                scores += weight * lexical_weights_2[token]
        return scores

    def save_db(self) -> None:
        logger.info("Saving VectorDB to disk.")
        data = {
            "documents": self.documents,
            "chunks": self.chunks,
            "dense_embeddings": self.dense_embeddings,
            "sparse_embeddings": self.sparse_embeddings,
            "query_cache": json.dumps(self.query_cache),
        }
        # Create parent directory if needed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)
        logger.info(f"Saved to {self.db_path}!")

    def load_db(self) -> None:
        logger.info("Loading VectorDB from disk.")
        if not self.db_path.exists():
            raise ValueError(f"Database not found at {self.db_path}")
        with open(self.db_path, "rb") as file:
            data = pickle.load(file)
        self.documents = data.get("documents")
        self.chunks = data.get("chunks")
        self.dense_embeddings = data.get("dense_embeddings")
        self.sparse_embeddings = data.get("sparse_embeddings")
        self.query_cache = json.loads(data.get("query_cache"))
        logger.info("Loaded!")
