import pytest
import torch
from sentence_transformers import SentenceTransformer

from src.indexing.chunking.semantic import SemanticTextSplitter

model_path = "/media/work/learn/ai/models/embeddings/bge-m3"


@pytest.fixture(scope="session")
def embedding_model():
    model = SentenceTransformer(model_path, device="cuda")
    yield model
    del model
    torch.cuda.empty_cache()


@pytest.fixture
def semantic_text_splitter(embedding_model):
    return SemanticTextSplitter(
        embedding_model_path=embedding_model, algorithm="dbscan"
    )


def test_empty_input(semantic_text_splitter):
    """
    Test handling of empty input.
    """
    with pytest.raises(ValueError, match="text is empty"):
        semantic_text_splitter.split_text("")


def test_single_sentence(semantic_text_splitter):
    """
    Test splitting when the input contains only one sentence.
    """
    text = "This is a single sentence."
    chunks = semantic_text_splitter.split_text(text)
    assert chunks == [text]  # The output should be the same as the input


def test_noise_handling_dbscan(semantic_text_splitter):
    """
    Test handling of noise (label -1) in DBSCAN clustering.
    """
    text = "a b c d e f g"
    chunks = semantic_text_splitter.split_text(text)
    print(f"{chunks=}")
    assert chunks == [text]


@pytest.mark.parametrize("algorithm", ["dbscan", "agglomerative"])
def test_algorithm_consistency(embedding_model, algorithm):
    """
    Test consistency of output across different algorithms.
    """
    text = (
        "The cat sat on the mat. Dogs are great companions. "
        "The sun is shining today. Cats and dogs are popular pets."
    )
    splitter = SemanticTextSplitter(
        embedding_model_path=embedding_model, algorithm=algorithm
    )
    chunks = splitter.split_text(text)

    # Ensure the output is consistent and valid
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)
