import pytest
from nltk.tokenize import sent_tokenize
from src.indexing.chunking.meta import MetaChunker
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer

# Model and tokenizer setup
embedding_path = "/media/work/learn/ai/models/embeddings/bge-m3"
model_path = "/media/work/learn/ai/models/llm/meta/llama-3.2-1b-Instruct/llama-3.2-1b-instruct-q8_0.gguf"
model_id = "/media/work/learn/ai/models/llm/meta/llama-3.2-1b-Instruct/"


@pytest.fixture(scope="session")
def model():
    return AutoModelForCausalLM.from_pretrained(model_id, gguf_file=model_path).to(
        "cuda"
    )


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(model_id, gguf_file=model_path)


def calculate_coherence_score(
    chunk: str, embedding_model: SentenceTransformer
) -> float:
    """
    Calculate coherence score for a chunk using cosine similarity between sentence embeddings.
    Higher scores indicate better coherence.
    """
    sentences = sent_tokenize(chunk)
    embeddings = embedding_model.encode(sentences, normalize_embeddings=True)

    # Compute pairwise cosine similarity
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            similarities.append(sim)

    # Return average similarity as coherence score
    return np.mean(similarities) if similarities else 0.0


def calculate_boundary_accuracy(actual_chunks, expected_chunks):
    """
    Calculate the proportion of sentences correctly grouped into chunks.
    """
    total_sentences = sum(len(sent_tokenize(chunk)) for chunk in expected_chunks)
    correct_sentences = 0

    for actual_chunk in actual_chunks:
        actual_sents = set(sent_tokenize(actual_chunk))
        for expected_chunk in expected_chunks:
            expected_sents = set(sent_tokenize(expected_chunk))
            if actual_sents.issubset(expected_sents) or actual_sents.issuperset(
                expected_sents
            ):
                correct_sentences += len(actual_sents)
                break

    return correct_sentences / total_sentences if total_sentences > 0 else 0.0


def test_meta_chunker_performance(model, tokenizer):
    """
    Test the performance of the MetaChunker and assert it meets predefined thresholds.
    """
    # Initialize the chunker
    chunker = MetaChunker(model, tokenizer, threshold=1.5)

    del model
    del tokenizer

    embedding_model = SentenceTransformer(embedding_path, device="cuda")

    # Input text with clear logical boundaries
    travel = [
        "Traveling to new countries allows you to experience diverse cultures and cuisines",
        "Exploring different nations provides the opportunity to immerse yourself in unique traditions and local foods",
    ]
    print(len(" ".join(travel)))
    tech = [
        "The rapid advancement of artificial intelligence is transforming industries and reshaping job markets",
        "Renewable energy technologies are crucial for combating climate change and promoting sustainable development.",
    ]
    extra = [
        "Gardening can be a therapeutic hobby, offering a peaceful escape from the hustle and bustle of everyday life"
    ]
    text = " ".join(travel + tech)

    # Expected chunks (perfectly split)
    expected_chunks = [" ".join(travel), " ".join(tech), " ".join(extra)]

    # Perform chunking
    actual_chunks = chunker.split_text(text)
    print(f"{actual_chunks=}")

    # Metric 1: Chunk Coherence
    coherence_scores = [
        calculate_coherence_score(chunk, embedding_model) for chunk in actual_chunks
    ]
    avg_coherence = np.mean(coherence_scores)
    min_coherence_threshold = 0.6  # Minimum acceptable coherence score
    assert avg_coherence >= min_coherence_threshold, (
        f"Average coherence score {avg_coherence} below threshold {min_coherence_threshold}"
    )

    # Metric 2: Boundary Accuracy
    boundary_accuracy = calculate_boundary_accuracy(actual_chunks, expected_chunks)
    boundary_accuracy_threshold = 0.8  # Proportion of sentences correctly grouped
    assert boundary_accuracy >= boundary_accuracy_threshold, (
        f"Boundary accuracy {boundary_accuracy} below threshold {boundary_accuracy_threshold}"
    )
