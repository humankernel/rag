import pytest
from unittest.mock import Mock
import torch
from nltk.tokenize import sent_tokenize
from src.indexing.chunking.meta import MetaChunker
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/media/work/learn/ai/models/llm/meta/llama-3.2-1b-Instruct/llama-3.2-1b-instruct-q8_0.gguf"
model_id = "/media/work/learn/ai/models/llm/meta/llama-3.2-1b-Instruct/"


@pytest.fixture(scope="session")
def mock_model():
    model = Mock()
    model.device = "cpu"

    # Mock model responses to simulate perplexity patterns
    def forward_mock(input_ids, attention_mask, **kwargs):
        response = Mock()

        # Predefined loss pattern for testing
        # Simulating low loss (high prob) within groups, high loss between groups
        if input_ids.shape[1] == 3:  # First batch
            loss = torch.tensor([0.1, 0.1, 0.9])
        else:  # Second batch
            loss = torch.tensor([0.1, 0.9, 0.1])

        response.logits = torch.randn(1, input_ids.shape[1], 50257)  # Random logits
        response.past_key_values = None
        return response

    model.forward = forward_mock
    return model


@pytest.fixture(scope="session")
def mock_tokenizer():
    tokenizer = Mock()
    tokenizer.return_tensors = "pt"
    tokenizer.add_special_tokens = False
    tokenizer.side_effect = lambda text, **kwargs: {
        "input_ids": torch.tensor([[1] * len(text.split())]),
        "attention_mask": torch.tensor([[1] * len(text.split())]),
    }
    return tokenizer


def test_empty_input(mock_model, mock_tokenizer):
    chunker = MetaChunker(mock_model, mock_tokenizer)
    assert chunker.split_text("") == []


def test_single_sentence(mock_model, mock_tokenizer):
    chunker = MetaChunker(mock_model, mock_tokenizer)
    text = "This is a single sentence."
    assert chunker.split_text(text) == [text]


# def test_perplexity_boundary_detection(mock_model, mock_tokenizer):
#     chunker = MetaChunker(mock_model, mock_tokenizer, threshold=0.5)

#     # Simulated perplexity values with clear minima at index 2
#     test_perplexities = [2.0, 1.9, 0.5, 2.1, 2.0]
#     boundaries = chunker._find_semantic_boundaries(test_perplexities)
#     assert boundaries == [2]

# def test_dynamic_merging(mock_model, mock_tokenizer):
#     chunker = MetaChunker(
#         mock_model, mock_tokenizer,
#         dynamic_merge=True,
#         target_chunk_size=50
#     )

#     text = ". ".join(["Short sentence"] * 10)
#     chunks = chunker.split_text(text)

#     # Verify merging happened
#     assert len(chunks) < 10
#     # Verify chunk sizes under target
#     assert all(len(chunk.split()) <= 50 for chunk in chunks)

# @pytest.mark.performance
# def test_chunking_quality_benchmark():
#     """End-to-end quality test with semantic similarity verification"""
#     model = AutoModelForCausalLM.from_pretrained("gpt2")
#     tokenizer = AutoTokenizer.from_pretrained("gpt2")

#     # Create test document with known semantic groups
#     group1 = [
#         "Neural networks require large amounts of training data.",
#         "Deep learning models use multiple hidden layers.",
#         "Backpropagation is used to adjust model weights."
#     ]

#     group2 = [
#         "Quantum computing uses qubits instead of classical bits.",
#         "Superposition allows quantum states to exist simultaneously.",
#         "Quantum entanglement enables correlated particle behavior."
#     ]

#     mixed_text = " ".join(group1 + group2 + group1)

#     chunker = MetaChunker(model, tokenizer, threshold=1.2)
#     chunks = chunker.split_text(mixed_text)

#     # Verify we get 3 chunks (group1, group2, group1)
#     assert len(chunks) == 3

#     # Verify chunk contents using embedding similarity
#     from sentence_transformers import SentenceTransformer
#     embedder = SentenceTransformer('all-MiniLM-L6-v2')

#     # Calculate intra-chunk and inter-chunk similarities
#     chunk_embeddings = [embedder.encode(chunk) for chunk in chunks]

#     intra_similarities = []
#     for emb in chunk_embeddings:
#         sim = np.mean(emb @ emb.T)
#         intra_similarities.append(sim)

#     inter_similarities = []
#     for i in range(len(chunk_embeddings)):
#         for j in range(i+1, len(chunk_embeddings)):
#             sim = np.mean(chunk_embeddings[i] @ chunk_embeddings[j].T)
#             inter_similarities.append(sim)

#     # Intra-chunk similarity should be higher than inter-chunk
#     assert np.mean(intra_similarities) > np.mean(inter_similarities), \
#         "Chunks show poor semantic cohesion"

# @pytest.mark.performance
# def test_processing_speed(benchmark):
#     model = AutoModelForCausalLM.from_pretrained("gpt2")
#     tokenizer = AutoTokenizer.from_pretrained("gpt2")
#     chunker = MetaChunker(model, tokenizer)

#     # Generate large test text
#     large_text = ". ".join(["This is a test sentence."] * 1000)

#     # Benchmark performance
#     result = benchmark(chunker.split_text, large_text)

#     assert len(result) > 0
#     assert benchmark.stats['mean'] < 5.0  # Max 5 seconds average
