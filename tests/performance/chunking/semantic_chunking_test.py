import pytest
import torch
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from src.indexing.chunking.semantic import SemanticTextSplitter
from src.indexing.chunking.meta import MetaChunker
from transformers import AutoTokenizer, AutoModelForCausalLM

embedding_path = "/media/work/learn/ai/models/embeddings/bge-m3"
model_path = "/media/work/learn/ai/models/llm/meta/llama-3.2-1b-Instruct/llama-3.2-1b-instruct-q8_0.gguf"
model_id = "/media/work/learn/ai/models/llm/meta/llama-3.2-1b-Instruct/"


@pytest.fixture(scope="session")
def embedding_model():
    model = SentenceTransformer(embedding_path, device="cuda")
    yield model
    del model
    torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def model():
    model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=model_path).to(
        "cuda"
    )
    yield model
    del model
    torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=model_path)
    yield tokenizer
    del tokenizer
    torch.cuda.empty_cache()


@pytest.mark.parametrize("chunker", ["semantic_chunking", "meta_chunking"])
def test_semantic_chunking_quality(chunker, model, tokenizer, embedding_model):
    """Test that semantically related sentences are correctly grouped together."""
    seasons = [
        "The vibrant colors of autumn leaves create a stunning tapestry across the landscape.",
        "As spring arrives, flowers bloom in a dazzling array of colors, signaling the end of winter's chill.",
    ]
    tech = [
        "The rapid advancement of artificial intelligence is transforming industries and reshaping the workforce.",
        "Innovative technologies like blockchain are revolutionizing how we conduct transactions and manage data.",
        "The rise of smart home devices is making everyday life more convenient and efficient for consumers.",
    ]
    pets = [
        "Cats are independent animals. They often sleep during the day.",
        "Dogs are loyal and energetic. They enjoy playing fetch.",
        "Both cats and dogs are popular pets.",
    ]
    text = " ".join(seasons + tech + pets)

    if chunker == "semantic_chunking":
        splitter = SemanticTextSplitter(embedding_model, algorithm="agglomerative")
    elif chunker == "meta_chunking":
        splitter = MetaChunker(model, tokenizer)
    else:
        return

    chunks = splitter.split_text(text)
    chunk_sets = [set(sent_tokenize(chunk)) for chunk in chunks]

    # Create expected groups with proper sentence tokenization
    expected_groups = [
        set(sent_tokenize(" ".join(group))) for group in [seasons, tech, pets]
    ]

    # Verify each expected group exists exactly in one chunk
    for expected_group in expected_groups:
        matches = [chunk for chunk in chunk_sets if chunk.issuperset(expected_group)]
        assert len(matches) == 1, f"Group not properly contained: {expected_group}"


@pytest.mark.parametrize("chunker", ["semantic_chunking", "meta_chunking"])
def test_semantic_chunking_quality2(chunker, model, tokenizer, embedding_model):
    travel = [
        "Traveling to new countries allows you to experience diverse cultures and cuisines.",
        "Exploring different nations provides the opportunity to immerse yourself in unique traditions and local foods.",
    ]
    tech = [
        "The rapid advancement of artificial intelligence is transforming industries and reshaping job markets.",
        "Renewable energy technologies are crucial for combating climate change and promoting sustainable development.",
    ]
    extra = [
        "Gardening can be a therapeutic hobby, offering a peaceful escape from the hustle and bustle of everyday life."
    ]
    text = " ".join(travel + tech + extra)

    if chunker == "semantic_chunking":
        splitter = SemanticTextSplitter(embedding_model, algorithm="agglomerative")
    elif chunker == "meta_chunking":
        splitter = MetaChunker(model, tokenizer)
    else:
        return

    chunks = splitter.split_text(text)
    chunk_sets = [set(sent_tokenize(chunk)) for chunk in chunks]

    # Create expected groups with proper sentence tokenization
    expected_groups = [set(group) for group in [travel, tech, extra]]

    # Verify each expected group exists exactly in one chunk
    for expected_group in expected_groups:
        matches = [chunk for chunk in chunk_sets if chunk.issuperset(expected_group)]
        assert len(matches) == 1, f"Group not properly contained: {expected_group}"
