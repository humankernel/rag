import pytest
from src.indexing.chunking.recursive import RecursiveTextSplitter


@pytest.fixture
def splitter():
    return RecursiveTextSplitter(max_length=10, overlap=2, overlap_by_words=True)


def test_basic_split(splitter):
    text = "This is a simple test sentence."
    chunks = splitter.split_text(text)
    assert chunks == [
        "This is a",
        "is a simple",
        "a simple test",
        "simple test sentence.",
    ]


def test_sentence_split(splitter):
    text = "Sentence one. Sentence two. Sentence three."
    chunks = splitter.split_text(text)
    assert chunks == [
        "Sentence one.",
        "Sentence one. Sentence two.",
        "Sentence two. Sentence three.",
    ]


def test_character_overlap():
    splitter = RecursiveTextSplitter(max_length=5, overlap=1, overlap_by_words=False)
    text = "abcdefg asdasd"
    chunks = splitter.split_text(text)
    assert chunks == ["abcdefg", "g asdasd"]


def test_empty_input(splitter):
    text = ""
    chunks = splitter.split_text(text)
    assert chunks == [""]


def test_max_length_respected():
    """
    Ensure that no chunk exceeds the specified max_length.
    """
    text = "Fast and extensible, Pydantic plays nicely with your linters/IDE/brain. Define how data should be in pure, canonical Python 3.8+; validate it with Pydantic."
    splitter = RecursiveTextSplitter(max_length=30, overlap=1, overlap_by_words=False)
    chunks = splitter.split_text(text)
    assert len(chunks) > 1  # Ensure it splits into multiple chunks
    for chunk in chunks:
        assert len(chunk) <= splitter.max_length


def test_overlap_consistency(splitter):
    """
    Ensure that overlap is consistent and does not exceed max_length.
    """
    text = "This is a test sentence with overlap."
    chunks = splitter.split_text(text)
    for i in range(1, len(chunks)):
        prev_chunk = chunks[i - 1]
        current_chunk = chunks[i]
        if splitter.overlap_by_words:
            overlap_words = prev_chunk.split()[-splitter.overlap :]
            assert " ".join(overlap_words) in current_chunk
        else:
            overlap_chars = prev_chunk[-splitter.overlap :]
            assert overlap_chars in current_chunk


def test_edge_case_small_max_length():
    """
    Test edge case where max_length is very small.
    """
    with pytest.raises(ValueError) as excinfo:
        splitter = RecursiveTextSplitter(
            max_length=1, overlap=0, overlap_by_words=False
        )
        text = "abc"
        chunks = splitter.split_text(text)
    assert "max_length is too small" in str(excinfo.value)


def test_edge_case_no_overlap():
    """
    Test behavior when overlap is set to zero.
    """
    splitter = RecursiveTextSplitter(max_length=5, overlap=0, overlap_by_words=True)
    text = "This is a test sentence."
    chunks = splitter.split_text(text)
    assert chunks == ["This", "is a", "test", "sentence."]
