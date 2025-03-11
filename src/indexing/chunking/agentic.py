import re
from typing import List
import llama_cpp
import instructor
from pydantic import BaseModel


class Chunk(BaseModel):
    text: str
    context: str


class ChunkResponse(BaseModel):
    chunks: List[Chunk]


LLM_PATH = "/media/work/learn/ai/models/llm/meta/llama-3.2-1b-Instruct/llama-3.2-1b-instruct-q8_0.gguf"


class AgenticChunker:
    def __init__(
        self, model_path: str = LLM_PATH, min_chunk_size: int = 200, max_chunk_size=1000
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.model_path = model_path
        self.model = llama_cpp.Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False,
        )
        self.create = instructor.patch(
            create=self.model.create_chat_completion,
            mode=instructor.Mode.JSON_SCHEMA,
        )

    def chunk_text(self, text: str) -> List[Chunk]:
        # Split into paragraphs
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        # Group small paragraphs
        grouped_paragraphs = self._group_paragraphs(paragraphs)

        # Process each group
        all_chunks = []
        for group in grouped_paragraphs:
            chunks = self._process_chunk(group)
            all_chunks.extend(chunks)
        return all_chunks

    def _group_paragraphs(self, paragraphs: List[str]) -> List[str]:
        groups = []
        current_group = ""

        for para in paragraphs:
            # If adding this paragraph would exceed max size, start new group
            if len(current_group) + len(para) > self.max_chunk_size:
                if current_group:
                    groups.append(current_group)
                current_group = para
            else:
                if current_group:
                    current_group += "\n\n"
                current_group += para

        if current_group:
            groups.append(current_group)
        return groups

    def _process_chunk(self, text: str) -> List[Chunk]:
        prompt = f"""Analyze this text and split it into independent ideas. For each chunk:
1. Keep the original text exactly as is
2. Add a 3-5 word context summary

Text to process:
{text}"""

        response = self.create(
            messages=[{"role": "user", "content": prompt}],
            response_model=ChunkResponse,
            temperature=0.3,
            max_tokens=4096,
        )

        return response.chunks


if __name__ == "__main__":
    chunker = AgenticChunker()

    with open("book.txt", "r") as f:
        long_text = f.read()
    chunks = chunker.chunk_text(long_text)

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:")
        print(f"Text: {chunk.text}")
        print(f"Context: {chunk.context}\n")
