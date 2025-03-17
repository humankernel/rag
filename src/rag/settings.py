from typing import Literal, Optional
from pydantic import BaseModel, computed_field
import torch


class Settings(BaseModel):
    """Configuration for the app"""

    # Paths
    EMBEDDING_MODEL_PATH: str = "/media/work/learn/ai/models/embeddings/bge-m3"
    EMBEDDING_TOKEN_LIMIT: int = 8190
    RERANKER_MODEL_PATH: str = "/media/work/learn/ai/models/reranker/bge-reranker-v2-m3"
    # LLM_MODEL_PATH: str = "/media/work/learn/ai/models/llm/deepseek/deepseek-r1-destill-qwen-1.5B/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"
    LLM_MODEL_PATH: str = "/media/work/learn/ai/models/llm/meta/llama-3.2-3b-Instruct/llama-3.2-3b-Instruct-f16.gguf"
    # LLM_MODEL_PATH: str = "/media/work/learn/ai/models/llm/qwen/qwen2.5/qwen2.5-1.5b-instruct-q8_0.gguf"
    CTX_WINDOW = 4096

    # General models
    TORCH_DEVICE: Optional[Literal["cuda", "cpu"]] = None

    @computed_field
    @property
    def DEVICE(self) -> Literal["cuda", "cpu"]:
        if self.TORCH_DEVICE:
            return self.TORCH_DEVICE

        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    MODELS = [
        {"name": "DeepSeek R1 Destill", "path": "/"},
        {"name": "Qwen2.5 Instruct", "path": "/"},
    ]


settings = Settings()
