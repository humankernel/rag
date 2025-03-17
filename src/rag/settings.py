import os
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, computed_field
from typing_extensions import Literal, Optional, TypedDict

load_dotenv()


class Model(TypedDict):
    name: str
    path: Optional[str]


class Settings(BaseModel):
    """Configuration for the app"""

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_TOKEN_LIMIT: int = 8190
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "./DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf")
    CTX_WINDOW: int = 8192

    # List of additional models with names and optional paths
    MODELS: list[Model] = [
        {"name": "DeepSeek-R1-Distill-Qwen-1.5B-GGUF", "path": os.getenv("DEEPSEEK_PATH")},
        {"name": "Qwen2.5-Instruct", "path": os.getenv("QWEN_PATH")},
    ]

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


settings = Settings()
