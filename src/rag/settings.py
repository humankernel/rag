import os
from typing import Literal, Optional

import torch
from dotenv import load_dotenv
from gradio_client.utils import TypedDict
from pydantic import BaseModel, computed_field

load_dotenv()


class Model(TypedDict):
    name: str
    path: Optional[str]


class Settings(BaseModel):
    """Configuration for the app"""

    # Paths
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_TOKEN_LIMIT: int = 8190
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
    CTX_WINDOW: int = 8192

    # General models
    TORCH_DEVICE: Optional[Literal["cuda", "cpu"]] = None

    # List of additional models with names and optional paths
    MODELS: list[Model] = [
        {"name": "DeepSeek R1 Destill", "path": os.getenv("DEEPSEEK_PATH")},
        {"name": "Qwen2.5 Instruct", "path": os.getenv("QWEN_PATH")},
    ]

    @computed_field
    @property
    def DEVICE(self) -> Literal["cuda", "cpu"]:
        if self.TORCH_DEVICE:
            return self.TORCH_DEVICE
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


settings = Settings()
