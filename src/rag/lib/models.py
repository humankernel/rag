from functools import cache
from threading import Lock
from typing import Generator

from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

from rag.settings import settings


class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    """

    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class LLMModel(metaclass=SingletonMeta):
    _model: Llama | None = None
    _lock: Lock = Lock()

    def __init__(self, model_path: str = settings.LLM_MODEL):
        self._model = Llama(
            model_path=model_path,
            device=settings.DEVICE,
            n_gpu_layers=-1 if settings.DEVICE == "cuda" else 0,
            n_ctx=settings.CTX_WINDOW,
            n_batch=512,  # Increased batch size for better throughput (adjust based on VRAM)
            n_threads=4 if settings.DEVICE == "cuda" else 8,  # Optimize CPU threading
            offload_kqv=True,  # Enable memory optimization
            flash_attn=True,  # Enable flash attention if supported
            # draft_model=LlamaPromptLookupDecoding(
            #     max_ngram_size=7,  # Optimal for most use cases
            #     num_pred_tokens=10,
            # ),
        )

    @cache
    def count_tokens(self, text: str) -> int:
        assert self._model is not None, "Model is not initialized"
        return len(self._model.tokenize(text.encode("utf-8"), add_bos=False))

    def generate_response_stream(
        self, messages, **kwargs
    ) -> Generator[str, None, None]:
        assert self._model is not None, "Model is not initialized"
        assert all(isinstance(msg["content"], str) for msg in messages), (
            "content should be str"
        )
        assert (
            sum(self.count_tokens(msg["content"]) for msg in messages)
            <= settings.CTX_WINDOW
        ), "Input exceeds context window"

        with self._lock:
            for stream in self._model.create_chat_completion(
                messages, stream=True, **kwargs
            ):
                yield stream["choices"][0]["delta"].get("content", "")

    def generate_response(self, messages, **kwargs) -> str:
        assert self._model is not None, "Model is not initialized"
        assert all(isinstance(msg["content"], str) for msg in messages), (
            "content should be str"
        )

        with self._lock:
            response = self._model.create_chat_completion(messages, **kwargs)
            return response["choices"][0]["message"].get("content", "")
