from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
from rag.settings import settings


def create_model(model_path: str) -> Llama:
    return Llama(
        model_path=model_path,
        device=settings.DEVICE,
        main_gpu=0,  # Use primary GPU
        tensor_split=[0],  # Distribute layers across GPUs if multi-GPU
        n_gpu_layers=-1 if settings.DEVICE == "cuda" else 0,
        n_ctx=settings.CTX_WINDOW,
        n_batch=1024,  # Increased batch size for better throughput (adjust based on VRAM)
        n_threads=4 if settings.DEVICE == "cuda" else 8,  # Optimize CPU threading
        offload_kqv=True,  # Enable memory optimization
        flash_attn=True,  # Enable flash attention if supported
        draft_model=LlamaPromptLookupDecoding(
            max_ngram_size=7,  # Optimal for most use cases
            num_pred_tokens=10,
        ),
    )
