import logging
from contextlib import contextmanager
import os
from typing import Generator

import torch
from FlagEmbedding import BGEM3FlagModel, FlagReranker
import llama_cpp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextmanager
def embedding_model_context(
    path: str, device: str
) -> Generator[BGEM3FlagModel, None, None]:
    """Load/unload models automatically"""
    model = None
    try:
        logger.info("Loading embedding model...")
        model = BGEM3FlagModel(path, device=device, use_fp16=True)
        yield model
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise
    finally:
        logger.info("Unloading embedding model...")
        if model is not None:
            del model
        torch.cuda.empty_cache()


@contextmanager
def reranker_context(path: str, device: str) -> Generator[FlagReranker, None, None]:
    """Load/unload reranker model automatically"""
    reranker = None
    try:
        logger.info("Loading reranker model...")
        reranker = FlagReranker(path, device=device, use_fp16=True)
        yield reranker
    except Exception as e:
        logger.error(f"Failed to load reranker model: {e}")
        raise
    finally:
        logger.info("Unloading reranker model...")
        if reranker is not None:
            del reranker
        torch.cuda.empty_cache()


@contextmanager
def llm_context(path: str, device: str) -> Generator[llama_cpp.Llama, None, None]:
    "Load/unload llm model automatically"
    assert os.path.exists(path), "Path should exists"

    model = None
    try:
        logger.info("Loading llm model...")
        model = llama_cpp.Llama(path, device=device, n_ctx=2048, verbose=False)
        yield model
    except Exception as e:
        logger.error(f"Failed to load llm model: {e}")
        raise
    finally:
        logger.info("Unloading llm model...")
        if model is not None:
            model.close()
            del model
        torch.cuda.empty_cache()
