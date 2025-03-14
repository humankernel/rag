import time
from functools import cache
from typing import Generator, Literal, Optional, TypedDict

import gradio as gr
from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

from rag.lib.vectordb import RetrievedChunk, VectorDB
from rag.prompt import PROMPT
from rag.settings import settings


class Message(TypedDict):
    text: str
    files: list[str]


CTX_WINDOW = 4096

model = Llama(
    model_path=settings.LLM_MODEL_PATH,
    device=settings.DEVICE,
    main_gpu=0,  # Use primary GPU
    tensor_split=[0],  # Distribute layers across GPUs if multi-GPU
    n_gpu_layers=-1 if settings.DEVICE == "cuda" else 0,
    n_ctx=CTX_WINDOW,
    n_batch=1024,       # Increased batch size for better throughput (adjust based on VRAM)
    n_threads=4 if settings.DEVICE == "cuda" else 8,  # Optimize CPU threading
    offload_kqv=True,   # Enable memory optimization
    flash_attn=True,    # Enable flash attention if supported
    temperature=0.7,    # Added for sampling diversity
    top_p=0.90,         # Slightly more focused than 0.95
    top_k=40,           # Balanced between diversity and quality
    repeat_penalty=1.1, # Helps reduce repetition
    draft_model=LlamaPromptLookupDecoding(
        max_ngram_size=7,      # Optimal for most use cases
        num_pred_tokens=10,
    ),
    seed=42,            # For reproducibility
    verbose=False,
    # Add if your implementation supports these:
    # rope_scaling=1.0,  # For context extension techniques
    # logits_all=True,   # If you need all logits
    # main_gpu=0,        # For multi-GPU setups
)



@cache
def count_tokens(text: str) -> int:
    return len(text) // 4


def compress_history(
    history: list[gr.ChatMessage], max_tokens: int = CTX_WINDOW
) -> list[gr.ChatMessage]:
    """Only keep relevant history messages (using reranking)"""
    relevant_history = []

    # Keep last messages that fit in the ctx_window
    total_tokens = 0
    for h in reversed(history):
        h_tokens = count_tokens(h["content"])
        if h_tokens + total_tokens > max_tokens:
            break
        relevant_history.append(h)
        total_tokens += h_tokens
    relevant_history = list(reversed(relevant_history))

    # Keep relevant using reranking
    # todo

    return relevant_history


def compress_context(chunks: list[RetrievedChunk], max_tokens: int = CTX_WINDOW) -> str:
    relevant_context = ""

    # Format the chunks
    formatted = [f"[Doc {i + 1}] {chunk.chunk.text}" for i, chunk in enumerate(chunks)]
    relevant_context = "\n\n".join(formatted)
    relevant_context = relevant_context[: (max_tokens - 10) * 4]

    # Use LLM to clean the context
    # todo

    return relevant_context


def generate_answer(
    history: list[gr.ChatMessage],
    chunks: list[RetrievedChunk],
    temperature: float = 0.7,
    max_tokens: int = 300,
    mode: Literal["rag", "normal"] = "normal",
) -> Generator[str, None, None]:
    assert history, "History should not be empty"
    assert all(isinstance(msg["content"], str) for msg in history), (
        'h["content"] should be str'
    )

    query = history[-1]["content"]
    tokens_hst = sum(count_tokens(h["content"]) for h in history)
    relevant_history = compress_history(history)

    if mode == "normal" or not chunks:
        assert tokens_hst < CTX_WINDOW, "History exceeds the ctx window"
        response = model.create_chat_completion(
            relevant_history,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
    elif mode == "rag":
        relevant_context = compress_context(chunks)

        prompt = PROMPT["generate_answer"].format(query=query, context=relevant_context)
        tokens_pmt = count_tokens(prompt)

        # prioritize the context over the chat history
        if tokens_hst + tokens_pmt > CTX_WINDOW:
            relevant_history = []
            tokens_hst = 0
        if tokens_pmt > CTX_WINDOW:
            # todo: define what to do
            pass

        assert tokens_hst + tokens_pmt < CTX_WINDOW, (
            "History + Prompt exceeds the ctx window"
        )

        # messages = [*relevant_history, {"role": "user", "content": prompt}]
        messages = [
            {
                "role": "system",
                "content": f"The user has a question: {query} \n\n This is some context that may help answering the question: {'relevant_context'} \n\n Answer: ",
            },
        ]
        response = model.create_chat_completion(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

    try:
        for stream in response:
            text = stream["choices"][0]["delta"].get("content", "")
            yield text
    except Exception as e:
        print(f"Error: {e}")


def ask(
    message: Message,
    history: list[gr.ChatMessage],
    db: VectorDB,
    temperature: Optional[float] = 0.75,
    max_tokens: Optional[int] = 300,
):
    msg = message.get("text", "") if isinstance(message, dict) else message
    files = message.get("files", []) if isinstance(message, dict) else []
    response: list[gr.ChatMessage] = [{"role": "assistant", "content": ""}]
    start_time = time.time()

    history = history or [{"role": "system", "content": "system prompt"}]
    history.append({"role": "user", "content": msg})
    # Change files from history['content']
    for h in history:
        if isinstance(h["content"], tuple):
            if len(h["content"]) > 1:
                content: str = h["content"][1]
            else:
                content = ""
            h["content"] = content

    # Indexing
    if files:
        if any(not f.endswith(".pdf") for f in files):
            gr.Warning("The system only supports pdf files currently.")
            return

        pdfs = list(filter(lambda f: f.endswith(".pdf"), files))
        response[-1] = {
            "content": f"📥 Received {len(pdfs)} PDF(s). Starting indexing...",
            "metadata": {
                "title": "🔎 Indexing...",
                "duration": time.time() - start_time,
                "status": "pending",
            },
        }
        yield response, []
        db.insert(pdfs)
        response[-1] = {
            "content": f"Indexed {len(pdfs)} files",
            "metadata": {
                "title": "📥 Finished Indexing",
                "duration": time.time() - start_time,
                "status": "done",
            },
        }
        yield response, []

    # todo: explore tool_use to call the rag instead of the other way around

    # Retrieval
    chunks = []
    if msg and not db.is_empty():
        response[-1] = {
            "content": "Searching relevant sources",
            "metadata": {
                "title": "🔎 Searching...",
                "duration": time.time() - start_time,
                "status": "pending",
            },
        }
        yield response, chunks
        chunks = db.search(msg)
        response[-1] = {
            "content": f"Retrieved {len(chunks)} relevant sources",
            "metadata": {
                "title": f"🔎 Found {len(chunks)} relevant chunks",
                "duration": time.time() - start_time,
                "status": "done",
            },
        }
        yield response, chunks

    # Generation
    response[-1] = {"role": "assistant", "content": ""}
    answer_buffer = ""
    try:
        for stream in generate_answer(
            history,
            chunks,
            temperature,
            max_tokens,
            mode="normal" if db.is_empty() or not msg else "rag",
        ):
            answer_buffer += stream
            response[-1]["content"] = answer_buffer
            yield response, chunks
    except Exception as e:
        print(f"Error {e}")


def main() -> None:
    with gr.Blocks() as demo:
        print(f"{demo.app_id=}")
        db = gr.State(VectorDB("att"))
        local_storage = gr.BrowserState()
        chunks = gr.State([])

        with gr.Sidebar(position="left", open=False):
            gr.Markdown("# Model Settings")
            # todo: wire up the additional params

            with gr.Group():
                model = gr.Dropdown(
                    choices=["DeepSeek R1 Destill", "Qwen2.5 Instruct"],
                    value="DeepSeek R1 Destill",
                    label="Model Selection",
                )
            temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Temperature",
                info="Lower = deterministic, Higher = creative",
            )
            max_tokens = gr.Slider(
                minimum=1,
                maximum=4000,
                value=512,
                step=10,
                label="Max Tokens",
                info="Maximum response length",
            )
            top_p = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.9,
                step=0.05,
                label="Top-p (Nucleus Sampling)",
            )
            frequency_penalty = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.5,
                step=0.1,
                label="Frequency Penalty",
                info="Reduce repetition of common phrases",
            )
            presence_penalty = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.5,
                step=0.1,
                label="Presence Penalty",
                info="Encourage new topics",
            )
            top_k = gr.Slider(
                minimum=1,
                maximum=100,
                value=50,
                step=1,
                label="Top-k Sampling",
                info="Consider top K tokens",
            )
            repetition_penalty = gr.Slider(
                minimum=1.0,
                maximum=2.0,
                value=1.2,
                step=0.1,
                label="Repetition Penalty",
                info="Higher = less repetition",
            )
            system_prompt = gr.Textbox(
                value=PROMPT["system_prompt"],
                lines=3,
                label="System Prompt",
                placeholder="Enter system instructions...",
            )
            language_strictness = gr.Radio(
                choices=["Professional", "Informal"],
                value="Professional",
                label="Language Style",
            )
            saved_message = gr.Markdown("")

        with gr.Sidebar(position="right", open=False):
            gr.Markdown("# References")

            @gr.render(inputs=chunks)
            def render_chunks(chunks: list[RetrievedChunk]):
                for chunk in chunks:
                    with gr.Accordion(
                        label=f"id: {chunk.chunk.id}, doc_id: {chunk.chunk.doc_id}",
                        open=False,
                    ):
                        gr.Markdown(
                            f"scores: (dense:{chunk.scores['dense_score']:.2f}, sparse:{chunk.scores['sparse_score']:.2f}, hybrid:{chunk.scores['hybrid_score']:.2f}, rerank:{chunk.scores['rerank_score']:.2f}) \n\n"
                            f"{chunk.chunk.text}",
                        )

        gr.ChatInterface(
            fn=ask,
            type="messages",
            multimodal=True,
            editable="user",
            flagging_mode="manual",
            flagging_options=[
                "Like",
                "Dislike",
                "Hallucination",
                "Inappropriate",
                "Harmful",
            ],
            flagging_dir=".",
            additional_inputs=[db, temperature, max_tokens],
            additional_outputs=[chunks],
            save_history=True,
        )

        @demo.load(inputs=[local_storage], outputs=[temperature, max_tokens])
        def load_from_local_storage(saved_values):
            saved_values = saved_values or {"temperature": 0.75, "max_tokens": 300}
            return saved_values["temperature"], saved_values["max_tokens"]

        @gr.on(
            [temperature.change, max_tokens.change],
            inputs=[temperature, max_tokens],
            outputs=[local_storage],
        )
        def save_to_local_storage(temp, max_tks):
            return {"temperature": temp, "max_tokens": max_tks}

        @gr.on(local_storage.change, outputs=[saved_message])
        def show_saved_message():
            timestamp = time.strftime("%I:%M:%S %p")
            return gr.Markdown(
                f"✅ Saved to local storage at {timestamp}", visible=True
            )

    demo.queue(api_open=False).launch(inbrowser=True, max_file_size="300mb")


if __name__ == "__main__":
    main()
