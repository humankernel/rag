from functools import cache
import time
from typing import Generator, Optional, TypedDict, Literal

import gradio as gr
from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

from settings import settings
from prompt import PROMPT
from lib.vectordb import RetrievedChunk, VectorDB


class Message(TypedDict):
    text: str
    files: list[str]


CTX_WINDOW = 4096

model = Llama(
    model_path=settings.LLM_MODEL_PATH,
    device=settings.DEVICE,
    n_ctx=CTX_WINDOW,
    top_p=0.95,
    top_k=50,
    verbose=False,
    draft_model=LlamaPromptLookupDecoding(10),
)


@cache
def count_tokens(text: str) -> int:
    return len(text) // 4


def compress_history(
    history: list[gr.Chatbot], max_tokens: int = CTX_WINDOW
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
    formatted = map(lambda chunk: f"Doc {chunk.chunk.text}", chunks)
    relevant_context = "\n".join(formatted)
    relevant_context[: max_tokens * 4]

    # Use LLM to clean the context
    # todo

    return relevant_context


def generate_answer(
    history: list[gr.Chatbot],
    chunks: list[RetrievedChunk],
    temperature: float = 0.7,
    max_tokens: int = 300,
    mode: Literal["rag", "normal"] = "normal",
) -> Generator[str, None, None]:
    assert history, "History should not be empty"
    assert any(isinstance(msg["content"], str) for msg in history), (
        "Messages should be str"
    )
    # explore tool_use to call the rag instead of the other way around
    # make sure the chat history does not exceed the ctx_window
    tokens_hst = sum(count_tokens(h["content"]) for h in history)
    relevant_history = history
    if tokens_hst > CTX_WINDOW:
        relevant_history = compress_history(history)

    if mode == "normal" or not chunks:
        assert tokens_hst < CTX_WINDOW, "History exceeds the ctx window"
        response = model.create_chat_completion(
            relevant_history,
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
            stream=True,
        )
    elif mode == "rag":
        relevant_history = (
            compress_history(history)
            if len(relevant_history) != len(history)
            else relevant_history
        )
        relevant_context = compress_context(chunks)
        query = history[-1]["content"]

        prompt = PROMPT["generate_answer"].format(query=query, context=relevant_context)
        tokens_pmt = count_tokens(prompt)

        # prioritize the context over the chat history
        if tokens_hst + tokens_pmt > CTX_WINDOW:
            relevant_history = []
        if tokens_pmt > CTX_WINDOW:
            # todo: define what to do
            pass

        assert tokens_hst + tokens_pmt < CTX_WINDOW, (
            "History + Prompt exceeds the ctx window"
        )

        response = model.create_chat_completion(
            [*relevant_history, {"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
            stream=True,
        )

    for stream in response:
        yield stream["choices"][0]["delta"].get("content", "")


def ask(
    message: Message,
    history: list[gr.ChatMessage],
    db: VectorDB,
    temperature: Optional[float] = 0.75,
    max_tokens: Optional[int] = 300,
):
    msg = message.get("text", "") if isinstance(message, dict) else message
    files = message.get("files", []) if isinstance(message, dict) else []
    history = history or [{"role": "system", "content": "system prompt"}]
    history.append({"role": "user", "content": msg})
    response: list[gr.ChatMessage] = [{"role": "assistant", "content": ""}]
    start_time = time.time()

    # Indexing
    if files:
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

    # Retrieval
    chunks = []
    if not db.is_empty():
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
            mode="normal" if db.is_empty() else "rag",
        ):
            answer_buffer += stream
            response[-1]["content"] = answer_buffer
            yield response, chunks
    except Exception as e:
        print(f"Error {e}")


# n_ctx
# top_p
# top_k

with gr.Blocks(fill_width=True, fill_height=True) as demo:
    print(f"{demo.app_id=}")
    db = gr.State(VectorDB("session"))
    local_storage = gr.BrowserState()
    chunks = gr.State([])

    with gr.Sidebar(position="left", open=False):
        gr.Markdown("# Model Params")
        temperature = gr.Slider(
            0,
            1,
            0.75,
            0.5,
            label="Temperature",
            interactive=True,
            show_reset_button=True,
        )
        max_tokens = gr.Number(value=300, step=50, label="Max Tokens")
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

    chat = gr.ChatInterface(
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
            "Other",
        ],
        flagging_dir=".",
        additional_inputs=[db, temperature, max_tokens],
        additional_outputs=[chunks],
        save_history=True,
    )

    @demo.load(inputs=[local_storage], outputs=[temperature, max_tokens])
    def load_from_local_storage(saved_values):
        saved_values = saved_values or {"temperature": 0.75, "max_tokens": 300}
        print("loading from local storage", saved_values)
        return saved_values["temperature"], saved_values["max_tokens"]

    @gr.on(
        [temperature.change, max_tokens.change],
        inputs=[temperature, max_tokens],
        outputs=[local_storage],
    )
    def save_to_local_storage(temperature, max_tokens):
        return {"temperature": temperature, "max_tokens": max_tokens}

    @gr.on(local_storage.change, outputs=[saved_message])
    def show_saved_message():
        timestamp = time.strftime("%I:%M:%S %p")
        return gr.Markdown(f"✅ Saved to local storage at {timestamp}", visible=True)

    # chat.chatbot_value

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, max_file_size="300mb")
