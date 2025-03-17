import time
from functools import cache
from typing_extensions import Generator, Literal, NotRequired, Optional, TypedDict

import gradio as gr

from rag.lib.models import LLMModel
from rag.lib.vectordb import RetrievedChunk, VectorDB
from rag.prompt import PROMPT
from rag.settings import settings


class Message(TypedDict):
    text: str
    files: list[str]


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str
    metadata: NotRequired[Optional[dict]]


model = LLMModel()


@cache
def count_tokens(text: str) -> int:
    # TODO: Replace with model's actual token count for accuracy
    return len(text) // 4


def compress_history(
    history: list[ChatMessage], max_tokens: int = settings.CTX_WINDOW
) -> list[ChatMessage]:
    # TODO: Enhance to keep only relevant previous messages
    relevant_history = []
    total_tokens = 0
    for h in reversed(history):
        h_tokens = count_tokens(h["content"])
        if h_tokens + total_tokens > max_tokens:
            break
        relevant_history.append(h)
        total_tokens += h_tokens
    return list(reversed(relevant_history))

    # Keep relevant using reranking
    # todo

    return relevant_history


def compress_context(
    chunks: list[RetrievedChunk], max_tokens: int = settings.CTX_WINDOW
) -> str:
    # TODO: Use LLM to refine context instead of simple truncation
    formatted = [f"[Doc {i + 1}] {chunk.chunk.text}" for i, chunk in enumerate(chunks)]
    context = "\n\n".join(formatted)
    return context[: max_tokens * 4]
    # todo: instead use LLM to clean the context


def generate_answer(
    history: list[ChatMessage],
    chunks: list[RetrievedChunk],
    mode: Literal["rag", "normal"] = "normal",
    system_prompt: str = PROMPT["system_prompt"],
    language_strictness: Literal["Professional", "Informal"] = "Professional",
    temperature=None,
    max_tokens=None,
    top_p=None,
    top_k=None,
    frequency_penalty=None,
    presence_penalty=None,
) -> Generator[str, None, None]:
    assert history, "History should not be empty"
    assert all(isinstance(msg["content"], str) for msg in history), (
        "History content must be strings"
    )

    query = history[-1]["content"]
    relevant_history = compress_history(history)
    tokens_relevant_hst = sum(count_tokens(h["content"]) for h in relevant_history)

    if mode == "normal" or not chunks:
        # Fix: Check compressed history tokens, not original
        assert tokens_relevant_hst < settings.CTX_WINDOW, (
            "Compressed history exceeds context window"
        )
        messages = relevant_history

    elif mode == "rag":
        relevant_context = compress_context(chunks)
        messages: list[ChatMessage] = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context.",
            },
            {
                "role": "user",
                "content": f"Context: {relevant_context}\n\nQuestion: {query}",
            },
        ]
        total_tokens = sum(count_tokens(msg["content"]) for msg in messages)
        assert total_tokens < settings.CTX_WINDOW, "RAG messages exceed context window"

    try:
        for response in model.generate_response_stream(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        ):
            yield response
    except Exception as e:
        yield f"Error: {e}"


def ask(
    message: Message,
    history: list[ChatMessage],
    db: VectorDB,
    system_prompt: str = PROMPT["system_prompt"],
    language_strictness: Literal["Professional", "Informal"] = "Professional",
    temperature=None,
    max_tokens=None,
    top_p=None,
    top_k=None,
    frequency_penalty=None,
    presence_penalty=None,
) -> Generator[tuple[ChatMessage, list[RetrievedChunk]], None, None]:
    start_time = time.time()
    msg = message.get("text", "") if isinstance(message, dict) else message
    files = message.get("files", []) if isinstance(message, dict) else []
    chunks: list[RetrievedChunk] = []

    # Fix: Handle empty input
    if not msg and not files:
        yield {"role": "assistant", "content": "Please provide a message or files."}, []
        return

    # Ensure system prompt is included as a system message
    if not history:
        history = [{"role": "system", "content": system_prompt}]
    for h in history:
        if isinstance(h["content"], tuple):
            h["content"] = h["content"][1] if len(h["content"]) > 1 else ""

    # Indexing files
    if files:
        if any(not f.endswith(".pdf") for f in files):
            gr.Warning("The system only supports PDF files currently.")
        else:
            pdfs = [f for f in files if f.endswith(".pdf")]
            response = {
                "role": "assistant",
                "content": f"📥 Received {len(pdfs)} PDF(s). Starting indexing...",
                "metadata": {"title": "🔎 Indexing...", "status": "pending"},
            }
            yield response, chunks
            db.insert(pdfs)
            response = {
                "role": "assistant",
                "content": f"Indexed {len(pdfs)} files",
                "metadata": {
                    "title": "📥 Finished Indexing",
                    "duration": time.time() - start_time,
                    "status": "done",
                },
            }
            yield response, chunks

    # Retrieval and generation only if there's a message
    if msg:
        if not db.is_empty():
            response = {
                "role": "assistant",
                "content": "Searching relevant sources",
                "metadata": {
                    "title": "🔎 Searching...",
                    "duration": time.time() - start_time,
                    "status": "pending",
                },
            }
            yield response, chunks
            chunks = db.search(msg)
            response = {
                "role": "assistant",
                "content": f"Retrieved {len(chunks)} relevant sources",
                "metadata": {
                    "title": f"🔎 Found {len(chunks)} relevant chunks",
                    "duration": time.time() - start_time,
                    "status": "done",
                },
            }
            yield response, chunks

        # Generate response
        response = {"role": "assistant", "content": ""}
        answer_buffer = ""
        try:
            for stream in generate_answer(
                history,
                chunks,
                mode="rag" if chunks else "normal",
                system_prompt=system_prompt,
                language_strictness=language_strictness,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            ):
                answer_buffer += stream
                response["content"] = answer_buffer
                yield response, chunks
        except Exception as e:
            print(f"Error {e}")


def main() -> None:
    with gr.Blocks(fill_height=True) as demo:
        db = gr.State(VectorDB("att"))
        local_storage = gr.BrowserState()
        chunks = gr.State([])

        with gr.Sidebar(position="left", open=False):
            gr.Markdown("# Model Settings")
            model_name = gr.Dropdown(
                choices=[m["name"] for m in settings.MODELS],
                value=settings.MODELS[0]["name"],
            )
            temperature = gr.Slider(
                0.0,
                2.0,
                value=1.0,
                step=0.1,
                label="Temperature",
                info="Controls randomness: Lower = more deterministic, Higher = more creative.",
            )
            top_p = gr.Slider(
                0.0,
                1.0,
                value=0.9,
                step=0.05,
                label="Top-p",
                info="Considers the smallest set of tokens whose cumulative probability exceeds p.",
            )
            top_k = gr.Slider(
                1,
                100,
                value=50,
                step=1,
                label="Top-k Sampling",
                info="Limits sampling to the top K most likely tokens.",
            )
            max_tokens = gr.Slider(
                1,
                4000,
                value=512,
                step=10,
                label="Max Tokens",
                info="Sets the maximum number of tokens in the response.",
            )
            frequency_penalty = gr.Slider(
                0.0,
                2.0,
                value=0.5,
                step=0.1,
                label="Frequency Penalty",
                info="Reduces the likelihood of repeating common phrases.",
            )
            presence_penalty = gr.Slider(
                0.0,
                2.0,
                value=0.5,
                step=0.1,
                label="Presence Penalty",
                info="Encourages new topics.",
            )
            system_prompt = gr.Textbox(
                value=PROMPT["system_prompt"],
                lines=3,
                label="System Prompt",
                placeholder="Enter instructions for the assistant's behavior.",
                info="Defines the assistant's role or specific instructions.",
            )
            language_strictness = gr.Radio(
                choices=["Professional", "Informal"],
                value="Professional",
                label="Language Style",
                info="Select the tone of the assistant's responses.",
            )

            saved_message = gr.Markdown("", visible=False)

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
            editable=True,  # TODO: Restrict to "user" messages
            flagging_mode="manual",
            flagging_options=[
                "Like",
                "Dislike",
                "Hallucination",
                "Inappropriate",
                "Harmful",
            ],
            flagging_dir=".",
            additional_inputs=[
                db,
                system_prompt,
                language_strictness,
                temperature,
                max_tokens,
                top_p,
                top_k,
                frequency_penalty,
                presence_penalty,
            ],
            additional_outputs=[chunks],
            save_history=True,
        )

        @demo.load(
            inputs=[local_storage],
            outputs=[
                model_name,
                temperature,
                max_tokens,
                top_p,
                top_k,
                frequency_penalty,
                presence_penalty,
                system_prompt,
                language_strictness,
            ],
        )
        def load_from_local_storage(saved_values):
            defaults = {
                "model": settings.MODELS[0]["name"],
                "temperature": 0.75,
                "max_tokens": 300,
                "top_p": 0.9,
                "top_k": 50,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.5,
                "system_prompt": PROMPT["system_prompt"],
                "language_strictness": "Professional",
            }
            saved_values = saved_values or defaults
            return (
                saved_values["model"],
                saved_values["temperature"],
                saved_values["max_tokens"],
                saved_values["top_p"],
                saved_values["top_k"],
                saved_values["frequency_penalty"],
                saved_values["presence_penalty"],
                saved_values["system_prompt"],
                saved_values["language_strictness"],
            )

        @gr.on(
            [
                model_name.change,
                temperature.change,
                max_tokens.change,
                top_p.change,
                top_k.change,
                frequency_penalty.change,
                presence_penalty.change,
                system_prompt.change,
                language_strictness.change,
            ],
            inputs=[
                model_name,
                temperature,
                max_tokens,
                top_p,
                top_k,
                frequency_penalty,
                presence_penalty,
                system_prompt,
                language_strictness,
            ],
            outputs=[local_storage],
        )
        def save_to_local_storage(
            model,
            temp,
            max_tks,
            top_p,
            top_k,
            freq_pen,
            pres_pen,
            sys_prompt,
            lang_strict,
        ):
            return {
                "model": model,
                "temperature": temp,
                "max_tokens": max_tks,
                "top_p": top_p,
                "top_k": top_k,
                "frequency_penalty": freq_pen,
                "presence_penalty": pres_pen,
                "system_prompt": sys_prompt,
                "language_strictness": lang_strict,
            }

        @gr.on(local_storage.change, outputs=[saved_message])
        def show_saved_message():
            timestamp = time.strftime("%I:%M:%S %p")
            return gr.Markdown(
                f"✅ Saved to local storage at {timestamp}", visible=True
            )

    demo.queue(api_open=False).launch(inbrowser=True, max_file_size="300mb", share=True)


if __name__ == "__main__":
    main()
