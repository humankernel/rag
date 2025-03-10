import logging
from typing import Generator, TypedDict
from uuid import uuid4

import gradio as gr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

from prompt import PROMPT
from rag import RAGPipeline
from settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


client = Llama(
    model_path=settings.LLM_MODEL_PATH,
    device=settings.DEVICE,
    n_ctx=2048,
    top_p=0.95,
    temperature=0.8,
    top_k=50,
    verbose=False,
    # type_k=llama_cpp.GGML_TYPE_Q8_0,
    draft_model=LlamaPromptLookupDecoding(
        num_pred_tokens=10
    ),  # num_pred_tokens is the number of tokens to predict 10 is the default and generally good for gpu, 2 performs better for cpu-only machines
)


def handle_undo(history, undo_data: gr.UndoData):
    return history[: undo_data.index], history[undo_data.index]["content"]


def handle_retry(history, retry_data: gr.RetryData):
    new_history = history[: retry_data.index]
    yield from chat(None, new_history)


def handle_like(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: ", data.value)
    else:
        print("You downvoted this response: ", data.value)


def handle_edit(history, edit_data: gr.EditData):
    new_history = history[: edit_data.index]
    new_history[-1]["content"] = edit_data.value
    return new_history


def clear():
    print("Cleared uuid")
    return uuid4()


def chat_fn(user_input, history, uuid):
    return f"{user_input} with uuid {uuid}"


class Message(TypedDict):
    text: str
    files: list[str]


def chat(
    message: Message,
    history: list[gr.ChatMessage],
    error_prompt: str = PROMPT["error_prompt"],
    temperature: float = 0.25,
    max_tokens: int = 300,
) -> Generator[list[gr.ChatMessage], None, None]:
    msg = message.get("text", "")
    files = message.get("files", [])

    history = history or [{"role": "system", "content": PROMPT["system_prompt"]}]
    history.append({"role": "user", "content": msg})

    if not (msg or files):
        return history
    if files:
        pdfs = list(filter(lambda path: path.endswith(".pdf"), files))
        if len(pdfs) > 0:
            pass
            # rag = RAGPipeline("chat")
            # rag.insert(docs)
            # rag.ask(question)

    assert len(history) > 0, "History should not be empty"
    assert any(isinstance(msg["content"], str) for msg in history), (
        "Message content should be of type str"
    )

    history.append({"role": "assistant", "content": ""})
    try:
        for stream in client.create_chat_completion(
            history,
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
            stream=True,
        ):
            content = stream["choices"][0]["delta"].get("content", "")
            history[-1]["content"] += content
            yield history

    except Exception as e:
        print(f"An error occurred: {e}")
        history.append({"role": "assistant", "content": error_prompt})


with gr.Blocks(fill_height=True, fill_width=True) as demo:
    uuid_state = gr.State(uuid4)

    # chatbot = gr.Chatbot(
    #     type="messages",
    #     render_markdown=True,
    #     editable="user",
    # )

    # chat_input = gr.MultimodalTextbox(
    #     file_count="multiple",
    #     file_types=[".pdf"],
    #     placeholder="Type your message...",
    #     submit_btn=True,
    #     show_label=False,
    # )

    gr.ChatInterface(
        fn=chat,
        multimodal=True,
        type="messages",
        flagging_mode="manual",
        flagging_options=["Like", "Spam", "Inappropriate", "Other"],
        # chatbot=gr.Chatbot(
        #     type="messages",
        #     render_markdown=True,
        #     editable="user",
        # ),
        # textbox=gr.MultimodalTextbox(
        #     file_count="multiple",
        #     file_types=[".pdf"],
        #     placeholder="Type your message...",
        #     submit_btn=True,
        #     show_label=False,
        # ),
        save_history=True,
        additional_inputs_accordion=system_settings,
    )

    # chatbot.undo(handle_undo, chatbot, [chatbot, chat_input])
    # chatbot.retry(handle_retry, chatbot, chatbot)
    # chatbot.like(handle_like, None, None)
    # chatbot.edit(handle_edit, chatbot, chatbot)
    # chatbot.clear(clear, outputs=[uuid_state])

    with gr.Accordion(label="System Settings", open=False) as system_settings:
        error_prompt = gr.Textbox(
            value=PROMPT["error_prompt"], label="Error Prompt", interactive=True
        )
        temperature = gr.Slider(0, 1, value=0.75, label="Temperature")
        tokens = gr.Slider(10, 100, value=500, label="Tokens")


if __name__ == "__main__":
    demo.queue(api_open=False).launch(server_name="0.0.0.0", show_api=False, debug=True)
