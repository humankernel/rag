from functools import cache
import tiktoken
import json
from pathlib import Path
import wikipediaapi as wiki
from datetime import datetime
from typing import TypedDict
import gradio as gr

# Initialize Wikipedia APIs
wiki_en = wiki.Wikipedia(
    user_agent="RAG-Evaluator (contact@example.com)",
    language="en",
    extract_format=wiki.ExtractFormat.WIKI,
)

wiki_es = wiki.Wikipedia(
    user_agent="RAG-Evaluator (contact@example.com)",
    language="es",
    extract_format=wiki.ExtractFormat.WIKI,
)


enc = tiktoken.get_encoding("cl100k_base")


class Chunk(TypedDict):
    level: int
    title: str
    text: str
    tokens: int


class WikipediaArticle(TypedDict):
    title: str
    chunks: list[Chunk]
    categories: list[str]


class FetchedWikipediaArticle(TypedDict):
    url: str
    en: WikipediaArticle
    es: WikipediaArticle


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def split_content(text: str, max_tokens: int) -> list[str]:
    tokens = enc.encode(text)
    return [
        enc.decode(tokens[i : i + max_tokens])
        for i in range(0, len(tokens), max_tokens)
    ]


def process_sections(
    sections: list[wiki.WikipediaPageSection], level=0, max_tokens=2500
) -> list[Chunk]:
    if not sections:
        return []

    chunks: list[Chunk] = []
    for section in sections:
        if not section.text.strip():
            continue

        token_count = count_tokens(section.text)
        if token_count > max_tokens:
            # Split long sections into smaller chunks
            split_texts = split_content(section.text, max_tokens)
            for i, text in enumerate(split_texts):
                chunks.append(
                    {
                        "level": level,
                        "title": section.title,
                        "text": text,
                        "tokens": count_tokens(text),
                    }
                )
        else:
            chunks.append(
                {
                    "level": level,
                    "title": section.title,
                    "text": section.text,
                    "tokens": token_count,
                }
            )
        # Process subsections recursively
        chunks.extend(process_sections(section.sections, level + 1, max_tokens))
    return chunks


@cache
def fetch_wikipedia_content(title: str) -> FetchedWikipediaArticle:
    en_page = wiki_en.page(title)
    es_page = en_page.langlinks["es"]
    return {
        "url": title,
        "en": {
            "title": en_page.title,
            "chunks": process_sections(en_page.sections),
            "categories": list(en_page.categories.keys()),
        },
        "es": {
            "title": es_page.title,
            "chunks": process_sections(es_page.sections),
            "categories": list(es_page.categories.keys()),
        },
    }


def generate_section_html(chunks: list[Chunk]) -> str:
    html = []
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEEAD"]

    for i, chunk in enumerate(chunks):
        color = colors[i % len(colors)]
        header = f'<h{chunk["level"] + 2} style="margin-top: 10px;">{chunk["title"]} ({chunk["tokens"]} tokens)</h{chunk["level"] + 2}>'
        content = f'<div style="color: {color}; white-space: pre-wrap; margin-left: {chunk["level"] * 20}px;">{chunk["text"]}</div>'
        html.append(f'<div class="accordion">{header}{content}</div>')

    return '<div class="vertical_accordion">' + "".join(html) + "</div>"


def add_to_dataset(
    url: str,
    category: str,
    dataset_path: str,
    current_data: FetchedWikipediaArticle,
    update=False,
):
    dataset_path = Path(dataset_path)

    if dataset_path.is_dir():
        dataset_path = dataset_path / "dataset.json"
    elif not dataset_path.suffix:
        dataset_path = dataset_path.with_suffix(".json")

    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "url": url,
        "category": category,
        "en_title": current_data["en"]["title"],
        "en_chunks": current_data["en"]["chunks"],
        "es_title": current_data["es"]["title"],
        "es_chunks": current_data["es"]["chunks"],
        "timestamp": datetime.now().isoformat(),
    }

    try:
        # Handle empty file case
        if dataset_path.exists() and dataset_path.stat().st_size > 0:
            try:
                with open(dataset_path, "r") as f:
                    dataset = json.load(f)
            except json.JSONDecodeError:
                # Handle invalid JSON
                dataset = []
        else:
            dataset = []

        existing = next((i for i, e in enumerate(dataset) if e["url"] == url), None)

        if existing is not None:
            if update:
                dataset[existing] = entry
                message = "Entry updated successfully."
            else:
                return "Entry exists. Not updated.", str(dataset_path)
        else:
            dataset.append(entry)
            message = "Entry added successfully."

        # Ensure valid JSON structure
        with open(dataset_path, "w") as f:
            if not dataset:  # Handle empty dataset case
                f.write("[]")
            else:
                json.dump(dataset, f, indent=2)

        return message, str(dataset_path)

    except Exception as e:
        return f"Error: {str(e)}", str(dataset_path)


def handle_fetch(url: str):
    try:
        data = fetch_wikipedia_content(url)
        return (
            [data["en"]["chunks"], data["es"]["chunks"]],
            gr.update(
                choices=data["en"]["categories"],
                value=data["en"]["categories"][0] if data["en"]["categories"] else "",
            ),
            data,
        )

    except Exception as e:
        error_msg = str(e)
        if "Spanish version" in error_msg:
            error_msg += (
                "\n\nNote: Some technical articles may not have Spanish versions."
            )
        raise gr.Error(f"Error fetching article:\n{error_msg}")


with gr.Blocks() as demo:
    data = gr.State([])

    gr.Markdown("# WikiQA: Dataset Generator")

    with gr.Tab("Dataset Creation"):
        topic = gr.Textbox(
            label="Topic Title",
            placeholder="Prime Number",
            submit_btn=True,
        )

        @gr.render(inputs=data)
        def content_chunks(data):
            if not data:
                return

            en: Chunk = data[0]
            es: Chunk = data[1]

            with gr.Accordion("Article Content (Chunked)", open=False):
                with gr.Row():
                    for lang in (en, es):
                        with gr.Column():
                            for chunk in lang:
                                level = chunk.get("level", "")
                                title = chunk.get("title", "")
                                tokens = chunk.get("tokens", "")
                                text = chunk.get("text", "")
                                with gr.Accordion(
                                    open=False,
                                    label=f"level: {level} - {title} - ({tokens} tokens)",
                                ):
                                    gr.TextArea(
                                        value=text, label=None, interactive=True
                                    )

        category = gr.Dropdown(
            label="Article Category", allow_custom_value=True, interactive=True
        )

        json = gr.JSON(label="JSON Output")

        topic.submit(
            handle_fetch,
            inputs=[topic],
            outputs=[data, category, json],
        )

    with gr.Tab("QA Creation"):
        current_chunks = gr.State()

        dataset_file = gr.File(
            label="Dataset File",
            file_count="single",
            file_types=[".json"],
            type="filepath",
        )

        document_title = gr.Dropdown(label="Document Title", allow_custom_value=True)

        with gr.Accordion("Chunks", open=False):
            with gr.Row():
                document_chunks = gr.Dataframe(
                    headers=["level", "title", "text", "tokens"],
                    datatype=["int", "str", "str", "int"],
                )

        gr.Markdown("Generate QA")

        selected_chunks = gr.Textbox(
            label="Select Chunk Indices (comma-separated)", placeholder="0,2,5"
        )
        generate_btn = gr.Button("Generate QA", variant="primary")

        qa_output = gr.JSON(label="Generated QA Pair")
        qa_download = gr.File(label="Download QA Pair")

        def read_dataset(file_path: str):
            try:
                with open(file_path, "r") as f:
                    dataset = json.load(f)

                choices = [doc["en"]["title"] for doc in dataset]
                return {
                    data: dataset,
                    document_title: gr.update(
                        choices=choices, value=choices[0] if choices else ""
                    ),
                }
            except Exception as e:
                raise gr.Error(f"Error reading dataset: {str(e)}")

        def select_document(
            document_title: str, current_data: list[FetchedWikipediaArticle]
        ):
            try:
                document = next(
                    (
                        doc
                        for doc in current_data
                        if doc["en"]["title"] == document_title
                    ),
                    None,
                )
                if not document:
                    raise ValueError("Document not found in dataset")

                en_chunks = document["en"]["chunks"]
                chunk_data = [
                    [chunk["level"], chunk["title"], chunk["text"], chunk["tokens"]]
                    for chunk in en_chunks
                ]
                return {
                    document_chunks: chunk_data,
                    current_chunks: en_chunks,
                }

            except Exception as e:
                raise gr.Error(f"Error selecting document: {str(e)}")

        LLM_PATH = "/media/work/learn/ai/models/llm/deepseek/deepseek-r1-destill-qwen-1.5B/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"

        def generate_qa(indices_str: str, chunks: list[Chunk]):
            try:
                # Validate and parse indices
                indices = [
                    int(i.strip())
                    for i in indices_str.split(",")
                    if i.strip().isdigit()
                ]
                if not indices:
                    raise ValueError("No valid indices provided")

                if any(i >= len(chunks) or i < 0 for i in indices):
                    raise ValueError("Invalid index in selection")

                # Combine selected chunk texts
                selected_text = "\n\n".join(
                    [
                        f"Chunk {i} ({chunks[i]['title']}):\n{chunks[i]['text']}"
                        for i in indices
                    ]
                )

                # LLM Prompt - Replace with your actual LLM call
                prompt = f"""Generate one question and answer based on this context:
                {selected_text}
                
                Return JSON format: {{"question": "...", "answer": "...", "chunks": [indices]}}"""

                # Example OpenAI API call (install openai package first)
                from llama_cpp import Llama

                model = Llama(LLM_PATH, device="cuda", n_ctx=4096, verbose=False)

                response = model.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    response_format={
                        "type": "json_object",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string"},
                                "answer": {"type": "string"},
                            },
                            "required": ["question", "answer"],
                        },
                    },
                )["choices"][0]["message"]["content"]

                # Parse response
                try:
                    response = json.loads(response)
                    result = {
                        "question": response["question"],
                        "answer": response["answer"],
                        "chunks": indices,
                    }

                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".json", delete=False
                    ) as f:
                        json.dump(result, f, indent=2)
                        temp_path = f.name

                    return {
                        qa_output: result,
                        qa_download: temp_path,
                    }
                except json.JSONDecodeError:
                    raise ValueError("LLM returned invalid JSON format")

            except Exception as e:
                raise gr.Error(f"QA Generation Error: {str(e)}")

        dataset_file.upload(
            read_dataset,
            inputs=dataset_file,
            outputs=[data, document_title],
        )

        document_title.change(
            select_document,
            inputs=[document_title, data],
            outputs=[document_chunks, current_chunks],
        )

        generate_btn.click(
            generate_qa,
            inputs=[selected_chunks, current_chunks],
            outputs=[qa_output, qa_download],
        )


if __name__ == "__main__":
    demo.launch(inbrowser=True)
