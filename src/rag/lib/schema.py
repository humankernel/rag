from dataclasses import dataclass
from typing import TypedDict


class Metadata(TypedDict):
    created_at: str


@dataclass
class Chunk:
    id: str
    doc_id: str
    page: int
    text: str


@dataclass
class Document:
    id: str
    source: str
    metadata: Metadata
