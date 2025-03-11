
from dataclasses import dataclass
from typing import TypedDict


class Metadata(TypedDict):
    created_at: str

@dataclass
class Chunk():
    id: int
    doc_id: int
    page: int 
    text: str

@dataclass
class Document():
    id: int
    source: str
    metadata: Metadata