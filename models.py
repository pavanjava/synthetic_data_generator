from dataclasses import dataclass


@dataclass
class GeneratorParams:
    file_path: str
    llm_choice: str
    api_key: str
    chunk_size: int
    chunk_overlap: int
    questions_per_chunk: int
    use_vectordb: bool