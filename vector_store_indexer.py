import os
import uuid
from qdrant_client import QdrantClient, models
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class VectorStoreIndexer:
    def __init__(self):
        self.client = QdrantClient(url=os.environ.get("QDRANT_URL"), api_key=os.environ.get("QDRANT_API_KEY"))
        self.COLLECTION_NAME = os.environ.get("COLLECTION_NAME")
        self.model = HuggingFaceEmbedding(
            model_name="llamaindex/vdr-2b-multi-v1",
            device="mps",  # "mps" for mac, "cuda" for nvidia GPUs
            trust_remote_code=True,
        )
        if not self.client.collection_exists(collection_name=self.COLLECTION_NAME):
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config={
                    "chunk": models.VectorParams(size=1536, distance=models.Distance.COSINE),
                }
            )

    def index_data(self, chunk: str):
        print(f"in indexer: {chunk}")
        documents = [{"chunk": chunk}]
        text_embeddings = self.model.get_text_embedding_batch([doc["chunk"] for doc in documents])
        self.client.upload_points(
            collection_name=self.COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "chunk": text_embeddings[idx],
                    },
                    payload=doc
                )
                for idx, doc in enumerate(documents)
            ]
        )
