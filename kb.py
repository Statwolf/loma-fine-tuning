import json
import os

from pathlib import Path
from tqdm import tqdm
from agno.knowledge.document import Document
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.qdrant import Qdrant, SearchType
from agno.db.sqlite import SqliteDb
from agno.knowledge.embedder.ollama import OllamaEmbedder
from qdrant_client import QdrantClient, models

class KnowledgeBaseLoma:
    def __init__(
        self, 
        model: str, 
        path_json: str,
        collection_name: str, 
        embedder_model: str = "nomic-embed-text:latest", 
        qdrant_url: str = "http://localhost:6333", 
        ollama_url: str = "http://localhost:6000"
    ):
        self.model = model
        self.path_json = path_json
        self.collection_name = collection_name
        
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.embedder = OllamaEmbedder(
            id=embedder_model,
            dimensions=768,
            host=ollama_url
        )

    def _parse_jsonl_to_document(self):
        file_path = Path(os.path.join(os.getcwd(), self.path_json))
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {self.path_json}")
        
        docs = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                content = item.get("text")
                file_path = Path(item.get("metadata", {}).get("file_path", ""))
                parts = file_path.parts
                if len(parts) >= 3:
                    metadata = {
                        "file_path": "/".join(parts[-3:]),
                        "type": "Alarms" if "Alarms" in str(file_path) else "Manuals",
                        "machine": parts[-3],
                        "language": parts[-2],
                        "base_file": parts[-1]
                    }
                    docs.append(Document(content=content, meta_data=metadata))
        
        return docs

    def generate_kb(self):
        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=768, 
                    distance=models.Distance.COSINE
                )
            )
            print(f"Collection {self.collection_name} created.")
        else:
            print(f"Collection {self.collection_name} already exists.")

    def create_plain_kb(self, base_dir: str):
        documents = self._parse_jsonl_to_document()
        if not documents:
            print("No documents in the specified JSONL.")
            return

        for doc in tqdm(documents):
            m = doc.meta_data
            dir_path = Path(base_dir) / m["type"] / m["machine"] / m["language"]
            file_name = Path(m["base_file"]).stem + ".txt"
            full_path = dir_path / file_name

            os.makedirs(dir_path, exist_ok=True)

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(doc.content)
        
        print("Plain KB created.")

    def populate_kb(self):
        documents = self._parse_jsonl_to_document()
        if not documents:
            print("No documents in the specified JSONL.")
            return

        manuals = [doc for doc in documents if doc.meta_data.get("type") == "Manuals"]

        kb = self.load_kb()
        for doc in manuals:
            kb.insert(
                name=doc.meta_data.get("file_path"),
                text_content=doc.content,
                metadata={
                    "machine": doc.meta_data.get("machine")
                },
                skip_if_exists=True
            )

        print(f"Manuals loaded in {self.collection_name}")

    def load_kb(self, content_db_path: str = "./tmp/knowledge.db", **kw_args):
        vector_db = Qdrant(
            collection=self.collection_name,
            #client=self.qdrant_client,
            embedder=self.embedder,
            search_type=SearchType.hybrid
        )

        contents_db = SqliteDb(
            knowledge_table="knowledge_contents",
            db_file=content_db_path
        )

        return Knowledge(
            vector_db=vector_db,
            contents_db=contents_db,
            **kw_args
        )