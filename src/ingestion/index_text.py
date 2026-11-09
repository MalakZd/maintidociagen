import json, sys
from pathlib import Path

from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

# 1) Modèle d'embeddings open source (multilingue, 1024 dimensions)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
Settings.llm = None  # pas de LLM nécessaire pour indexer

# 2) Base vectorielle PGVector
vector_store = PGVectorStore.from_params(
    database="maintidoc",
    host="127.0.0.1",
    port=5433,
    user="maintidoc",
    password="change_me",
    table_name="maintidoc_text",   # LlamaIndex crée la table "data_maintidoc_text"
    embed_dim=1024,
)

def build_nodes(records_path: str):
    records = json.loads(Path(records_path).read_text(encoding="utf-8"))
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    nodes = []
    for r in records:
        meta = {"source": r["source"], "page": r["page"], "type": r["type"]}
        if r["type"] == "text":
            doc = Document(text=r["content"], metadata=meta)

            nodes.extend(splitter.get_nodes_from_documents([doc]))
        elif r["type"] == "table":
            # un tableau = un seul chunk, on ne le coupe pas
            nodes.append(TextNode(text=r["content"], metadata=meta))
    return nodes

if __name__ == "__main__":
    nodes = build_nodes(sys.argv[1])
    print(f"{len(nodes)} chunks à indexer")
    storage = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(nodes, storage_context=storage, show_progress=True)
    print("Indexation terminée")