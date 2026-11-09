import json, sys
from pathlib import Path
import psycopg2

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

try:
    psycopg2.connect(host="127.0.0.1", port=5433, user="maintidoc",
                     password="change_me", dbname="maintidoc").close()
    print("Connexion PostgreSQL OK")
except Exception as e:
    sys.exit(f"Connexion PostgreSQL impossible : {e}")

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
Settings.llm = None

vs = PGVectorStore.from_params(
    database="maintidoc", host="127.0.0.1", port=5433,
    user="maintidoc", password="change_me",
    table_name="maintidoc_text", embed_dim=1024)

desc_path = Path(sys.argv[1])
data = json.loads(desc_path.read_text(encoding="utf-8"))

nodes = []
for img_path, d in data.items():
    if not d.get("description"):
        continue  # on saute les descriptions échouées
    nodes.append(TextNode(
        text=d["description"],
        metadata={"source": d["source"], "page": d["page"],
                  "type": "image", "image_path": img_path}))

print(f"{len(nodes)} descriptions à indexer")
VectorStoreIndex(nodes, storage_context=StorageContext.from_defaults(vector_store=vs),
                 show_progress=True)
print("Indexation des images terminée")