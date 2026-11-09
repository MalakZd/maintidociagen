from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
Settings.llm = None

vs = PGVectorStore.from_params(
    database="maintidoc",
    host="127.0.0.1",
    port=5433,
    user="maintidoc",
    password="change_me",
    table_name="maintidoc_text",
    embed_dim=1024,
)

retriever = VectorStoreIndex.from_vector_store(vs).as_retriever(similarity_top_k=4)

for q in [
    "Comment vidanger l'huile du compresseur ?",
    "Que faire en cas d'alarme ?",
    "Le compresseur ne démarre pas, quelles sont les causes ?",
    "Quand remplacer les cartouches de filtre ?",
]:
    print(f"\n=== {q}")
    for n in retriever.retrieve(q):
        m = n.node.metadata
        print(f"  p.{m['page']} [{m['type']}] score={n.score:.2f} | {n.node.text[:80]!r}")