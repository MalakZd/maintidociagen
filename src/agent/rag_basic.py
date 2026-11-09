import sys
import ollama
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

LLM = "qwen3.5:latest"
TOP_K = 5

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
Settings.llm = None

vs = PGVectorStore.from_params(
    database="maintidoc", host="127.0.0.1", port=5433,
    user="maintidoc", password="change_me",
    table_name="maintidoc_text", embed_dim=1024)
retriever = VectorStoreIndex.from_vector_store(vs).as_retriever(similarity_top_k=TOP_K)

SYSTEM = """Tu es MAINTiDOC, assistant documentaire pour techniciens de maintenance.
Règles :
- Réponds en français, uniquement à partir des extraits fournis. N'ajoute aucune information externe.
- Chaque phrase ou étape se termine par la référence de l'extrait exact qui la soutient, par exemple [2]. N'utilise [1] que si l'information vient vraiment de l'extrait 1.
- Pour un tableau de pannes (PROBLEME / CAUSE / REMEDE), reprends chaque cause avec SON remède, sans mélanger les lignes ni inventer de lien entre elles.
- Ne répète jamais la même vérification.
- Si les extraits ne contiennent pas la réponse, écris : « Je ne trouve pas cette information dans les documents. »
- Les extraits « description automatique d'image » ne servent jamais pour des détails (numéros de pièces, couleurs, voyants).
- N'ajoute aucun conseil de sécurité qui ne figure pas dans les extraits, sauf cette phrase finale : « En cas de doute, faites intervenir un technicien habilité. »
- Réponse courte et structurée."""

def build_context(nodes):
    blocks = []
    for i, n in enumerate(nodes, 1):
        m = n.node.metadata
        tag = "description automatique d'image" if m["type"] == "image" else m["type"]

        blocks.append(f"[{i}] (page {m['page']}, {tag})\n{n.node.text[:1500]}")
    return "\n\n".join(blocks)

def answer(question):
    nodes = retriever.retrieve(question)
    context = build_context(nodes)
    resp = ollama.chat(
        model=LLM, think=False,
        options={"temperature": 0.1, "num_ctx": 6144},
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user",
                   "content": f"Extraits :\n{context}\n\nQuestion : {question}"}])
    return resp["message"]["content"].strip(), nodes

def ask(question):
    nodes = retriever.retrieve(question)
    context = build_context(nodes)
    resp = ollama.chat(
        model=LLM,
        think=False,
        options={"temperature": 0.1, "num_ctx": 6144},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": f"Extraits :\n{context}\n\nQuestion : {question}"},
        ])
    print("\n=== RÉPONSE ===")
    print(resp["message"]["content"].strip())
    print("\n=== SOURCES ===")
    for i, n in enumerate(nodes, 1):
        m = n.node.metadata
        print(f"[{i}] {m['source']} p.{m['page']} ({m['type']}) score={n.score:.2f}")

if __name__ == "__main__":
    ask(" ".join(sys.argv[1:]) or "Comment vidanger l'huile du compresseur ?")