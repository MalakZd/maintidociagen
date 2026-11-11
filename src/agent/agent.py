import sys
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

LLM = "qwen3.5:latest"
MAX_STEPS = 4

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
Settings.llm = None

vs = PGVectorStore.from_params(
    database="maintidoc", host="127.0.0.1", port=5433,
    user="maintidoc", password="change_me",
    table_name="maintidoc_text", embed_dim=1024)
index = VectorStoreIndex.from_vector_store(vs)

def make_retriever(doc_type: str, k: int):
    f = MetadataFilters(filters=[MetadataFilter(key="type", value=doc_type)])
    return index.as_retriever(similarity_top_k=k, filters=f)

RETRIEVERS = {
    "text": make_retriever("text", 4),
    "table": make_retriever("table", 3),
    "image": make_retriever("image", 3),
}

SOURCES = []  # références collectées pendant l'exécution

def _search(doc_type: str, query: str) -> str:
    nodes = RETRIEVERS[doc_type].retrieve(query)
    nodes = [n for n in nodes if n.score is not None and n.score >= 0.40]
    if not nodes:
        return f"Aucun résultat de type {doc_type}."
    out = []
    for n in nodes:
        m = n.node.metadata
        SOURCES.append((m["page"], m["type"], n.score))
        out.append(f"(page {m['page']}, {m['type']}, score {n.score:.2f})\n{n.node.text[:1200]}")
    return "\n\n".join(out)
    

@tool
def search_text(query: str) -> str:
    """Cherche dans le texte du manuel : procédures, explications, alarmes, sécurité."""
    return _search("text", query)

@tool
def search_tables(query: str) -> str:
    """Cherche dans les tableaux du manuel : pannes causes/remèdes, fréquences d'entretien, pièces."""
    return _search("table", query)

# @tool
# def search_images(query: str) -> str:
#     """Cherche dans les descriptions automatiques d'images (schémas, panneaux). Peu fiable pour les détails."""
#     return _search("image", query)

@tool
def search_images(query: str) -> str:
    """Cherche des schémas, photos et panneaux de commande, par ressemblance visuelle et par description."""
    import os
    out = []
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "retrieval"))
        from clip_search import search_images_clip
        for _, page, path, score in search_images_clip(query, 3):
            SOURCES.append((page, "image", float(score)))
            out.append(f"Image page {page} (similarité CLIP {score:.2f}) : {path}")
    except Exception as e:
        out.append(f"(Recherche CLIP indisponible : {type(e).__name__})")
    desc = _search("image", query)
    if not desc.startswith("Aucun"):
        out.append("Descriptions automatiques (peu fiables, jamais pour les détails) :\n" + desc)
    if len(out) == 1 and out[0].startswith("(Recherche CLIP"):
        return "Aucune image trouvée."
    return "\n".join(out) + "\nIndique les pages et invite à consulter les images."

TOOLS = [search_text, search_tables, search_images]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

SYSTEM = """Tu es MAINTiDOC, assistant documentaire pour techniciens de maintenance.
Tu disposes d'outils de recherche dans le manuel : utilise-les AVANT de répondre.
Règles :
- Choisis l'outil adapté : tableaux pour pannes et fréquences, texte pour procédures.
- Tu peux faire plusieurs recherches si la première ne suffit pas.
- Réponds en français, uniquement à partir des résultats des outils. N'ajoute aucune information externe.
- Cite la page après chaque information, par exemple (p.16).
- Pour un tableau de pannes, reprends chaque cause avec SON remède sans mélanger les lignes.
- Les descriptions d'images ne servent jamais pour des détails (numéros, couleurs, voyants).
- Si les résultats ne contiennent pas la réponse : « Je ne trouve pas cette information dans les documents. »
- Termine par : « En cas de doute, faites intervenir un technicien habilité. »"""

llm = ChatOllama(model=LLM, temperature=0, num_ctx=6144).bind_tools(TOOLS)

def run_agent(question: str, verbose: bool = True):
    SOURCES.clear()
    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=question)]
    steps = []
    for _ in range(MAX_STEPS):
        ai = llm.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            return ai.content.strip(), steps, sorted(set(SOURCES))
        for call in ai.tool_calls:
            steps.append((call["name"], call["args"]))
            if verbose:
                print(f"  -> outil : {call['name']} {call['args']}", flush=True)
            result = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    # limite atteinte : on force une réponse sans outils
    final = ChatOllama(model=LLM, temperature=0, num_ctx=6144).invoke(messages)
    return final.content.strip(), steps, sorted(set(SOURCES))

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Le compresseur ne démarre pas, quelles sont les causes et les remèdes ?"
    text, steps, sources = run_agent(q)
    print("\n=== RÉPONSE ===\n" + text)
    print("\n=== OUTILS UTILISÉS ===")
    for name, args in steps:
        print(f"- {name} {args}")
    print("\n=== PAGES CONSULTÉES ===")
    for page, typ, score in sources:
        print(f"p.{page} ({typ}) score={score:.2f}")