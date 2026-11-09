import os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTHONUTF8"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))

import streamlit as st

@st.cache_resource(show_spinner="Chargement des modèles...")
def load():
    import rag_basic
    return rag_basic

st.set_page_config(page_title="MAINTiDOC", page_icon="🔧")
st.title("🔧 MAINTiDOC")
st.caption("Assistant documentaire pour la maintenance. Il ne remplace pas un technicien habilité.")

question = st.text_input("Posez votre question technique",
                         placeholder="Ex. : Comment vidanger l'huile du compresseur ?")

if st.button("Rechercher") and question.strip():
    rag = load()
    with st.spinner("Recherche et génération en cours..."):
        text, nodes = rag.answer(question)
    st.subheader("Réponse")
    st.markdown(text)
    st.subheader("Sources")
    for i, n in enumerate(nodes, 1):
        m = n.node.metadata
        with st.expander(f"[{i}] {m['source']} — page {m['page']} ({m['type']}) — score {n.score:.2f}"):
            st.text(n.node.text[:1200])