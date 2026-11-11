import os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTHONUTF8"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))

import streamlit as st

st.set_page_config(page_title="MAINTiDOC", page_icon="🔧", layout="wide")

@st.cache_resource(show_spinner="Chargement des modèles (1re fois : quelques minutes)...")
def load_agent():
    import agent
    return agent

st.title("🔧 MAINTiDOC")
st.caption("Assistant documentaire pour la maintenance. Il ne pilote aucun équipement "
           "et ne remplace pas un technicien habilité.")

with st.sidebar:
    st.header("Document indexé")
    st.write("**Compresseur Pacific (Nardi)**, 59 pages")
    st.divider()
    for ex in ["Comment vidanger l'huile du compresseur ?",
               "Le compresseur ne démarre pas, quelles sont les causes et les remèdes ?",
               "Quand remplacer les cartouches de filtre ?",
               "Que faire en cas d'alarme ?"]:
        if st.button(ex, use_container_width=True):
            st.session_state["q"] = ex

question = st.text_input("Posez votre question technique", key="q")

if st.button("Rechercher", type="primary") and question.strip():

    ag = load_agent()
    with st.spinner("L'agent choisit ses outils et rédige la réponse..."):
        text, steps, sources = ag.run_agent(question, verbose=False)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Réponse")
        st.markdown(text)
    with col2:
        st.subheader("Raisonnement de l'agent")
        for name, args in steps:
            st.code(f"{name}({args.get('query', args)})")
        st.subheader("Pages consultées")
        for page, typ, score in sources:
            st.write(f"p.{page} ({typ}) · score {score:.2f}")