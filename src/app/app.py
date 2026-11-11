import os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTHONUTF8"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))

import streamlit as st

import subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def purge_source(stem: str):
    """Supprime les anciens chunks de ce document pour éviter les doublons."""
    import psycopg2
    try:
        c = psycopg2.connect(host="127.0.0.1", port=5433, user="maintidoc",
                             password="change_me", dbname="maintidoc")
        with c, c.cursor() as cur:
            cur.execute("DELETE FROM data_maintidoc_text WHERE metadata_->>'source' = %s", (stem,))
        c.close()
    except Exception as e:
        st.warning(f"Purge impossible (premier import ?) : {e}")

with st.sidebar:
    st.header("Ajouter un document")
    up = st.file_uploader("PDF à indexer", type="pdf")
    if up and st.button("Lancer l'indexation"):
        pdf_path = ROOT / "data" / "pdfs" / up.name
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(up.getbuffer())
        purge_source(pdf_path.stem)
        log = ROOT / "data" / f"ingest_{pdf_path.stem}.log"
        env = {**os.environ, "PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1"}
        proc = subprocess.Popen(
            [sys.executable, "src/ingestion/run_pipeline.py", f"data/pdfs/{up.name}"],
            cwd=ROOT, env=env, stdout=open(log, "w", encoding="utf-8"),
            stderr=subprocess.STDOUT)
        st.session_state["ingest"] = {"pid": proc.pid, "log": str(log),
                                      "name": up.name, "t0": time.time(), "proc": proc}

    ing = st.session_state.get("ingest")
    if ing:
        done = ing["proc"].poll() is not None
        mins = int((time.time() - ing["t0"]) / 60)
        if done and ing["proc"].returncode == 0:
            st.success(f"{ing['name']} indexé ({mins} min).")
        elif done:
            st.error("Échec de l'indexation, voir le journal ci-dessous.")
        else:
            st.info(f"Indexation de {ing['name']} en cours ({mins} min)...")
        if st.button("Actualiser l'état"):
            st.rerun()
        with st.expander("Journal"):
            st.code(Path(ing["log"]).read_text(encoding="utf-8", errors="ignore")[-1500:])

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