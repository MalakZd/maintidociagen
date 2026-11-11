import os, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTHONUTF8"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
import agent

TESTS = [
    ("Comment vidanger l'huile du compresseur ?", "p.16"),
    ("Le compresseur ne démarre pas, quelles sont les causes et les remèdes ?", "p.30"),
    ("Que faire en cas d'alarme ?", "p.29"),
    ("Quand faut-il remplacer les cartouches de filtre ?", "p.18"),
    ("Comment choisir l'unité de température (Celsius ou Fahrenheit) ?", "p.27"),
    ("Montre-moi le panneau de commande électronique", "p.24 à 28 (images)"),
    ("Quelle est la capitale de la France ?", "REFUS attendu"),
]

lines = ["| Question | Attendu | Outils | Pages consultées | Temps (s) | Réponse (début) | Verdict |",
         "|---|---|---|---|---|---|---|"]
for q, expected in TESTS:
    t0 = time.time()
    text, steps, sources = agent.run_agent(q, verbose=False)
    dt = time.time() - t0
    tools = ", ".join(n for n, _ in steps) or "aucun"
    pages = ", ".join(f"p.{p}" for p, _, _ in sources)
    short = text.replace("\n", " ").replace("|", "/")[:150]
    lines.append(f"| {q} | {expected} | {tools} | {pages} | {dt:.0f} | {short} | à remplir |")
    print(f"OK : {q} ({dt:.0f} s)", flush=True)

os.makedirs("docs", exist_ok=True)
open("docs/resultats_tests.md", "w", encoding="utf-8").write("\n".join(lines))
print("Résultats écrits dans docs/resultats_tests.md")