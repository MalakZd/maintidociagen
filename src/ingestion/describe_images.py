import json, re, sys
from pathlib import Path
import ollama

MODEL = "qwen2.5vl:7b"

PROMPT = """Tu analyses une figure extraite d'un manuel de maintenance de compresseur.
Décris UNIQUEMENT ce que tu vois, en français, en 3 à 5 phrases courtes.
Règles strictes :
- Indique le type de figure (schéma, photo, tableau, panneau de commande).
- Ne cite un nom de pièce ou un numéro que s'il est lisible sans ambiguïté sur l'image.
- Ne mentionne un voyant ou une couleur que si tu le vois clairement.
- Si un détail est illisible ou incertain, écris « illisible » au lieu de deviner.
- Ne répète jamais une phrase ou une liste.
- N'utilise jamais « probablement », « semble », « pourrait » : si tu n'es pas sûr, n'en parle pas.
- Ne décris pas les couleurs, sauf pour un voyant ou un panneau de commande.
- N'interprète pas le fonctionnement : décris seulement ce qui est dessiné et les mots lisibles.
- Limite-toi à 3 phrases.
Contexte (texte de la même page du manuel, à ne pas recopier) :
{context}"""

OPTIONS = {"temperature": 0, "num_predict": 220, "repeat_penalty": 1.3}

def is_degenerate(text: str) -> bool:
    words = text.lower().split()
    if len(words) < 8 or len(text) > 1500:
        return True
    grams = [" ".join(words[i:i+4]) for i in range(len(words) - 3)]
    return len(set(grams)) / max(len(grams), 1) < 0.6

def page_context(records, source, page, n=600):
    for r in records:
        if r["source"] == source and r["page"] == page and r["type"] == "text":
            return r["content"][:n]
    return "(aucun texte)"


def describe(image_path, context):
    for _ in range(2):  # une seconde tentative si dégénéré
        resp = ollama.chat(
            model=MODEL, options=OPTIONS,
            messages=[{"role": "user",
                       "content": PROMPT.format(context=context),
                       "images": [image_path]}])
        text = resp["message"]["content"].strip()
        if not is_degenerate(text):
            return text
    return None  # échec : à revoir à la main

def main(records_path, limit=None):
    rec_path = Path(records_path)
    records = json.loads(rec_path.read_text(encoding="utf-8"))
    out_path = rec_path.parent / "image_descriptions.json"
    done = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}

    images = [r for r in records if r["type"] == "image"]
    if limit:
        images = images[:limit]

    for i, r in enumerate(images, 1):
        if r["path"] in done and done[r["path"]]["description"]:
            continue
        print(f"[{i}/{len(images)}] p.{r['page']} {r['path']}", flush=True)
        ctx = page_context(records, r["source"], r["page"])
        desc = describe(r["path"], ctx)
        done[r["path"]] = {"source": r["source"], "page": r["page"],
                           "description": desc, "auto_generated": True}
        out_path.write_text(json.dumps(done, ensure_ascii=False, indent=1),
                            encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)