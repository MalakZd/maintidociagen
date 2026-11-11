import json, sys
from pathlib import Path
import ollama

MODEL = "qwen2.5vl:7b"
PRIORITY_PAGES = [24, 25, 26, 27, 28, 29, 11, 12, 13]   # panneau électronique, installation
MIN_BYTES = 10_000                                        # ignore logos et fonds

PROMPT = """Tu analyses une figure extraite d'un manuel de maintenance de compresseur.
Décris UNIQUEMENT ce que tu vois, en français, en 3 phrases courtes maximum.
Règles strictes :
- Indique le type de figure (schéma, photo, tableau, panneau de commande).
- Cite un nom de pièce ou un numéro seulement s'il est lisible sans ambiguïté.
- Mentionne un voyant ou une couleur seulement si tu les vois clairement.
- N'utilise jamais « probablement », « semble » ou « pourrait ». Si un détail est incertain ou illisible, n'en parle pas.
- N'interprète pas le fonctionnement. Ne répète jamais une phrase.
Contexte (texte de la même page, à ne pas recopier) :
{context}"""

OPTIONS = {"temperature": 0, "num_predict": 200, "repeat_penalty": 1.3}

def is_degenerate(text: str) -> bool:
    words = text.lower().split()
    if len(words) < 8 or len(text) > 1200:
        return True
    grams = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
    return len(set(grams)) / max(len(grams), 1) < 0.6

def page_context(records, source, page, n=500):
    for r in records:
        if r["source"] == source and r["page"] == page and r["type"] == "text":
            return r["content"][:n]

    return "(aucun texte)"

def describe(path, context):
    for _ in range(2):
        resp = ollama.chat(
            model=MODEL, options=OPTIONS,
            messages=[{"role": "user", "content": PROMPT.format(context=context),
                       "images": [path]}])
        text = resp["message"]["content"].strip()
        if not is_degenerate(text):
            return text
    return None

def main(records_path, limit=None):
    rec_path = Path(records_path)
    records = json.loads(rec_path.read_text(encoding="utf-8"))
    out_path = rec_path.parent / "image_descriptions.json"
    done = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}

    images = [r for r in records if r["type"] == "image"
              and Path(r["path"]).exists() and Path(r["path"]).stat().st_size >= MIN_BYTES]
    images.sort(key=lambda r: (r["page"] not in PRIORITY_PAGES, r["page"]))
    if limit:
        images = images[:limit]
    print(f"{len(images)} images à décrire", flush=True)

    for i, r in enumerate(images, 1):
        if done.get(r["path"], {}).get("description"):
            continue
        print(f"[{i}/{len(images)}] p.{r['page']} {r['path']}", flush=True)
        done[r["path"]] = {
            "source": r["source"], "page": r["page"],

            "description": describe(r["path"], page_context(records, r["source"], r["page"])),
        }
        out_path.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
    print("DESCRIPTIONS TERMINEES", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)