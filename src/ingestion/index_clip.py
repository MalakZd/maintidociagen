import json, sys
from pathlib import Path
import psycopg2
from PIL import Image
from sentence_transformers import SentenceTransformer

records = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
images = [r for r in records if r["type"] == "image"]
print(f"{len(images)} images à encoder")

model = SentenceTransformer("clip-ViT-B-32")
pil = [Image.open(r["path"]).convert("RGB") for r in images]
vecs = model.encode(pil, batch_size=8, normalize_embeddings=True, show_progress_bar=True)

conn = psycopg2.connect(host="127.0.0.1", port=5433, user="maintidoc",
                        password="change_me", dbname="maintidoc")
with conn, conn.cursor() as cur:
    cur.execute("""CREATE TABLE IF NOT EXISTS maintidoc_clip (
        id SERIAL PRIMARY KEY, source TEXT, page INT,
        image_path TEXT, embedding vector(512))""")
    for src in {r["source"] for r in images}:
        cur.execute("DELETE FROM maintidoc_clip WHERE source = %s", (src,))
    for r, v in zip(images, vecs):
        lit = "[" + ",".join(f"{x:.6f}" for x in v) + "]"
        cur.execute("INSERT INTO maintidoc_clip (source, page, image_path, embedding) "
                    "VALUES (%s, %s, %s, %s::vector)",
                    (r["source"], r["page"], r["path"], lit))
conn.close()
print("Indexation CLIP terminée")