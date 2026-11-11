import psycopg2
from sentence_transformers import SentenceTransformer

_text_model = None

def search_images_clip(query: str, k: int = 3):
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer(
            "sentence-transformers/clip-ViT-B-32-multilingual-v1")
    v = _text_model.encode(query, normalize_embeddings=True)
    lit = "[" + ",".join(f"{x:.6f}" for x in v) + "]"
    conn = psycopg2.connect(host="127.0.0.1", port=5433, user="maintidoc",
                            password="change_me", dbname="maintidoc")
    with conn, conn.cursor() as cur:
        cur.execute("""SELECT source, page, image_path,
                              1 - (embedding <=> %s::vector) AS score
                       FROM maintidoc_clip
                       ORDER BY embedding <=> %s::vector LIMIT %s""",
                    (lit, lit, k))
        rows = cur.fetchall()
    conn.close()
    return rows

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "panneau de commande électronique"
    for source, page, path, score in search_images_clip(q):
        print(f"p.{page} score={score:.2f}  {path}")