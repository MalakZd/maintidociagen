import json, sys
from pathlib import Path
import pymupdf as fitz

MIN_SIZE = 150  # px : ignore les fragments

def extract(pdf_path: str, out_dir: str = "data/extracted"):
    doc_name = Path(pdf_path).stem
    img_dir = Path(out_dir) / doc_name / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    records = []

    doc = fitz.open(pdf_path)
    for pno, page in enumerate(doc, start=1):
        base = {"source": doc_name, "page": pno}

        # 1) Texte
        text = page.get_text().strip()
        if len(text) > 100:
            records.append({**base, "type": "text", "content": text})

        # 2) Tableaux -> markdown
        for t_i, tab in enumerate(page.find_tables().tables):
            md = tab.to_markdown()
            if md.strip():
                records.append({**base, "type": "table",
                                "table_id": t_i, "content": md})

        # 3) Images utiles uniquement
        for i_i, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)

            if pix.width < MIN_SIZE or pix.height < MIN_SIZE:
                continue
            if pix.n - pix.alpha > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            path = img_dir / f"p{pno:03d}_{i_i}.png"
            pix.save(path)
            records.append({**base, "type": "image", "path": str(path)})

    out = Path(out_dir) / doc_name / "records.json"
    out.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    print(Counter(r["type"] for r in records))

if __name__ == "__main__":
    extract(sys.argv[1])