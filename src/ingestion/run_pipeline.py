import subprocess, sys
from pathlib import Path

pdf = sys.argv[1]
stem = Path(pdf).stem

print(f"[1/2] Extraction de {pdf}", flush=True)
subprocess.run([sys.executable, "src/ingestion/extract.py", pdf], check=True)

print("[2/2] Indexation (embeddings BGE-M3, long sur CPU)", flush=True)
subprocess.run([sys.executable, "src/ingestion/index_text.py",
                f"data/extracted/{stem}/records.json"], check=True)

print("PIPELINE TERMINE", flush=True)