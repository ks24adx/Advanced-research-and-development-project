
import argparse
import numpy as np
import pandas as pd
import faiss
import pickle
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

CTX_ENCODER   = "facebook-dpr-ctx_encoder-single-nq-base"
BATCH_SIZE    = 512
EMBEDDING_DIM = 768

parser = argparse.ArgumentParser()
parser.add_argument("--passages",     required=True)
parser.add_argument("--output",       required=True)
parser.add_argument("--max_passages", type=int, default=None)
args = parser.parse_args()

output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading passages...")
df = pd.read_csv(args.passages, sep="\t", on_bad_lines="skip")
df.columns = ["id", "text", "title"]
if args.max_passages:
    df = df.head(args.max_passages)
passages = (df["title"] + ". " + df["text"]).tolist()
print(f"  {len(passages):,} passages loaded.")

print(f"Loading DPR context encoder...")
model = SentenceTransformer(CTX_ENCODER)

print("Encoding passages...")
all_embeddings = []
for i in tqdm(range(0, len(passages), BATCH_SIZE), desc="Encoding"):
    batch = passages[i : i + BATCH_SIZE]
    embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    all_embeddings.append(embeddings)

all_embeddings = np.vstack(all_embeddings).astype("float32")
print(f"  Embeddings shape: {all_embeddings.shape}")

print("Building FAISS index...")
index = faiss.IndexFlatIP(EMBEDDING_DIM)
index.add(all_embeddings)
print(f"  Index contains {index.ntotal:,} vectors.")

index_path    = output_dir / "wiki.index"
passages_path = output_dir / "passages.pkl"

faiss.write_index(index, str(index_path))
print(f"  FAISS index saved -> {index_path}")

with open(passages_path, "wb") as f:
    pickle.dump(passages, f)
print(f"  Passages saved    -> {passages_path}")

print("Done! Index is ready.")
