"""
Build a ChromaDB vector index from chunked CVE/posture/audit records.

Uses a multilingual sentence-transformer so English and Tamil text land in
the same embedding space — required for the Tamil Q&A stage later without
re-architecting this layer. On first run this downloads the model from
HuggingFace, so it needs an internet connection once; after that it runs
fully offline, consistent with the project's local-mode design goal.

Usage:
    python src/rag/build_index.py \
        --in data/processed/cves_classified.json \
        --collection cve_knowledge \
        --persist-dir data/processed/chroma_store
"""

import argparse
import json
from pathlib import Path

import chromadb

from chunking import chunk_record

# Multilingual model: supports English + Tamil in one shared embedding space.
# 384-dim output, good balance of quality vs. speed for a local-mode deployment.
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def get_embedding_fn():
    """Lazy import so this module can be tested without sentence-transformers
    installed (e.g. when only exercising the ChromaDB plumbing)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def embed(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, show_progress_bar=False).tolist()

    return embed


def build_index(records: list[dict], source_type: str, collection_name: str,
                 persist_dir: str, embed_fn=None) -> chromadb.Collection:
    if embed_fn is None:
        embed_fn = get_embedding_fn()

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks = []
    for record in records:
        all_chunks.extend(chunk_record(record, source_type))

    if not all_chunks:
        print("No chunks produced — nothing to index.")
        return collection

    texts = [c["text"] for c in all_chunks]
    embeddings = embed_fn(texts)

    # Chroma metadata values must be str/int/float/bool — flatten lists like
    # business_impact into a comma-joined string.
    metadatas = []
    for c in all_chunks:
        meta = dict(c)
        meta.pop("text")
        meta["business_impact"] = ",".join(meta.get("business_impact") or [])
        metadatas.append({k: v for k, v in meta.items() if v is not None})

    collection.upsert(
        ids=[c["chunk_id"] for c in all_chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"Indexed {len(all_chunks)} chunks from {len(records)} records into "
          f"collection '{collection_name}' at {persist_dir}")
    return collection


def main():
    parser = argparse.ArgumentParser(description="Build the RAG vector index.")
    parser.add_argument("--in", dest="in_path", type=str, required=True)
    parser.add_argument("--source-type", type=str, default="cve",
                         help="cve | posture | audit_log")
    parser.add_argument("--collection", type=str, default="cve_knowledge")
    parser.add_argument("--persist-dir", type=str, default="data/processed/chroma_store")
    args = parser.parse_args()

    records = json.loads(Path(args.in_path).read_text())
    build_index(records, args.source_type, args.collection, args.persist_dir)


if __name__ == "__main__":
    main()
