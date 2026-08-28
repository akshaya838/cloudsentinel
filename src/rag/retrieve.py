"""
Query the vector index built by build_index.py.

This is the retrieval half of RAG — given a natural-language question, return
the top-k most relevant chunks along with their source metadata, so an answer
generated from them can be traced back to exactly where it came from. The
generation half (feeding these chunks + the question into a local LLM via
Ollama) is the next stage after this one.

Usage:
    python src/rag/retrieve.py --query "What is the impact of Log4Shell?" --top-k 3
"""

import argparse

import chromadb

from build_index import EMBEDDING_MODEL_NAME, get_embedding_fn


def query_index(query: str, collection_name: str, persist_dir: str,
                 top_k: int = 3, embed_fn=None) -> list[dict]:
    if embed_fn is None:
        embed_fn = get_embedding_fn()

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(collection_name)

    query_embedding = embed_fn([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return hits


def main():
    parser = argparse.ArgumentParser(description="Query the RAG vector index.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--collection", type=str, default="cve_knowledge")
    parser.add_argument("--persist-dir", type=str, default="data/processed/chroma_store")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    hits = query_index(args.query, args.collection, args.persist_dir, args.top_k)

    print(f"Query: {args.query}\n")
    for i, hit in enumerate(hits, 1):
        print(f"[{i}] {hit['metadata']['source_id']} (distance={hit['distance']:.4f})")
        print(f"    {hit['text']}")
        print(f"    severity={hit['metadata'].get('cvss_severity')} "
              f"impact={hit['metadata'].get('business_impact')}\n")


if __name__ == "__main__":
    main()
